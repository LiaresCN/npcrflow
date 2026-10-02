"""Effective-DOF proxy screening with optional season optimization."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import ScreeningConfig
from .data import annualize_record
from .records import ProxyCollection, ProxyRecord
from .time_windows import observed_window_means, window_table


def _effectively_constant(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    return bool(float(np.ptp(values)) <= 32.0 * np.finfo(float).eps * scale)


def _lag1_on_consecutive(series: pd.Series, year_step: int = 1) -> float:
    series = series.dropna().sort_index()
    if len(series) < 3:
        return 0.0
    index = series.index.to_numpy(dtype=int)
    values = series.to_numpy(dtype=float)
    adjacent = np.diff(index) == year_step
    if adjacent.sum() < 2:
        return 0.0
    first = values[:-1][adjacent]
    second = values[1:][adjacent]
    if _effectively_constant(first) or _effectively_constant(second):
        return 0.0
    result = float(np.corrcoef(first, second)[0, 1])
    return float(np.clip(result, -0.99, 0.99)) if np.isfinite(result) else 0.0


def effective_sample_size(x: pd.Series, y: pd.Series, year_step: int = 1) -> float:
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    n = len(joined)
    if n < 3:
        return float(n)
    product = (_lag1_on_consecutive(joined["x"], year_step)
               * _lag1_on_consecutive(joined["y"], year_step))
    value = n * (1.0 - product) / (1.0 + product)
    return float(np.clip(value, 3.0, float(n)))


def correlation_with_effective_dof(
    x: pd.Series, y: pd.Series, year_step: int = 1,
) -> tuple[float, float, float, int]:
    if year_step < 1:
        raise ValueError("year_step must be positive")
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    n = len(joined)
    x_values = joined["x"].to_numpy(float)
    y_values = joined["y"].to_numpy(float)
    if n < 3 or _effectively_constant(x_values) or _effectively_constant(y_values):
        return np.nan, np.nan, float(n), n
    r = float(np.corrcoef(x_values, y_values)[0, 1])
    n_eff = effective_sample_size(joined["x"], joined["y"], year_step)
    denominator = max(1.0 - r * r, np.finfo(float).eps)
    t_value = r * np.sqrt(max(n_eff - 2.0, 0.0) / denominator)
    p_value = float(2.0 * stats.t.sf(abs(t_value), max(n_eff - 2.0, 1.0)))
    return r, p_value, n_eff, n


def _linear_detrend(series: pd.Series) -> pd.Series:
    finite = series.dropna()
    if len(finite) < 3:
        return series
    x = finite.index.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, finite.to_numpy(dtype=float), 1)
    result = series.copy()
    result.loc[finite.index] = finite - (slope * x + intercept)
    return result


def _holm(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(values, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values, preserving input order."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(values, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _candidate_months(record: ProxyRecord, config: ScreeningConfig) -> tuple[tuple[int, ...], ...]:
    if config.season_mode == "fixed" or record.resolution >= 0.75:
        return (config.months,)
    unique: list[tuple[int, ...]] = []
    for months in config.season_candidates:
        if months not in unique:
            unique.append(months)
    return tuple(unique)


def _native_screening_pair(
    record: ProxyRecord, target: pd.Series, config: ScreeningConfig,
) -> tuple[pd.Series, pd.Series, int]:
    """Complete observational windows only, using actual native proxy values.

    Anchor windows at the declared screening start (or first target year).
    Missing target years and partial terminal windows invalidate a window:
    held-out years can never leak into a training-only screening calculation.
    Records beyond the declared bins use ceil(native resolution), but this
    does not make them eligible for the <=10-year reconstruction layers.
    """
    width = next((value for value in config.native_window_bins
                  if record.resolution <= value), int(np.ceil(record.resolution)))
    finite_target = target.dropna().sort_index()
    if finite_target.empty:
        empty = pd.Series(dtype=float)
        return empty, empty.copy(), width
    start, end = (config.period if config.period is not None
                  else (int(finite_target.index.min()), int(finite_target.index.max())))
    windows = window_table(start, end, width)
    local = finite_target.loc[start:end]
    groups = start + ((local.index.to_numpy(int) - start) // width) * width
    means = local.groupby(groups).mean().reindex(windows.index)
    counts = local.groupby(groups).count().reindex(windows.index, fill_value=0)
    complete = (windows.window_year_count == width) & (counts == width)
    # Values outside the declared period must not sneak in via half-year edges.
    inside_period = (record.time >= start) & (record.time < end + 1)
    proxy = observed_window_means(
        record.time[inside_period], record.value[inside_period], windows,
    ).where(complete)
    return proxy, means.where(complete), width


def _screen_one(record: ProxyRecord, target: pd.Series, config: ScreeningConfig) -> dict[str, object]:
    required_overlap = config.min_overlap
    if (
        config.low_resolution_min_overlap is not None
        and np.isfinite(record.resolution)
        and record.resolution >= config.low_resolution_cutoff_years
    ):
        required_overlap = config.low_resolution_min_overlap
    evaluations: list[dict[str, object]] = []
    for months in _candidate_months(record, config):
        native = (config.low_resolution_pairing == "window"
                  and np.isfinite(record.resolution)
                  and record.resolution > config.native_window_cutoff_years)
        if native:
            proxy, local_target, step = _native_screening_pair(record, target, config)
        else:
            proxy = annualize_record(
                record, months=months, season_year=config.season_year,
                minimum_month_fraction=config.minimum_month_fraction,
            )
            local_target = target
            step = 1
            if config.period is not None:
                start, end = config.period
                proxy = proxy.loc[(proxy.index >= start) & (proxy.index <= end)]
                local_target = target.loc[(target.index >= start) & (target.index <= end)]
        if config.detrend:
            proxy = _linear_detrend(proxy)
            local_target = _linear_detrend(local_target)
        r, p_value, n_eff, n = correlation_with_effective_dof(proxy, local_target, step)
        paired = pd.concat([proxy, local_target], axis=1).dropna()
        lag_pairs = int((np.diff(paired.index.to_numpy(int)) == step).sum())
        evaluations.append(
            {"months": months, "r": r, "p_effective_raw": p_value, "n_eff": n_eff,
             "n_overlap": n, "screening_pairing": "native_window" if native else "annual",
             "screening_window_years": step, "lag1_pair_count": lag_pairs}
        )
    p_values = [float(item["p_effective_raw"]) for item in evaluations]
    adjusted = _holm(p_values) if config.multiple_testing == "holm" else np.asarray(p_values)
    for item, value in zip(evaluations, adjusted):
        item["p_effective_adjusted"] = float(value)
    eligible = [item for item in evaluations if int(item["n_overlap"]) >= required_overlap]
    best = max(
        eligible or evaluations,
        key=lambda item: abs(float(item["r"])) if np.isfinite(float(item["r"])) else -np.inf,
    )
    p_selected = float(best["p_effective_adjusted"])
    r_selected = float(best["r"])
    selected = (
        int(best["n_overlap"]) >= required_overlap
        and np.isfinite(r_selected)
        and abs(r_selected) >= config.r_threshold
        and np.isfinite(p_selected)
        and p_selected <= config.p_threshold
    )
    return {
        "pid": record.pid,
        "archive": record.archive,
        "proxy": record.proxy,
        "lat": record.lat,
        "lon": record.lon,
        "native_resolution": record.resolution,
        "screening_pairing": best["screening_pairing"],
        "screening_window_years": best["screening_window_years"],
        "lag1_pair_count": best["lag1_pair_count"],
        "dof_fallback_no_adjacent_pairs": best["lag1_pair_count"] < 2,
        "best_months": ",".join(map(str, best["months"])),
        "n_seasons_tested": len(evaluations),
        "n_overlap": int(best["n_overlap"]),
        "minimum_overlap_required": int(required_overlap),
        "n_eff": float(best["n_eff"]),
        "r": r_selected,
        "p_effective_raw": float(best["p_effective_raw"]),
        "p_effective_adjusted": p_selected,
        "passes_local_threshold": bool(selected),
    }


def screen_proxies(
    records: ProxyCollection | Mapping[str, ProxyRecord] | Iterable[ProxyRecord],
    target: pd.Series,
    config: ScreeningConfig | None = None,
) -> pd.DataFrame:
    config = config or ScreeningConfig()
    if isinstance(records, Mapping):
        iterable = records.values()
    else:
        iterable = records
    rows = [_screen_one(record, target, config) for record in iterable]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    local_p = frame["p_effective_adjusted"].to_numpy(float)
    if config.proxy_multiple_testing == "fdr_bh":
        proxy_adjusted = _fdr_bh(local_p)
    elif config.proxy_multiple_testing == "holm":
        proxy_adjusted = _holm(local_p)
    else:
        proxy_adjusted = local_p
    frame["p_proxy_adjusted"] = proxy_adjusted
    frame["selected"] = (
        frame["passes_local_threshold"].astype(bool)
        & np.isfinite(frame["p_proxy_adjusted"])
        & (frame["p_proxy_adjusted"] <= config.p_threshold)
    )
    frame["selected_before_archive_cap"] = frame["selected"]
    frame["archive_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["excluded_by_archive_cap"] = False
    for archive, limit in config.archive_max_counts:
        candidates = frame.loc[
            frame["selected"]
            & frame["archive"].astype(str).str.casefold().eq(str(archive).strip().casefold())
        ].copy()
        if candidates.empty:
            continue
        candidates["absolute_r"] = candidates["r"].abs()
        candidates = candidates.sort_values(
            ["p_proxy_adjusted", "absolute_r", "n_eff", "pid"],
            ascending=[True, False, False, True],
        )
        frame.loc[candidates.index, "archive_rank"] = np.arange(1, len(candidates) + 1)
        excluded = candidates.index[int(limit):]
        frame.loc[excluded, "selected"] = False
        frame.loc[excluded, "excluded_by_archive_cap"] = True
    return frame.sort_values(
        ["selected", "p_proxy_adjusted", "pid"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

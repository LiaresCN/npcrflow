"""Native-support constraints that modify only a low-pass state component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr, spsolve

from .config import MultiresolutionConfig
from .records import ProxyRecord


@dataclass
class LowFrequencyResult:
    adjusted: pd.Series
    increment: pd.Series
    core_low_frequency: pd.Series
    adjusted_low_frequency: pd.Series
    core_high_frequency: pd.Series
    constraints: pd.DataFrame
    observations: pd.DataFrame


def _window(years: np.ndarray, center: float, width: float) -> np.ndarray:
    half = max(width, 1.0) / 2.0
    return np.flatnonzero((years >= center - half) & (years < center + half))


def _native_support_widths(
    record: ProxyRecord,
    maximum_multiplier: float | None = None,
) -> np.ndarray:
    """Approximate each sample's support from neighboring native timestamps."""

    time = record.time
    if time.size == 1:
        return np.ones(1, dtype=float)
    boundaries = np.empty(time.size + 1, dtype=float)
    boundaries[1:-1] = (time[:-1] + time[1:]) / 2.0
    boundaries[0] = time[0] - (time[1] - time[0]) / 2.0
    boundaries[-1] = time[-1] + (time[-1] - time[-2]) / 2.0
    widths = np.maximum(np.diff(boundaries), 1.0)
    if maximum_multiplier is not None and np.isfinite(record.resolution):
        widths = np.minimum(widths, max(1.0, maximum_multiplier * record.resolution))
    return widths


def native_support_counts(
    records: Iterable[ProxyRecord],
    years: pd.Index,
    *,
    maximum_support_multiplier: float | None = 3.0,
) -> pd.Series:
    """Count low-resolution records whose native support covers each year."""

    annual_years = years.to_numpy(float)
    counts = np.zeros(len(years), dtype=int)
    for record in records:
        covered = np.zeros(len(years), dtype=bool)
        for time, width in zip(
            record.time,
            _native_support_widths(record, maximum_support_multiplier),
        ):
            covered[_window(annual_years, time, width)] = True
        counts += covered.astype(int)
    return pd.Series(counts, index=years, name="low_resolution_support_count")


def _second_difference(n: int) -> sparse.csr_matrix:
    if n < 3:
        return sparse.csr_matrix((0, n))
    diagonals = [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)]
    return sparse.diags(diagonals, [0, 1, 2], shape=(n - 2, n)).tocsr()


def _period_lambda(period_years: float, multiplier: float = 1.0) -> float:
    """Whittaker strength whose half-power wavelength is approximately period."""

    frequency_term = max(2.0 * np.sin(np.pi / period_years), 1e-6)
    return float(multiplier / frequency_term**4)


def lowpass_annual_state(
    series: pd.Series,
    period_years: float,
    *,
    smoothness_multiplier: float = 1.0,
) -> pd.Series:
    """Estimate a low-pass annual state while treating missing years as missing."""

    if series.empty:
        return series.astype(float).copy()
    start = int(np.floor(series.index.min()))
    end = int(np.ceil(series.index.max()))
    years = pd.Index(np.arange(start, end + 1), name=series.index.name or "Year")
    aligned = series.groupby(series.index.astype(int)).mean().reindex(years).astype(float)
    values = aligned.to_numpy(float)
    finite = np.isfinite(values)
    if not finite.any():
        return pd.Series(np.nan, index=years, name=series.name, dtype=float)
    weights = finite.astype(float)
    difference = _second_difference(len(years))
    penalty = _period_lambda(period_years, smoothness_multiplier)
    system = sparse.diags(weights) + penalty * (difference.T @ difference)
    system = system + 1e-10 * sparse.eye(len(years))
    rhs = np.where(finite, values, 0.0)
    smoothed = spsolve(system.tocsc(), rhs)
    return pd.Series(smoothed, index=years, name=series.name)


def _calibrate_proxy(
    record: ProxyRecord,
    target_low: pd.Series,
    minimum_overlap: int,
    maximum_support_multiplier: float | None,
):
    resolution = record.resolution
    if not np.isfinite(resolution):
        return None
    target_years = target_low.index.to_numpy(float)
    target_values = target_low.to_numpy(float)
    proxy_values: list[float] = []
    target_averages: list[float] = []
    support_widths = _native_support_widths(record, maximum_support_multiplier)
    for time, value, width in zip(record.time, record.value, support_widths):
        indices = _window(target_years, time, width)
        finite = indices[np.isfinite(target_values[indices])]
        if finite.size:
            proxy_values.append(float(value))
            target_averages.append(float(np.mean(target_values[finite])))
    if len(proxy_values) < minimum_overlap or np.std(proxy_values) == 0:
        return None
    design = np.column_stack([np.ones(len(proxy_values)), proxy_values])
    coefficients = np.linalg.lstsq(design, target_averages, rcond=None)[0]
    residual = np.asarray(target_averages) - design @ coefficients
    variance = float(np.var(residual, ddof=2)) if len(residual) > 2 else float(np.var(residual))
    return coefficients, max(variance, 1e-6), len(proxy_values)


def variational_low_frequency_adjustment(
    core: pd.Series,
    low_resolution_records: Iterable[ProxyRecord],
    target: pd.Series,
    *,
    config: MultiresolutionConfig | None = None,
) -> LowFrequencyResult:
    """Assimilate native-window observations into only the low-pass component.

    The annual state is decomposed as ``core_high + core_low``. The variational
    solve estimates a smooth increment to ``core_low`` from native proxy-window
    averages. ``core_high`` is then added back unchanged. Thus a decadal sample
    can alter a decadal trend but cannot manufacture annual observations.
    """

    config = config or MultiresolutionConfig(enabled=True)
    if config.state_timestep_years != 1:
        raise ValueError("only an annual latent state is currently supported")
    years = core.index.to_numpy(int)
    n_years = len(years)
    core_low = lowpass_annual_state(
        core,
        config.lowpass_period_years,
        smoothness_multiplier=config.smoothness_multiplier,
    ).reindex(core.index)
    core_high = (core - core_low).where(core.notna(), 0.0).rename("core_high_frequency")

    # Smooth on an annual grid, then retain only genuinely observed training
    # years. Withheld target years never become calibration observations.
    target_low_grid = lowpass_annual_state(
        target,
        config.lowpass_period_years,
        smoothness_multiplier=config.smoothness_multiplier,
    )
    target_low = target_low_grid.reindex(target.index).rename("target_low_frequency")
    target_scale = float(np.nanstd(target_low.to_numpy(float)))

    rows: list[sparse.csr_matrix] = []
    right_hand: list[float] = []
    constraint_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    finite_core = np.flatnonzero(np.isfinite(core_low.to_numpy(float)))
    core_scale = float(np.nanstd(core_low.to_numpy(float)))
    reference_scale = max(core_scale, target_scale, 1e-6)
    if finite_core.size and config.core_anchor_weight > 0:
        weight = np.sqrt(config.core_anchor_weight) / reference_scale
        rows.append(
            sparse.coo_matrix(
                (np.full(finite_core.size, weight), (np.arange(finite_core.size), finite_core)),
                shape=(finite_core.size, n_years),
            ).tocsr()
        )
        right_hand.extend(np.zeros(finite_core.size).tolist())

    core_low_values = core_low.to_numpy(float)
    for record in low_resolution_records:
        calibration = _calibrate_proxy(
            record,
            target_low,
            config.minimum_calibration_overlap,
            config.maximum_support_multiplier,
        )
        if calibration is None:
            constraint_rows.append(
                {"pid": record.pid, "status": "excluded_no_calibration", "native_resolution": record.resolution}
            )
            continue
        coefficients, residual_variance, overlap = calibration
        local_rows: list[int] = []
        local_cols: list[int] = []
        local_values: list[float] = []
        local_rhs: list[float] = []
        support_widths = _native_support_widths(record, config.maximum_support_multiplier)
        for time, value, width in zip(record.time, record.value, support_widths):
            indices = _window(years.astype(float), time, width)
            if indices.size == 0:
                continue
            finite_low = indices[np.isfinite(core_low_values[indices])]
            if finite_low.size == 0:
                continue
            mapped_observation = float(coefficients[0] + coefficients[1] * value)
            core_window_mean = float(np.mean(core_low_values[finite_low]))
            row_number = len(local_rhs)
            weight = np.sqrt(config.proxy_constraint_weight) / np.sqrt(residual_variance)
            local_rows.extend([row_number] * len(indices))
            local_cols.extend(indices.tolist())
            local_values.extend([weight / len(indices)] * len(indices))
            local_rhs.append((mapped_observation - core_window_mean) * weight)
            observation_rows.append(
                {
                    "pid": record.pid,
                    "native_time": float(time),
                    "support_width_years": float(width),
                    "support_start": float(time - max(width, 1.0) / 2.0),
                    "support_end": float(time + max(width, 1.0) / 2.0),
                    "annual_state_start": int(years[indices].min()),
                    "annual_state_end": int(years[indices].max()),
                    "annual_state_count": int(indices.size),
                    "proxy_value": float(value),
                    "mapped_lowpass_value": mapped_observation,
                    "core_lowpass_window_mean": core_window_mean,
                    "innovation": mapped_observation - core_window_mean,
                    "constraint_weight": weight,
                }
            )
        if local_rhs:
            rows.append(
                sparse.coo_matrix(
                    (local_values, (local_rows, local_cols)),
                    shape=(len(local_rhs), n_years),
                ).tocsr()
            )
            right_hand.extend(local_rhs)
            constraint_rows.append(
                {
                    "pid": record.pid,
                    "status": "used",
                    "native_resolution": record.resolution,
                    "minimum_support_years": float(np.min(support_widths)),
                    "median_support_years": float(np.median(support_widths)),
                    "maximum_support_years": float(np.max(support_widths)),
                    "calibration_overlap": overlap,
                    "constraint_count": len(local_rhs),
                    "psm_intercept": float(coefficients[0]),
                    "psm_slope": float(coefficients[1]),
                    "psm_residual_variance": residual_variance,
                    "target_component": "lowpass",
                    "state_timestep_years": config.state_timestep_years,
                    "lowpass_period_years": config.lowpass_period_years,
                    "core_low_frequency_scale": core_scale,
                    "calibration_target_low_frequency_scale": target_scale,
                    "increment_reference_scale": reference_scale,
                }
            )

    used = [row for row in constraint_rows if row["status"] == "used"]
    if not rows or not used:
        unchanged = core.copy().rename("reconstruction_adjusted")
        return LowFrequencyResult(
            adjusted=unchanged,
            increment=pd.Series(0.0, index=core.index, name="low_frequency_increment"),
            core_low_frequency=core_low.rename("core_low_frequency"),
            adjusted_low_frequency=core_low.rename("adjusted_low_frequency"),
            core_high_frequency=core_high,
            constraints=pd.DataFrame(constraint_rows),
            observations=pd.DataFrame(observation_rows),
        )

    difference = _second_difference(n_years)
    if difference.shape[0] and config.smoothness_multiplier > 0:
        penalty = _period_lambda(config.lowpass_period_years, config.smoothness_multiplier)
        rows.append(np.sqrt(penalty) * difference)
        right_hand.extend(np.zeros(difference.shape[0]).tolist())
    operator = sparse.vstack(rows).tocsr()
    increment_values = lsqr(operator, np.asarray(right_hand), atol=1e-9, btol=1e-9)[0]
    increment = pd.Series(increment_values, index=core.index, name="low_frequency_increment")
    increment = lowpass_annual_state(
        increment,
        config.lowpass_period_years,
        smoothness_multiplier=config.smoothness_multiplier,
    ).reindex(core.index)

    calibration_index = increment.index.intersection(target.index)
    if config.preserve_calibration_mean and len(calibration_index):
        increment = increment - float(increment.reindex(calibration_index).mean())
    if config.maximum_increment_ratio is not None:
        increment_sd = float(np.nanstd(increment.to_numpy(float)))
        # A core reconstruction can be nearly devoid of low-frequency
        # variance—the exact deficiency this layer is meant to correct. A cap
        # referenced only to that core would force every useful increment back
        # toward zero. The target scale uses only the calibration years passed
        # by the caller, so the larger reference is still fold-safe.
        maximum_sd = config.maximum_increment_ratio * reference_scale
        if increment_sd > maximum_sd:
            increment = increment * (maximum_sd / increment_sd)

    adjusted_low = (core_low + increment).rename("adjusted_low_frequency")
    adjusted = (core_high + adjusted_low).rename("reconstruction_adjusted")
    return LowFrequencyResult(
        adjusted=adjusted,
        increment=increment.rename("low_frequency_increment"),
        core_low_frequency=core_low.rename("core_low_frequency"),
        adjusted_low_frequency=adjusted_low,
        core_high_frequency=core_high,
        constraints=pd.DataFrame(constraint_rows),
        observations=pd.DataFrame(observation_rows),
    )

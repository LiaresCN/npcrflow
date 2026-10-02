"""Native-window resolution sub-NESTs for mixed-resolution NPCR."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .config import PCAConfig, ReconstructionConfig
from .model import NativePCRModel, fit_native_pcr
from .records import ProxyRecord


@dataclass
class ResolutionNestModel:
    resolution_years: int
    native_pids: tuple[str, ...]
    model: NativePCRModel
    matrix: pd.DataFrame
    target: pd.Series
    prediction: pd.Series
    windows: pd.DataFrame
    internal_row: pd.Series | None


def assign_resolution_bands(
    records: Mapping[str, ProxyRecord],
    pids: Sequence[str],
    reconstruction_config: ReconstructionConfig,
) -> dict[int, list[str]]:
    """Assign each nonannual record to the smallest declared native window."""

    config = reconstruction_config.nest
    result = {width: [] for width in config.subnest_resolution_bins}
    for pid in pids:
        resolution = records[pid].resolution
        if (
            not np.isfinite(resolution)
            or resolution <= config.direct_annual_resolution_years
            or resolution > config.subnest_max_resolution_years
        ):
            continue
        for width in config.subnest_resolution_bins:
            if resolution <= width:
                result[width].append(pid)
                break
    return {width: values for width, values in result.items() if values}


def _window_table(start: int, end: int, width: int) -> pd.DataFrame:
    starts = np.arange(start, end + 1, width, dtype=int)
    ends = np.minimum(starts + width - 1, end)
    return pd.DataFrame(
        {
            "window_id": starts,
            "window_start": starts,
            "window_end": ends,
            "window_center": (starts + ends) / 2.0,
            "window_year_count": ends - starts + 1,
        }
    ).set_index("window_id")


def _raw_record_window_means(
    record: ProxyRecord,
    windows: pd.DataFrame,
    standardization_period: tuple[int, int] | None = None,
    standardization_years: Sequence[int] | None = None,
) -> pd.Series:
    source_values = np.asarray(record.value, dtype=float)
    if standardization_period is not None:
        start, end = standardization_period
        reference = (
            (record.time >= start)
            & (record.time <= end)
            & np.isfinite(source_values)
        )
        if standardization_years is not None:
            observed_years = np.rint(record.time).astype(int)
            reference &= np.isin(
                observed_years,
                np.asarray(list(standardization_years), dtype=int),
            )
        reference_values = source_values[reference]
        mean = float(np.mean(reference_values))
        scale = float(np.std(reference_values, ddof=1))
        if len(reference_values) < 2 or not np.isfinite(scale) or scale <= 0:
            raise RuntimeError(
                f"proxy {record.pid} cannot be standardized over "
                f"{standardization_period} within the training years"
            )
        source_values = (source_values - mean) / scale
    positions = np.searchsorted(
        windows["window_start"].to_numpy(float) - 0.5,
        record.time,
        side="right",
    ) - 1
    safe_positions = np.clip(positions, 0, len(windows) - 1)
    upper = windows["window_end"].to_numpy(float)[safe_positions] + 0.5
    inside = (
        (positions >= 0)
        & (record.time < upper)
        & np.isfinite(record.time)
        & np.isfinite(source_values)
    )
    observed = pd.Series(
        source_values[inside],
        index=windows.index.to_numpy()[positions[inside]],
        dtype=float,
    )
    return observed.groupby(level=0).mean().reindex(windows.index).rename(record.pid)


def build_resolution_matrix(
    annual_matrix: pd.DataFrame,
    records: Mapping[str, ProxyRecord],
    native_pids: Sequence[str],
    target: pd.Series,
    start: int,
    end: int,
    width: int,
    standardization_period: tuple[int, int] | None = None,
    standardization_years: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Aggregate annual predictors and target onto native windows, without interpolation."""

    windows = _window_table(start, end, width)
    annual = annual_matrix.loc[start:end]
    annual_window_ids = start + ((annual.index.to_numpy(int) - start) // width) * width
    matrix = annual.groupby(annual_window_ids).mean().reindex(windows.index)
    for pid in native_pids:
        matrix[pid] = _raw_record_window_means(
            records[pid],
            windows,
            standardization_period,
            standardization_years,
        )
    observed_target = target.loc[start:end]
    target_window_ids = start + ((observed_target.index.to_numpy(int) - start) // width) * width
    window_target = observed_target.groupby(target_window_ids).mean().reindex(windows.index)
    matrix.attrs["proxy_metadata"] = {
        pid: {
            "lat": records[pid].lat,
            "lon": records[pid].lon,
            "archive": records[pid].archive,
            "proxy": records[pid].proxy,
            "site": records[pid].metadata.get("geo_siteName", ""),
            "source": records[pid].metadata.get("dataSetName", ""),
            "metadata": dict(records[pid].metadata),
        }
        for pid in matrix.columns
    }
    return matrix, window_target, windows


def _selected_internal_row(table: pd.DataFrame) -> pd.Series | None:
    if table.empty:
        return None
    selected = table.loc[table["selected"].astype(bool)] if "selected" in table else table
    if selected.empty:
        return None
    if "selection_stage" in selected:
        amplitude = selected.loc[selected["selection_stage"] == "amplitude"]
        if not amplitude.empty:
            return amplitude.iloc[-1]
    return selected.iloc[-1]


def fit_resolution_subnests(
    annual_matrix: pd.DataFrame,
    records: Mapping[str, ProxyRecord],
    nonannual_pids: Sequence[str],
    target: pd.Series,
    train_years: Sequence[int],
    start: int,
    end: int,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
) -> tuple[list[ResolutionNestModel], pd.DataFrame]:
    """Fit independent PCA/PCR models at each represented native resolution."""

    if (
        not reconstruction_config.nest.multiresolution_subnests
        or reconstruction_config.nest.subnest_constraint_weight == 0
    ):
        return [], pd.DataFrame()
    bands = assign_resolution_bands(records, nonannual_pids, reconstruction_config)
    train = set(int(year) for year in train_years)
    models: list[ResolutionNestModel] = []
    audit: list[dict[str, object]] = []
    for width, native_pids in bands.items():
        matrix, window_target, windows = build_resolution_matrix(
            annual_matrix,
            records,
            native_pids,
            target,
            start,
            end,
            width,
            reconstruction_config.standardization_period,
            train_years,
        )
        native_available = matrix[native_pids].notna().any(axis=1)
        train_windows = [
            int(index)
            for index, row in windows.iterrows()
            if set(range(int(row.window_start), int(row.window_end) + 1)).issubset(train)
            and np.isfinite(window_target.loc[index])
            and bool(native_available.loc[index])
        ]
        base = {
            "resolution_years": width,
            "native_pids": "|".join(native_pids),
            "native_proxy_count": len(native_pids),
            "predictor_count": matrix.shape[1],
            "native_observed_window_count": int(native_available.sum()),
            "calibration_window_count": len(train_windows),
        }
        if len(train_windows) < reconstruction_config.nest.minimum_subnest_calibration_windows:
            audit.append({**base, "accepted": False, "reason": "insufficient_windows"})
            continue
        try:
            local_pca = replace(
                pca_config,
                max_components=min(
                    pca_config.max_components,
                    max(1, (2 * len(train_windows)) // 3 - 3),
                ),
                min_pairwise_overlap=min(
                    pca_config.min_pairwise_overlap,
                    max(4, (2 * len(train_windows)) // 3),
                ),
            )
            local_config = replace(
                reconstruction_config,
                validation_block_years=max(
                    2,
                    min(
                        reconstruction_config.validation_block_years,
                        len(train_windows) // 3,
                    ),
                ),
                minimum_internal_train_samples=4,
            )
            model = fit_native_pcr(
                matrix,
                window_target,
                train_windows,
                local_pca,
                local_config,
            )
            fitted_native = tuple(pid for pid in native_pids if pid in model.columns)
            if not fitted_native:
                audit.append({
                    **base,
                    "accepted": False,
                    "reason": "no_native_proxy_in_fitted_pca",
                })
                continue
            internal = _selected_internal_row(model.selection_table)
            passed = bool(
                internal is not None
                and internal.get(
                    "passes_internal_ce_re",
                    internal.get("passes_skill_floor", False),
                )
            )
            if (
                reconstruction_config.nest.require_internal_ce_re
                and reconstruction_config.auto_tune
                and not passed
            ):
                audit.append({
                    **base,
                    "accepted": False,
                    "reason": "internal_ce_re",
                    "internal_median_ce": (
                        float(internal.get("median_ce", np.nan))
                        if internal is not None else np.nan
                    ),
                    "internal_median_re": (
                        float(internal.get("median_re", np.nan))
                        if internal is not None else np.nan
                    ),
                })
                continue
            fitted_native_available = matrix[list(fitted_native)].notna().any(axis=1)
            prediction = model.predict(matrix).where(fitted_native_available)
            models.append(ResolutionNestModel(
                resolution_years=width,
                native_pids=fitted_native,
                model=model,
                matrix=matrix,
                target=window_target,
                prediction=prediction,
                windows=windows,
                internal_row=internal,
            ))
            audit.append({
                **base,
                "accepted": True,
                "reason": "accepted",
                "fitted_native_pids": "|".join(fitted_native),
                "fitted_native_proxy_count": len(fitted_native),
                "n_components": model.n_components,
                "regression": model.regression_name,
                "alpha": model.alpha,
                "internal_median_ce": (
                    float(internal.get("median_ce", np.nan))
                    if internal is not None else np.nan
                ),
                "internal_median_re": (
                    float(internal.get("median_re", np.nan))
                    if internal is not None else np.nan
                ),
                "passes_internal_ce_re": passed,
                "prediction_window_count": int(prediction.notna().sum()),
            })
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            audit.append({
                **base,
                "accepted": False,
                "reason": f"{type(error).__name__}: {error}",
            })
    return models, pd.DataFrame(audit)


def fuse_resolution_subnests(
    core: pd.Series,
    models: Sequence[ResolutionNestModel],
    reconstruction_config: ReconstructionConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Assimilate sub-NEST window predictions into an annual state."""

    if not models or reconstruction_config.nest.subnest_constraint_weight == 0:
        return core.copy(), pd.DataFrame()
    years = core.index.to_numpy(int)
    finite_core = np.isfinite(core.to_numpy(float))
    scale = max(float(np.nanstd(core.to_numpy(float))), 1e-6)
    rows: list[sparse.csr_matrix] = []
    rhs: list[float] = []
    audit: list[dict[str, object]] = []
    if finite_core.any():
        indices = np.flatnonzero(finite_core)
        anchor = 1.0 / scale
        rows.append(sparse.coo_matrix(
            (np.full(len(indices), anchor), (np.arange(len(indices)), indices)),
            shape=(len(indices), len(years)),
        ).tocsr())
        rhs.extend(np.zeros(len(indices)).tolist())
    for layer in sorted(models, key=lambda item: item.resolution_years):
        weight = np.sqrt(reconstruction_config.nest.subnest_constraint_weight) / scale
        for window_id, predicted in layer.prediction.dropna().items():
            window = layer.windows.loc[window_id]
            indices = np.flatnonzero(
                (years >= int(window.window_start))
                & (years <= int(window.window_end))
                & finite_core
            )
            if not len(indices):
                continue
            core_mean = float(np.mean(core.to_numpy(float)[indices]))
            row = sparse.coo_matrix(
                (
                    np.full(len(indices), weight / len(indices)),
                    (np.zeros(len(indices), dtype=int), indices),
                ),
                shape=(1, len(years)),
            ).tocsr()
            rows.append(row)
            rhs.append((float(predicted) - core_mean) * weight)
            audit.append({
                "resolution_years": layer.resolution_years,
                "window_start": int(window.window_start),
                "window_end": int(window.window_end),
                "subnest_prediction": float(predicted),
                "core_window_mean": core_mean,
                "innovation": float(predicted) - core_mean,
                "native_pids": "|".join(layer.native_pids),
            })
    if len(rows) <= 1:
        return core.copy(), pd.DataFrame(audit)
    if len(years) >= 3 and reconstruction_config.nest.subnest_smoothness_multiplier > 0:
        difference = sparse.diags(
            [np.ones(len(years) - 2), -2 * np.ones(len(years) - 2), np.ones(len(years) - 2)],
            [0, 1, 2],
            shape=(len(years) - 2, len(years)),
        ).tocsr()
        smooth = reconstruction_config.nest.subnest_smoothness_multiplier
        rows.append(np.sqrt(smooth) * difference)
        rhs.extend(np.zeros(difference.shape[0]).tolist())
    increment = lsqr(
        sparse.vstack(rows).tocsr(), np.asarray(rhs), atol=1e-9, btol=1e-9
    )[0]
    result = core + pd.Series(increment, index=core.index)
    return result.rename(core.name), pd.DataFrame(audit)

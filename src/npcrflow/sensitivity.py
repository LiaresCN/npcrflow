"""Configurable network and temporal sensitivity experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import pandas as pd

from .config import PCAConfig, ReconstructionConfig, ScreeningConfig, SensitivityConfig
from .reconstruction import (
    blocked_validation,
    build_proxy_matrix,
    split_resolution_roles,
    tune_multiresolution_config,
)
from .low_frequency import variational_low_frequency_adjustment
from .model import fit_native_pcr
from .records import ProxyRecord
from .screening import screen_proxies
from .validation import (
    multiscale_reconstruction_metrics,
    reconstruction_metrics,
    spectral_reconstruction_metrics,
)


def summarize_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    """Return compact uncertainty ranges for the skill columns in a test."""

    base_metric_columns = (
            "median_r", "median_rmse", "median_re", "median_ce",
            "median_sd_ratio", "median_variance_ratio",
            "r", "rmse", "re", "ce", "sd_ratio", "variance_ratio",
    )
    metric_columns = [column for column in base_metric_columns if column in table]
    metric_columns.extend(
        column
        for column in table
        if (column.startswith("lowpass_") or column.startswith("spectral_"))
        and column not in metric_columns
    )
    rows: list[dict[str, float | int | str]] = []
    for metric in metric_columns:
        values = pd.to_numeric(table[metric], errors="coerce").dropna().to_numpy(float)
        if not values.size:
            continue
        rows.append(
            {
                "metric": metric,
                "n": int(values.size),
                "minimum": float(np.min(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "maximum": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def _fit_network(
    records: Mapping[str, ProxyRecord],
    pids: list[str],
    screening: pd.DataFrame,
    target: pd.Series,
    screening_config: ScreeningConfig,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
) -> dict[str, float | int | str]:
    core_pids, low_resolution_pids = split_resolution_roles(
        records,
        pids,
        pca_config,
        reconstruction_config,
    )
    local_pca = pca_config
    if len(core_pids) == 1:
        local_pca = replace(pca_config, max_components=1, min_proxies_per_year=1)
    matrix = build_proxy_matrix(
        records,
        core_pids,
        screening_config,
        screening[screening["pid"].isin(pids)],
        reconstruction_config.reconstruction_period,
        reconstruction_config.interpolation,
    )
    years = target.dropna().index.to_numpy(int)
    if reconstruction_config.calibration_period is not None:
        years = years[(years >= reconstruction_config.calibration_period[0]) & (years <= reconstruction_config.calibration_period[1])]
    years = years[np.isin(years, matrix.index)]
    model = fit_native_pcr(matrix, target, years, local_pca, reconstruction_config)
    prediction = model.predict(matrix)
    low_resolution_records = [records[pid] for pid in low_resolution_pids]
    selected_multiresolution = reconstruction_config.multiresolution
    if low_resolution_records:
        selected_multiresolution, _ = tune_multiresolution_config(
            matrix,
            target,
            years,
            local_pca,
            reconstruction_config,
            low_resolution_records,
            n_components=model.n_components,
            alpha=model.alpha,
            regression_name=model.regression_name,
        )
    else:
        selected_multiresolution = replace(
            selected_multiresolution,
            proxy_constraint_weight=0.0,
            auto_tune=False,
        )
    if low_resolution_records and selected_multiresolution.proxy_constraint_weight > 0:
        prediction = variational_low_frequency_adjustment(
            prediction,
            low_resolution_records,
            target.reindex(years).dropna(),
            config=selected_multiresolution,
        ).adjusted
    in_sample = reconstruction_metrics(
        target.reindex(years), prediction.reindex(years), calibration_mean=float(target.reindex(years).mean())
    )
    folds = blocked_validation(
        matrix,
        target,
        local_pca,
        reconstruction_config,
        low_resolution_records=low_resolution_records,
    )
    fold_summary: dict[str, float] = {}
    for metric in ("r", "rmse", "re", "ce", "sd_ratio", "variance_ratio"):
        fold_summary[f"median_{metric}"] = (
            float(folds[metric].median()) if metric in folds and not folds.empty else np.nan
        )
    return {
        "proxy_count": len(pids),
        "model_proxy_count": len(model.columns),
        "low_frequency_proxy_count": len(low_resolution_pids),
        "pids": "|".join(pids),
        "regression": model.regression_name,
        "n_components": model.n_components,
        "alpha": model.alpha,
        "weighted_proxy_count": float(model.proxy_weights.sum()),
        "downweighted_proxy_count": int((model.proxy_weights < 1.0).sum()),
        "amplitude_method": model.amplitude_calibrator.method,
        "amplitude_reference": model.amplitude_calibrator.reference,
        "amplitude_reference_overlap": model.amplitude_calibrator.reference_overlap,
        "amplitude_reference_min_proxy_count": model.amplitude_calibrator.reference_min_proxy_count,
        "amplitude_slope": model.amplitude_calibrator.slope,
        "fold_count": len(folds),
        "in_sample_r": in_sample["r"],
        **fold_summary,
    }


def run_network_sensitivities(
    records: Mapping[str, ProxyRecord],
    screening: pd.DataFrame,
    target: pd.Series,
    screening_config: ScreeningConfig,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
    sensitivity_config: SensitivityConfig,
) -> dict[str, pd.DataFrame]:
    selected = screening.loc[screening["selected"], "pid"].astype(str).tolist()
    output: dict[str, pd.DataFrame] = {}
    if sensitivity_config.single_proxy:
        rows = []
        for pid in selected:
            try:
                rows.append({"held_or_used_pid": pid, **_fit_network(records, [pid], screening, target, screening_config, pca_config, reconstruction_config)})
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
        output["single_proxy"] = pd.DataFrame(rows)
        output["single_proxy_summary"] = summarize_sensitivity(output["single_proxy"])
    if sensitivity_config.leave_one_proxy_out and len(selected) > 2:
        rows = []
        for omitted in selected:
            pids = [pid for pid in selected if pid != omitted]
            try:
                rows.append({"held_or_used_pid": omitted, **_fit_network(records, pids, screening, target, screening_config, pca_config, reconstruction_config)})
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
        output["leave_one_proxy_out"] = pd.DataFrame(rows)
        output["leave_one_proxy_out_summary"] = summarize_sensitivity(output["leave_one_proxy_out"])
    if sensitivity_config.random_delete_repeats > 0 and len(selected) > 2:
        rng = np.random.default_rng(reconstruction_config.random_seed)
        keep_count = max(2, int(np.ceil(len(selected) * (1.0 - sensitivity_config.random_delete_fraction))))
        rows = []
        for repeat in range(1, sensitivity_config.random_delete_repeats + 1):
            pids = sorted(rng.choice(selected, size=keep_count, replace=False).tolist())
            try:
                rows.append({"repeat": repeat, **_fit_network(records, pids, screening, target, screening_config, pca_config, reconstruction_config)})
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
        output["random_proxy_deletion"] = pd.DataFrame(rows)
        output["random_proxy_deletion_summary"] = summarize_sensitivity(
            output["random_proxy_deletion"]
        )
    if sensitivity_config.split_periods:
        rows = []
        for calibration_period, validation_period in sensitivity_config.split_periods:
            local_screen_config = replace(screening_config, period=calibration_period)
            local_screening = screen_proxies(records, target, local_screen_config)
            pids = local_screening.loc[local_screening["selected"], "pid"].astype(str).tolist()
            if not pids:
                continue
            try:
                core_pids, low_resolution_pids = split_resolution_roles(
                    records,
                    pids,
                    pca_config,
                    reconstruction_config,
                )
                local_pca = pca_config
                if len(core_pids) == 1:
                    local_pca = replace(pca_config, max_components=1, min_proxies_per_year=1)
                matrix = build_proxy_matrix(
                    records,
                    core_pids,
                    local_screen_config,
                    local_screening,
                    reconstruction_config.reconstruction_period,
                    reconstruction_config.interpolation,
                )
                train = target.loc[
                    (target.index >= calibration_period[0])
                    & (target.index <= calibration_period[1])
                    & target.index.isin(matrix.index)
                ].index.to_numpy(int)
                validation = target.loc[
                    (target.index >= validation_period[0])
                    & (target.index <= validation_period[1])
                    & target.index.isin(matrix.index)
                ].index.to_numpy(int)
                model = fit_native_pcr(
                    matrix,
                    target,
                    train,
                    local_pca,
                    replace(reconstruction_config, calibration_period=calibration_period),
                )
                prediction = model.predict(matrix.loc[validation])
                low_resolution_records = [records[pid] for pid in low_resolution_pids]
                selected_multiresolution = reconstruction_config.multiresolution
                if low_resolution_records:
                    selected_multiresolution, _ = tune_multiresolution_config(
                        matrix,
                        target,
                        train,
                        local_pca,
                        reconstruction_config,
                        low_resolution_records,
                        n_components=model.n_components,
                        alpha=model.alpha,
                        regression_name=model.regression_name,
                    )
                else:
                    selected_multiresolution = replace(
                        selected_multiresolution,
                        proxy_constraint_weight=0.0,
                        auto_tune=False,
                    )
                if (
                    low_resolution_records
                    and selected_multiresolution.proxy_constraint_weight > 0
                ):
                    full_prediction = model.predict(matrix)
                    prediction = variational_low_frequency_adjustment(
                        full_prediction,
                        low_resolution_records,
                        target.reindex(train).dropna(),
                        config=selected_multiresolution,
                    ).adjusted.reindex(validation)
                metrics = reconstruction_metrics(
                    target.reindex(validation),
                    prediction,
                    calibration_mean=float(target.reindex(train).mean()),
                )
                metrics.update(
                    multiscale_reconstruction_metrics(
                        target.reindex(validation),
                        prediction,
                        reconstruction_config.evaluation_lowpass_periods,
                        calibration_mean=float(target.reindex(train).mean()),
                    )
                )
                metrics.update(
                    spectral_reconstruction_metrics(
                        target.reindex(validation),
                        prediction,
                        reconstruction_config.evaluation_period_bands,
                    )
                )
                rows.append(
                    {
                        "calibration_start": calibration_period[0],
                        "calibration_end": calibration_period[1],
                        "validation_start": validation_period[0],
                        "validation_end": validation_period[1],
                        "proxy_count": len(pids),
                        "model_proxy_count": len(model.columns),
                        "low_frequency_proxy_count": len(low_resolution_pids),
                        "selected_proxy_constraint_weight": (
                            selected_multiresolution.proxy_constraint_weight
                        ),
                        "selected_lowpass_period_years": (
                            selected_multiresolution.lowpass_period_years
                        ),
                        "pids": "|".join(pids),
                        "regression": model.regression_name,
                        "n_components": model.n_components,
                        "alpha": model.alpha,
                        "weighted_proxy_count": float(model.proxy_weights.sum()),
                        "downweighted_proxy_count": int((model.proxy_weights < 1.0).sum()),
                        "amplitude_method": model.amplitude_calibrator.method,
                        "amplitude_reference": model.amplitude_calibrator.reference,
                        "amplitude_reference_overlap": model.amplitude_calibrator.reference_overlap,
                        "amplitude_reference_min_proxy_count": model.amplitude_calibrator.reference_min_proxy_count,
                        "amplitude_slope": model.amplitude_calibrator.slope,
                        **metrics,
                    }
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
        output["split_period"] = pd.DataFrame(rows)
        output["split_period_summary"] = summarize_sensitivity(output["split_period"])
    return output

"""End-to-end in-memory reconstruction; nest workbooks are never required."""

from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import MultiresolutionConfig, PCAConfig, ReconstructionConfig, ScreeningConfig
from .data import annualize_record
from .low_frequency import (
    LowFrequencyResult,
    native_support_counts,
    variational_low_frequency_adjustment,
)
from .model import NativePCRModel, fit_native_pcr
from .records import ProxyCollection, ProxyRecord
from .screening import screen_proxies
from .validation import (
    contiguous_folds,
    directed_edge_folds,
    multiscale_reconstruction_metrics,
    reconstruction_metrics,
    spectral_reconstruction_metrics,
)


@dataclass
class ReconstructionResult:
    reconstruction: pd.DataFrame
    model: NativePCRModel
    validation_folds: pd.DataFrame
    validation_summary: pd.DataFrame
    model_selection: pd.DataFrame
    proxy_weights: pd.DataFrame
    availability: pd.DataFrame
    low_frequency_constraints: pd.DataFrame
    low_frequency_observations: pd.DataFrame
    multiresolution_selection: pd.DataFrame


def _months_from_screening(screening: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    if screening.empty or "best_months" not in screening:
        return result
    for _, row in screening.iterrows():
        result[str(row["pid"])] = tuple(int(value) for value in str(row["best_months"]).split(",") if value)
    return result


def build_proxy_matrix(
    records: Mapping[str, ProxyRecord],
    pids: Sequence[str],
    screening_config: ScreeningConfig,
    screening: pd.DataFrame | None = None,
    reconstruction_period: tuple[int | None, int | None] | None = None,
    interpolation: str = "none",
) -> pd.DataFrame:
    month_lookup = _months_from_screening(screening if screening is not None else pd.DataFrame())
    series: dict[str, pd.Series] = {}
    for pid in pids:
        proxy = annualize_record(
            records[pid],
            months=month_lookup.get(pid, screening_config.months),
            season_year=screening_config.season_year,
            minimum_month_fraction=screening_config.minimum_month_fraction,
        )
        series[pid] = proxy
    if not series:
        raise RuntimeError("no proxies were selected")
    automatic_start = min(int(item.index.min()) for item in series.values() if not item.empty)
    automatic_end = max(int(item.index.max()) for item in series.values() if not item.empty)
    if reconstruction_period is None:
        start, end = automatic_start, automatic_end
    else:
        requested_start, requested_end = reconstruction_period
        start = automatic_start if requested_start is None else requested_start
        end = automatic_end if requested_end is None else requested_end
    years = pd.Index(np.arange(start, end + 1), name="Year")
    matrix = pd.DataFrame({pid: item.reindex(years) for pid, item in series.items()}, index=years)
    matrix.attrs["proxy_metadata"] = {
        pid: {
            "lat": records[pid].lat,
            "lon": records[pid].lon,
            "archive": records[pid].archive,
            "proxy": records[pid].proxy,
            "site": records[pid].metadata.get("geo_siteName", ""),
            "source": records[pid].metadata.get(
                "dataSetName", records[pid].metadata.get("originalDataURL", "")
            ),
            "metadata": dict(records[pid].metadata),
        }
        for pid in pids
    }
    if interpolation != "none":
        raise ValueError("proxy interpolation is disabled; native missing values must be retained")
    return matrix


def split_resolution_roles(
    records: Mapping[str, ProxyRecord],
    pids: Sequence[str],
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
) -> tuple[list[str], list[str]]:
    """Route usable sub-decadal records to PCR and slower records to low-pass.

    Records at or below ``regression_max_resolution_years`` retain their
    native missing years in the PCR matrix; they are never interpolated. Only
    records slower than that declared limit are reserved for the low-frequency
    observation operator.
    """

    selected = list(pids)
    if not reconstruction_config.multiresolution.enabled:
        return selected, []
    maximum = reconstruction_config.multiresolution.regression_max_resolution_years
    low_resolution = [
        pid
        for pid in selected
        if np.isfinite(records[pid].resolution) and records[pid].resolution > maximum
    ]
    core = [pid for pid in selected if pid not in low_resolution]
    if len(core) < pca_config.min_proxies_per_year:
        return selected, []
    return core, low_resolution


def _calibration_years(target: pd.Series, config: ReconstructionConfig) -> np.ndarray:
    years = target.dropna().index.to_numpy(int)
    if config.calibration_period is not None:
        years = years[
            (years >= config.calibration_period[0]) & (years <= config.calibration_period[1])
        ]
    return years


def _holdout_position(validation: np.ndarray, all_years: np.ndarray) -> str:
    """Label whether a contiguous outer holdout is at the early or late edge."""

    if validation.min() == all_years.min():
        return "early"
    if validation.max() == all_years.max():
        return "late"
    return "middle"


def tune_multiresolution_config(
    matrix: pd.DataFrame,
    target: pd.Series,
    train_years: np.ndarray,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
    low_resolution_records: Sequence[ProxyRecord],
    *,
    n_components: int,
    alpha: float,
    regression_name: str,
) -> tuple[MultiresolutionConfig, pd.DataFrame]:
    """Select low-frequency influence using training-only contiguous folds.

    Weight zero is an explicit candidate. It means that sparse or conflicting
    low-resolution evidence cannot degrade the reconstruction merely because
    the multiresolution layer was enabled.
    """

    config = reconstruction_config.multiresolution
    if not config.enabled or not config.auto_tune or not low_resolution_records:
        return config, pd.DataFrame()
    folds = contiguous_folds(
        train_years,
        reconstruction_config.validation_block_years,
        minimum_train_years=max(20, pca_config.max_components + 3),
    )
    if len(folds) < config.minimum_tuning_folds:
        fallback = replace(config, proxy_constraint_weight=0.0, auto_tune=False)
        return fallback, pd.DataFrame(
            [{
                "status": "insufficient_tuning_folds",
                "selected": True,
                "proxy_constraint_weight": 0.0,
                "lowpass_period_years": config.lowpass_period_years,
                "folds": len(folds),
            }]
        )
    candidates: list[tuple[float, float]] = [(0.0, config.lowpass_period_years)]
    candidates.extend(
        (float(weight), float(period))
        for weight in config.proxy_constraint_weight_candidates
        if weight > 0
        for period in config.lowpass_period_candidates
    )
    fold_rows: list[dict[str, float | int]] = []
    fitting_config = replace(
        reconstruction_config,
        multiresolution=replace(config, auto_tune=False),
    )
    for fold_number, (inner_train, inner_validation) in enumerate(folds, start=1):
        try:
            model = fit_native_pcr(
                matrix,
                target,
                inner_train,
                pca_config,
                fitting_config,
                forced_n_components=n_components,
                forced_alpha=alpha,
                forced_regression=regression_name,
            )
            core = model.predict(matrix)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        for weight, period in candidates:
            prediction = core.reindex(inner_validation)
            used_count = 0
            if weight > 0:
                candidate_config = replace(
                    config,
                    proxy_constraint_weight=weight,
                    lowpass_period_years=period,
                    auto_tune=False,
                )
                adjustment = variational_low_frequency_adjustment(
                    core,
                    low_resolution_records,
                    target.reindex(inner_train).dropna(),
                    config=candidate_config,
                )
                prediction = adjustment.adjusted.reindex(inner_validation)
                if not adjustment.constraints.empty and "status" in adjustment.constraints:
                    used_count = int((adjustment.constraints["status"] == "used").sum())
            annual = reconstruction_metrics(
                target.reindex(inner_validation),
                prediction,
                calibration_mean=float(target.reindex(inner_train).mean()),
            )
            multiscale = multiscale_reconstruction_metrics(
                target.reindex(inner_validation),
                prediction,
                (config.selection_lowpass_period_years,),
                calibration_mean=float(target.reindex(inner_train).mean()),
            )
            label = f"{config.selection_lowpass_period_years:g}".replace(".", "p")
            prefix = f"lowpass_{label}y_"
            fold_rows.append(
                {
                    "fold": fold_number,
                    "proxy_constraint_weight": weight,
                    "lowpass_period_years": period,
                    "low_frequency_proxy_count": used_count,
                    "annual_r": annual["r"],
                    "annual_ce": annual["ce"],
                    "annual_sd_ratio": annual["sd_ratio"],
                    "lowpass_r": multiscale[f"{prefix}r"],
                    "lowpass_ce": multiscale[f"{prefix}ce"],
                    "lowpass_sd_ratio": multiscale[f"{prefix}sd_ratio"],
                }
            )
    fold_table = pd.DataFrame(fold_rows)
    if fold_table.empty:
        fallback = replace(config, proxy_constraint_weight=0.0, auto_tune=False)
        return fallback, pd.DataFrame(
            [{
                "status": "no_successful_tuning_folds",
                "selected": True,
                "proxy_constraint_weight": 0.0,
                "lowpass_period_years": config.lowpass_period_years,
                "folds": 0,
            }]
        )
    rows: list[dict[str, float | int | bool | str]] = []
    for (weight, period), group in fold_table.groupby(
        ["proxy_constraint_weight", "lowpass_period_years"], sort=True
    ):
        annual_ce = float(group["annual_ce"].median())
        lowpass_ce = float(group["lowpass_ce"].median())
        lowpass_r = float(group["lowpass_r"].median())
        valid_skill = np.isfinite(annual_ce) and np.isfinite(lowpass_ce)
        objective = (
            min(annual_ce, lowpass_ce) + 0.1 * lowpass_r
            if valid_skill and np.isfinite(lowpass_r)
            else -np.inf
        )
        rows.append(
            {
                "status": "evaluated",
                "selected": False,
                "proxy_constraint_weight": float(weight),
                "lowpass_period_years": float(period),
                "folds": int(group["fold"].nunique()),
                "median_low_frequency_proxy_count": float(
                    group["low_frequency_proxy_count"].median()
                ),
                "median_annual_r": float(group["annual_r"].median()),
                "median_annual_ce": annual_ce,
                "median_annual_sd_ratio": float(group["annual_sd_ratio"].median()),
                "median_lowpass_r": lowpass_r,
                "median_lowpass_ce": lowpass_ce,
                "median_lowpass_sd_ratio": float(group["lowpass_sd_ratio"].median()),
                "objective": objective,
            }
        )
    selection = pd.DataFrame(rows)
    best = selection.sort_values(
        ["objective", "proxy_constraint_weight"],
        ascending=[False, True],
    ).iloc[0]
    selection.loc[best.name, "selected"] = True
    selected_config = replace(
        config,
        proxy_constraint_weight=float(best["proxy_constraint_weight"]),
        lowpass_period_years=float(best["lowpass_period_years"]),
        auto_tune=False,
    )
    return selected_config, selection


def blocked_validation(
    matrix: pd.DataFrame,
    target: pd.Series,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
    low_resolution_records: Sequence[ProxyRecord] = (),
) -> pd.DataFrame:
    """Contiguous sensitivity with the complete screened proxy grid fixed.

    Proxy membership and selected seasons come from the full screening step;
    PCA and regression are still refitted without each held-out block.  This
    intentionally conditional experiment complements, rather than replaces,
    the stricter fold-rescreened sensitivity.
    """
    years = _calibration_years(target, reconstruction_config)
    folds = directed_edge_folds(
        years,
        reconstruction_config.external_validation_fraction,
        minimum_train_years=max(20, pca_config.max_components + 3),
    )
    rows: list[dict[str, object]] = []
    for fold_number, (train, validation) in enumerate(folds, start=1):
        try:
            model = fit_native_pcr(matrix, target, train, pca_config, reconstruction_config)
            core_prediction = model.predict(matrix)
            raw_prediction = model.predict_raw(matrix)
            prediction = core_prediction.reindex(validation)
            core_metrics = reconstruction_metrics(
                target.reindex(validation),
                prediction,
                calibration_mean=float(target.reindex(train).mean()),
            )
            low_frequency_used = 0
            selected_multiresolution = reconstruction_config.multiresolution
            if low_resolution_records:
                selected_multiresolution, _ = tune_multiresolution_config(
                    matrix,
                    target,
                    train,
                    pca_config,
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
                adjustment = variational_low_frequency_adjustment(
                    core_prediction,
                    low_resolution_records,
                    target.reindex(train).dropna(),
                    config=selected_multiresolution,
                )
                prediction = adjustment.adjusted.reindex(validation)
                if not adjustment.constraints.empty and "status" in adjustment.constraints:
                    low_frequency_used = int((adjustment.constraints["status"] == "used").sum())
            metrics = reconstruction_metrics(
                target.reindex(validation),
                prediction,
                calibration_mean=float(target.reindex(train).mean()),
            )
            raw_metrics = reconstruction_metrics(
                target.reindex(validation),
                raw_prediction.reindex(validation),
                calibration_mean=float(target.reindex(train).mean()),
            )
            metrics.update({f"raw_{name}": value for name, value in raw_metrics.items()})
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
            core_multiscale = multiscale_reconstruction_metrics(
                target.reindex(validation),
                core_prediction.reindex(validation),
                reconstruction_config.evaluation_lowpass_periods,
                calibration_mean=float(target.reindex(train).mean()),
            )
            core_spectral = spectral_reconstruction_metrics(
                target.reindex(validation),
                core_prediction.reindex(validation),
                reconstruction_config.evaluation_period_bands,
            )
            if reconstruction_config.multiresolution.enabled:
                metrics.update({f"core_{name}": value for name, value in core_metrics.items()})
                metrics.update(
                    {f"core_{name}": value for name, value in core_multiscale.items()}
                )
                metrics.update(
                    {f"core_{name}": value for name, value in core_spectral.items()}
                )
                metrics["low_frequency_proxy_count"] = low_frequency_used
                metrics["selected_proxy_constraint_weight"] = (
                    selected_multiresolution.proxy_constraint_weight
                )
                metrics["selected_lowpass_period_years"] = (
                    selected_multiresolution.lowpass_period_years
                )
            metrics.update(
                {
                    "validation_role": "sensitivity_only",
                    "validation_mode": "full_proxy_network",
                    "assessment_metric": "correlation",
                    "fold": fold_number,
                    "validation_start": int(validation.min()),
                    "validation_end": int(validation.max()),
                    "holdout_position": _holdout_position(validation, years),
                    "regression": model.regression_name,
                    "pca_selection": pca_config.selection,
                    "n_components": model.n_components,
                    "kaiser_component_count": int(
                        np.sum(model.eigenvalues > pca_config.kaiser_threshold)
                    ),
                    "alpha": model.alpha,
                    "weighted_proxy_count": float(model.proxy_weights.sum()),
                    "downweighted_proxy_count": int((model.proxy_weights < 1.0).sum()),
                    "amplitude_method": model.amplitude_calibrator.method,
                    "amplitude_reference": model.amplitude_calibrator.reference,
                    "amplitude_reference_overlap": model.amplitude_calibrator.reference_overlap,
                    "amplitude_reference_min_proxy_count": model.amplitude_calibrator.reference_min_proxy_count,
                    "amplitude_slope": model.amplitude_calibrator.slope,
                    "amplitude_intercept": model.amplitude_calibrator.intercept,
                    "selected_proxy_count": len(matrix.columns),
                    "model_proxy_count": len(model.columns),
                    "pids": "|".join(map(str, matrix.columns)),
                    "screening_refit": False,
                }
            )
            rows.append(metrics)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
    return pd.DataFrame(rows)


def blocked_pipeline_validation(
    records: ProxyCollection | Mapping[str, ProxyRecord],
    target: pd.Series,
    screening_config: ScreeningConfig,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
) -> pd.DataFrame:
    """Contiguous sensitivity that repeats proxy screening in each fold."""

    years = _calibration_years(target, reconstruction_config)
    folds = directed_edge_folds(
        years,
        reconstruction_config.external_validation_fraction,
        minimum_train_years=max(20, pca_config.max_components + 3),
    )
    rows: list[dict[str, object]] = []
    for fold_number, (train, validation) in enumerate(folds, start=1):
        try:
            # The index contains only outer-training years, so even an automatic
            # season search cannot see the held-out target values.
            training_target = target.reindex(train).dropna()
            fold_screening = screen_proxies(
                records,
                training_target,
                replace(screening_config, period=None),
            )
            selected = fold_screening.loc[fold_screening["selected"], "pid"].astype(str).tolist()
            if not selected:
                continue
            core_selected, low_resolution_pids = split_resolution_roles(
                records,
                selected,
                pca_config,
                reconstruction_config,
            )
            matrix = build_proxy_matrix(
                records,
                core_selected,
                screening_config,
                fold_screening,
                reconstruction_config.reconstruction_period,
                reconstruction_config.interpolation,
            )
            model = fit_native_pcr(matrix, target, train, pca_config, reconstruction_config)
            core_prediction = model.predict(matrix)
            raw_prediction = model.predict_raw(matrix)
            prediction = core_prediction.reindex(validation)
            core_metrics = reconstruction_metrics(
                target.reindex(validation),
                prediction,
                calibration_mean=float(training_target.mean()),
            )
            low_frequency_used = 0
            low_resolution_records = [records[pid] for pid in low_resolution_pids]
            selected_multiresolution = reconstruction_config.multiresolution
            if low_resolution_records:
                selected_multiresolution, _ = tune_multiresolution_config(
                    matrix,
                    target,
                    train,
                    pca_config,
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
                adjustment = variational_low_frequency_adjustment(
                    core_prediction,
                    low_resolution_records,
                    training_target,
                    config=selected_multiresolution,
                )
                prediction = adjustment.adjusted.reindex(validation)
                if not adjustment.constraints.empty and "status" in adjustment.constraints:
                    low_frequency_used = int((adjustment.constraints["status"] == "used").sum())
            metrics: dict[str, object] = reconstruction_metrics(
                target.reindex(validation),
                prediction,
                calibration_mean=float(training_target.mean()),
            )
            raw_metrics = reconstruction_metrics(
                target.reindex(validation),
                raw_prediction.reindex(validation),
                calibration_mean=float(training_target.mean()),
            )
            metrics.update({f"raw_{name}": value for name, value in raw_metrics.items()})
            metrics.update(
                multiscale_reconstruction_metrics(
                    target.reindex(validation),
                    prediction,
                    reconstruction_config.evaluation_lowpass_periods,
                    calibration_mean=float(training_target.mean()),
                )
            )
            metrics.update(
                spectral_reconstruction_metrics(
                    target.reindex(validation),
                    prediction,
                    reconstruction_config.evaluation_period_bands,
                )
            )
            core_multiscale = multiscale_reconstruction_metrics(
                target.reindex(validation),
                core_prediction.reindex(validation),
                reconstruction_config.evaluation_lowpass_periods,
                calibration_mean=float(training_target.mean()),
            )
            core_spectral = spectral_reconstruction_metrics(
                target.reindex(validation),
                core_prediction.reindex(validation),
                reconstruction_config.evaluation_period_bands,
            )
            if reconstruction_config.multiresolution.enabled:
                metrics.update({f"core_{name}": value for name, value in core_metrics.items()})
                metrics.update(
                    {f"core_{name}": value for name, value in core_multiscale.items()}
                )
                metrics.update(
                    {f"core_{name}": value for name, value in core_spectral.items()}
                )
            metrics.update(
                {
                    "validation_role": "sensitivity_only",
                    "validation_mode": "rescreened_network",
                    "assessment_metric": "correlation",
                    "fold": fold_number,
                    "validation_start": int(validation.min()),
                    "validation_end": int(validation.max()),
                    "holdout_position": _holdout_position(validation, years),
                    "selected_proxy_count": len(selected),
                    "model_proxy_count": len(model.columns),
                    "low_frequency_proxy_count": low_frequency_used,
                    "selected_proxy_constraint_weight": (
                        selected_multiresolution.proxy_constraint_weight
                    ),
                    "selected_lowpass_period_years": (
                        selected_multiresolution.lowpass_period_years
                    ),
                    "regression": model.regression_name,
                    "pca_selection": pca_config.selection,
                    "n_components": model.n_components,
                    "kaiser_component_count": int(
                        np.sum(model.eigenvalues > pca_config.kaiser_threshold)
                    ),
                    "alpha": model.alpha,
                    "weighted_proxy_count": float(model.proxy_weights.sum()),
                    "downweighted_proxy_count": int((model.proxy_weights < 1.0).sum()),
                    "amplitude_method": model.amplitude_calibrator.method,
                    "amplitude_reference": model.amplitude_calibrator.reference,
                    "amplitude_reference_overlap": model.amplitude_calibrator.reference_overlap,
                    "amplitude_reference_min_proxy_count": model.amplitude_calibrator.reference_min_proxy_count,
                    "amplitude_slope": model.amplitude_calibrator.slope,
                    "amplitude_intercept": model.amplitude_calibrator.intercept,
                    "pids": "|".join(selected),
                    "screening_refit": True,
                }
            )
            rows.append(metrics)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
    return pd.DataFrame(rows)


def _moving_block_sample(years: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    if len(years) <= block:
        return years.copy()
    samples: list[int] = []
    while len(samples) < len(years):
        start = int(rng.integers(0, len(years) - block + 1))
        samples.extend(years[start : start + block].tolist())
    return np.asarray(samples[: len(years)], dtype=int)


def summarize_external_sensitivities(
    fold_table: pd.DataFrame,
    reconstruction_config: ReconstructionConfig,
) -> pd.DataFrame:
    """Summarize contiguous holdouts without judging the final reconstruction.

    Correlation is the declared external assessment metric.  CE, RE, error,
    amplitude, and spectral fields remain in the audit output for context, but
    no pass/fail or strong/weak class is derived from them.
    """

    if fold_table.empty:
        return pd.DataFrame(
            columns=(
                "validation_role", "validation_mode", "assessment_metric",
                "fold_count", "median_r", "minimum_r", "maximum_r",
            )
        )
    if "validation_mode" in fold_table:
        groups = fold_table.groupby("validation_mode", sort=False, dropna=False)
    else:
        groups = (("unspecified", fold_table),)
    summaries: list[dict[str, object]] = []
    for mode, table in groups:
        summary: dict[str, object] = {
            "validation_role": "sensitivity_only",
            "validation_mode": str(mode),
            "assessment_metric": "correlation",
            "ce_re_role": "reported_not_used_for_external_judgment",
            "fold_count": len(table),
        }
        for metric in (
            "r", "p_effective", "n_eff", "rmse", "re", "ce",
            "sd_ratio", "variance_ratio",
        ):
            values = pd.to_numeric(
                table.get(metric, pd.Series(dtype=float)), errors="coerce"
            ).dropna()
            summary[f"median_{metric}"] = float(values.median()) if not values.empty else np.nan
            summary[f"mean_{metric}"] = float(values.mean()) if not values.empty else np.nan
            summary[f"minimum_{metric}"] = float(values.min()) if not values.empty else np.nan
            summary[f"maximum_{metric}"] = float(values.max()) if not values.empty else np.nan
            summary[f"q05_{metric}"] = float(values.quantile(0.05)) if not values.empty else np.nan
            summary[f"q95_{metric}"] = float(values.quantile(0.95)) if not values.empty else np.nan
            core_metric = f"core_{metric}"
            if core_metric in table:
                core_values = pd.to_numeric(table[core_metric], errors="coerce").dropna()
                summary[f"median_{core_metric}"] = (
                    float(core_values.median()) if not core_values.empty else np.nan
                )
            raw_metric = f"raw_{metric}"
            if raw_metric in table:
                raw_values = pd.to_numeric(table[raw_metric], errors="coerce").dropna()
                summary[f"median_{raw_metric}"] = (
                    float(raw_values.median()) if not raw_values.empty else np.nan
                )
        for period in reconstruction_config.evaluation_lowpass_periods:
            label = f"{float(period):g}".replace(".", "p")
            for metric in ("r", "sd_ratio", "variance_ratio", "rmse", "re", "ce"):
                name = f"lowpass_{label}y_{metric}"
                values = pd.to_numeric(
                    table.get(name, pd.Series(dtype=float)), errors="coerce"
                ).dropna()
                summary[f"median_{name}"] = float(values.median()) if not values.empty else np.nan
                summary[f"q05_{name}"] = float(values.quantile(0.05)) if not values.empty else np.nan
                summary[f"q95_{name}"] = float(values.quantile(0.95)) if not values.empty else np.nan
                core_name = f"core_{name}"
                if core_name in table:
                    core_values = pd.to_numeric(table[core_name], errors="coerce").dropna()
                    summary[f"median_{core_name}"] = (
                        float(core_values.median()) if not core_values.empty else np.nan
                    )
        for column in table.columns:
            if not column.startswith("spectral_"):
                continue
            values = pd.to_numeric(table[column], errors="coerce").dropna()
            summary[f"median_{column}"] = float(values.median()) if not values.empty else np.nan
            summary[f"q05_{column}"] = float(values.quantile(0.05)) if not values.empty else np.nan
            summary[f"q95_{column}"] = float(values.quantile(0.95)) if not values.empty else np.nan
        if "holdout_position" in table:
            for position in ("early", "late"):
                edge = table.loc[table["holdout_position"] == position]
                for metric in ("r", "p_effective", "rmse", "re", "ce"):
                    values = pd.to_numeric(
                        edge.get(metric, pd.Series(dtype=float)), errors="coerce"
                    ).dropna()
                    summary[f"{position}_holdout_{metric}"] = (
                        float(values.median()) if not values.empty else np.nan
                    )
        summaries.append(summary)
    return pd.DataFrame(summaries)


def reconstruct(
    records: ProxyCollection | Mapping[str, ProxyRecord],
    screening: pd.DataFrame,
    target: pd.Series,
    screening_config: ScreeningConfig,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
) -> ReconstructionResult:
    selected = screening.loc[screening["selected"], "pid"].astype(str).tolist()
    if not selected:
        raise RuntimeError("no proxy passed screening")
    core_selected, low_resolution_pids = split_resolution_roles(
        records,
        selected,
        pca_config,
        reconstruction_config,
    )
    effective_period = reconstruction_config.reconstruction_period
    if reconstruction_config.multiresolution.enabled:
        automatic_period = (
            int(np.floor(min(records[pid].start for pid in selected))),
            int(np.floor(max(records[pid].end for pid in selected))),
        )
        if effective_period is None:
            effective_period = automatic_period
        else:
            effective_period = (
                automatic_period[0] if effective_period[0] is None else effective_period[0],
                automatic_period[1] if effective_period[1] is None else effective_period[1],
            )
    matrix = build_proxy_matrix(
        records,
        core_selected,
        screening_config,
        screening,
        effective_period,
        reconstruction_config.interpolation,
    )
    calibration_years = _calibration_years(target, reconstruction_config)
    calibration_years = calibration_years[np.isin(calibration_years, matrix.index)]
    model = fit_native_pcr(matrix, target, calibration_years, pca_config, reconstruction_config)
    point = model.predict(matrix)
    point_raw = model.predict_raw(matrix)
    low_resolution_records = [records[pid] for pid in low_resolution_pids]
    selected_multiresolution = reconstruction_config.multiresolution
    multiresolution_selection = pd.DataFrame()
    if low_resolution_records:
        selected_multiresolution, multiresolution_selection = tune_multiresolution_config(
            matrix,
            target,
            calibration_years,
            pca_config,
            reconstruction_config,
            low_resolution_records,
            n_components=model.n_components,
            alpha=model.alpha,
            regression_name=model.regression_name,
        )
    validation_tables: list[pd.DataFrame] = []
    if reconstruction_config.full_network_outer_validation:
        validation_tables.append(blocked_validation(
            matrix,
            target,
            pca_config,
            reconstruction_config,
            low_resolution_records=low_resolution_records,
        ))
    if reconstruction_config.rescreen_outer_folds:
        validation_tables.append(blocked_pipeline_validation(
            records,
            target,
            screening_config,
            pca_config,
            reconstruction_config,
        ))
    nonempty_validation = [
        table.dropna(axis=1, how="all")
        for table in validation_tables
        if not table.empty
    ]
    fold_table = (
        pd.concat(nonempty_validation, ignore_index=True, sort=False)
        if nonempty_validation
        else pd.DataFrame()
    )
    validation_summary = summarize_external_sensitivities(
        fold_table, reconstruction_config
    )

    rng = np.random.default_rng(reconstruction_config.random_seed)
    ensemble: list[np.ndarray] = []
    raw_ensemble: list[np.ndarray] = []
    bootstrap_errors: list[str] = []
    for _ in range(reconstruction_config.n_bootstrap):
        sampled = _moving_block_sample(
            calibration_years,
            reconstruction_config.bootstrap_block_years,
            rng,
        )
        try:
            fitted = fit_native_pcr(
                matrix,
                target,
                sampled,
                pca_config,
                reconstruction_config,
                forced_n_components=model.n_components,
                forced_alpha=model.alpha,
                forced_regression=model.regression_name,
                forced_amplitude=model.amplitude_config,
            )
            ensemble.append(fitted.predict(matrix).to_numpy(float))
            raw_ensemble.append(fitted.predict_raw(matrix).to_numpy(float))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            bootstrap_errors.append(f"{type(error).__name__}: {error}")
            continue
    minimum_successes = int(
        np.ceil(
            reconstruction_config.n_bootstrap
            * reconstruction_config.minimum_bootstrap_success_fraction
        )
    )
    if reconstruction_config.n_bootstrap and len(ensemble) < minimum_successes:
        first_error = bootstrap_errors[0] if bootstrap_errors else "unknown error"
        raise RuntimeError(
            f"only {len(ensemble)}/{reconstruction_config.n_bootstrap} bootstrap fits succeeded; "
            f"required at least {minimum_successes}. First failure: {first_error}"
        )
    reconstruction = pd.DataFrame(index=matrix.index)
    if reconstruction_config.amplitude.method != "none":
        reconstruction["point_raw"] = point_raw
    reconstruction["point"] = point
    if ensemble:
        values = np.asarray(ensemble)
        finite_count = np.sum(np.isfinite(values), axis=0)
        usable = finite_count > 0
        for name, quantile in (("median", 0.5), ("q05", 0.05), ("q25", 0.25), ("q75", 0.75), ("q95", 0.95)):
            statistic = np.full(values.shape[1], np.nan)
            statistic[usable] = np.nanquantile(values[:, usable], quantile, axis=0)
            reconstruction[name] = statistic
        if reconstruction_config.amplitude.method != "none":
            raw_values = np.asarray(raw_ensemble)
            raw_statistic = np.full(raw_values.shape[1], np.nan)
            raw_statistic[usable] = np.nanquantile(raw_values[:, usable], 0.5, axis=0)
            reconstruction["median_raw"] = raw_statistic
        reconstruction["ensemble_n"] = finite_count
    else:
        reconstruction["median"] = point
        if reconstruction_config.amplitude.method != "none":
            reconstruction["median_raw"] = point_raw
        reconstruction["q05"] = point
        reconstruction["q25"] = point
        reconstruction["q75"] = point
        reconstruction["q95"] = point
        reconstruction["ensemble_n"] = np.isfinite(point).astype(int)

    low_frequency_constraints = pd.DataFrame()
    low_frequency_observations = pd.DataFrame()
    used_low_resolution_records: list[ProxyRecord] = []
    if reconstruction_config.multiresolution.enabled:
        low_resolution = low_resolution_records
        if low_resolution and selected_multiresolution.proxy_constraint_weight > 0:
            calibration_target = target.reindex(calibration_years).dropna()
            adjustment: LowFrequencyResult = variational_low_frequency_adjustment(
                reconstruction["median"],
                low_resolution,
                calibration_target,
                config=selected_multiresolution,
            )
            reconstruction["median_unadjusted"] = reconstruction["median"]
            reconstruction["low_frequency_increment"] = adjustment.increment
            reconstruction["core_low_frequency"] = adjustment.core_low_frequency
            reconstruction["adjusted_low_frequency"] = adjustment.adjusted_low_frequency
            reconstruction["core_high_frequency"] = adjustment.core_high_frequency
            reconstruction["median"] = adjustment.adjusted
            for quantile in ("point", "q05", "q25", "q75", "q95"):
                shifted = reconstruction[quantile] + adjustment.increment
                reconstruction[quantile] = shifted.combine_first(adjustment.adjusted)
            low_frequency_constraints = adjustment.constraints
            low_frequency_observations = adjustment.observations
            if not adjustment.constraints.empty and "status" in adjustment.constraints:
                used_pids = set(
                    adjustment.constraints.loc[
                        adjustment.constraints["status"] == "used", "pid"
                    ].astype(str)
                )
                used_low_resolution_records = [
                    record for record in low_resolution if record.pid in used_pids
                ]

    availability = pd.DataFrame(
        {
            "available_proxy_count": matrix.notna().sum(axis=1),
            "model_proxy_count": matrix.reindex(columns=model.columns).notna().sum(axis=1),
        }
    )
    availability["effective_model_proxy_count"] = (
        matrix.reindex(columns=model.columns)
        .notna()
        .mul(model.proxy_weights, axis=1)
        .sum(axis=1)
    )
    availability["low_resolution_support_count"] = native_support_counts(
        used_low_resolution_records,
        availability.index,
        maximum_support_multiplier=reconstruction_config.multiresolution.maximum_support_multiplier,
    )
    annual_supported = availability["model_proxy_count"] >= pca_config.min_proxies_per_year
    low_supported = availability["low_resolution_support_count"] > 0
    availability["state_support"] = np.select(
        [annual_supported, low_supported],
        ["annual_core", "low_frequency_only"],
        default="unconstrained",
    )
    reconstruction = reconstruction.reset_index()
    return ReconstructionResult(
        reconstruction=reconstruction,
        model=model,
        validation_folds=fold_table,
        validation_summary=validation_summary,
        model_selection=model.selection_table,
        proxy_weights=model.weight_table,
        availability=availability.reset_index(),
        low_frequency_constraints=low_frequency_constraints,
        low_frequency_observations=low_frequency_observations,
        multiresolution_selection=multiresolution_selection,
    )

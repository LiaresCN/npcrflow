"""Explicit coverage-defined NPCR NESTs fitted entirely in memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import PCAConfig, ReconstructionConfig, ScreeningConfig
from .data import annualize_record
from .low_frequency import LowFrequencyResult, variational_low_frequency_adjustment
from .model import NativePCRModel, fit_native_pcr
from .records import ProxyRecord
from .resolution_nest import (
    ResolutionNestModel,
    fit_resolution_subnests,
    fuse_resolution_subnests,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NestSpec:
    nest_id: str
    start: int
    end: int
    pids: tuple[str, ...]


@dataclass
class FittedNest:
    spec: NestSpec
    core_pids: tuple[str, ...]
    low_resolution_pids: tuple[str, ...]
    model: NativePCRModel
    matrix: pd.DataFrame
    resolution_models: tuple[ResolutionNestModel, ...]
    resolution_constraints: pd.DataFrame
    prediction: pd.Series
    raw_prediction: pd.Series
    selected_multiresolution: object
    low_frequency_constraints: pd.DataFrame
    low_frequency_observations: pd.DataFrame
    internal_row: pd.Series | None


def build_coverage_nests(
    records: Mapping[str, ProxyRecord],
    pids: Sequence[str],
    screening_config: ScreeningConfig,
    reconstruction_config: ReconstructionConfig,
    screening: pd.DataFrame | None = None,
) -> list[NestSpec]:
    """Return the longest unique coverage NEST for every predictor set.

    Coverage uses each annualized record's envelope, matching the original
    WNPSM/PDO logic. Internal gaps are handled later by native-missing PCR or
    the declared bounded Wood/Coral interpolation policy.
    """

    from .reconstruction import _months_from_screening

    month_lookup = _months_from_screening(
        screening if screening is not None else pd.DataFrame()
    )
    bounds: dict[str, tuple[int, int]] = {}
    for pid in pids:
        series = annualize_record(
            records[pid],
            months=month_lookup.get(pid, screening_config.months),
            season_year=screening_config.season_year,
            minimum_month_fraction=screening_config.minimum_month_fraction,
        )
        if not series.empty:
            bounds[pid] = (int(series.index.min()), int(series.index.max()))
    if not bounds:
        return []
    requested = reconstruction_config.reconstruction_period
    requested_start = None if requested is None else requested[0]
    requested_end = None if requested is None else requested[1]
    starts = [max(start, requested_start) if requested_start is not None else start for start, _ in bounds.values()]
    ends = [min(end, requested_end) if requested_end is not None else end for _, end in bounds.values()]
    candidate_starts = sorted(
        {year for year in starts if year >= reconstruction_config.nest.minimum_start_year}
    )
    early = [year for year in starts if year < reconstruction_config.nest.minimum_start_year]
    if early:
        candidate_starts.insert(0, max(early))
    candidate_ends = sorted(set(ends))
    longest_by_network: dict[tuple[str, ...], tuple[int, int, tuple[str, ...]]] = {}
    for start in candidate_starts:
        for end in candidate_ends:
            if end - start < reconstruction_config.nest.minimum_span_years:
                continue
            network = tuple(
                sorted(
                    pid
                    for pid, (record_start, record_end) in bounds.items()
                    if record_start <= start and record_end >= end
                )
            )
            if len(network) < reconstruction_config.nest.minimum_total_proxies:
                continue
            previous = longest_by_network.get(network)
            if previous is None or end - start > previous[1] - previous[0]:
                longest_by_network[network] = (start, end, network)
    ordered = sorted(
        longest_by_network.values(), key=lambda item: (item[0], item[1], item[2])
    )
    return [
        NestSpec(f"nest_{number:03d}", start, end, network)
        for number, (start, end, network) in enumerate(ordered, start=1)
    ]


def _selected_internal_row(table: pd.DataFrame) -> pd.Series | None:
    if table.empty:
        return None
    selected = table
    if "selected" in selected:
        selected = selected.loc[selected["selected"].astype(bool)]
    if selected.empty:
        return None
    if "selection_stage" in selected:
        amplitude = selected.loc[selected["selection_stage"] == "amplitude"]
        if not amplitude.empty:
            return amplitude.iloc[-1]
    return selected.iloc[-1]


def _split_nest_roles(
    records: Mapping[str, ProxyRecord],
    pids: Sequence[str],
    reconstruction_config: ReconstructionConfig,
) -> tuple[list[str], list[str]]:
    if not reconstruction_config.nest.multiresolution_subnests:
        return list(pids), []
    maximum = reconstruction_config.nest.direct_annual_resolution_years
    low = [
        pid for pid in pids
        if np.isfinite(records[pid].resolution) and records[pid].resolution > maximum
    ]
    core = [pid for pid in pids if pid not in low]
    return core, low


def _slice_proxy_matrix(
    matrix: pd.DataFrame,
    pids: Sequence[str],
    start: int,
    end: int,
) -> pd.DataFrame:
    """Slice a once-prepared proxy matrix while retaining its audit metadata."""

    result = matrix.loc[start:end, list(pids)].copy()
    metadata = matrix.attrs.get("proxy_metadata", {})
    result.attrs["proxy_metadata"] = {
        pid: metadata.get(pid, {}) for pid in pids
    }
    for name in ("native_observed_mask", "interpolated_mask"):
        source = matrix.attrs.get(name)
        if isinstance(source, pd.DataFrame):
            result.attrs[name] = source.reindex(
                index=result.index, columns=pids, fill_value=False
            ).copy()
    audit = matrix.attrs.get("interpolation_audit", pd.DataFrame())
    result.attrs["interpolation_audit"] = (
        audit.loc[audit["pid"].isin(pids)].copy()
        if not audit.empty and "pid" in audit else pd.DataFrame()
    )
    standardization = matrix.attrs.get("standardization_audit", pd.DataFrame())
    result.attrs["standardization_audit"] = (
        standardization.loc[standardization["pid"].isin(pids)].copy()
        if not standardization.empty and "pid" in standardization else pd.DataFrame()
    )
    return result


def fit_explicit_nests(
    records: Mapping[str, ProxyRecord],
    pids: Sequence[str],
    screening: pd.DataFrame,
    target: pd.Series,
    train_years: Sequence[int],
    screening_config: ScreeningConfig,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
) -> tuple[list[FittedNest], pd.DataFrame]:
    """Fit each coverage NEST independently and return compact audit rows."""

    from .reconstruction import (
        build_proxy_matrix,
        tune_multiresolution_config,
    )

    eligible_pids = list(pids)
    if not reconstruction_config.multiresolution.enabled:
        eligible_pids = [
            pid for pid in eligible_pids
            if not np.isfinite(records[pid].resolution)
            or records[pid].resolution
            <= reconstruction_config.nest.subnest_max_resolution_years
        ]
    specs = build_coverage_nests(
        records, eligible_pids, screening_config, reconstruction_config, screening
    )
    all_core_pids, _ = _split_nest_roles(
        records, eligible_pids, reconstruction_config
    )
    if not all_core_pids:
        return [], pd.DataFrame()
    global_train = np.asarray(list(train_years), dtype=int)
    prepared_matrix = build_proxy_matrix(
        records,
        all_core_pids,
        screening_config,
        screening,
        reconstruction_config.reconstruction_period,
        reconstruction_config.interpolation,
        reconstruction_config.interpolation_archives,
        reconstruction_config.interpolation_max_gap_years,
        reconstruction_config.interpolation_max_resolution_years,
        reconstruction_config.standardization_period,
        global_train,
    )
    fitted: list[FittedNest] = []
    audit_rows: list[dict[str, object]] = []
    logger.info("Fitting %d coverage NESTs for %d training years", len(specs), len(global_train))
    for nest_number, spec in enumerate(specs, start=1):
        if nest_number == 1 or nest_number % 25 == 0 or nest_number == len(specs):
            logger.info("Coverage NEST %d/%d: %s (%d-%d)", nest_number, len(specs), spec.nest_id, spec.start, spec.end)
        core_pids, low_pids = _split_nest_roles(
            records, spec.pids, reconstruction_config
        )
        base_row: dict[str, object] = {
            "row_type": "coverage_nest",
            "nest_id": spec.nest_id,
            "start": spec.start,
            "end": spec.end,
            "span_years": spec.end - spec.start + 1,
            "proxy_count": len(spec.pids),
            "core_proxy_count": len(core_pids),
            "low_resolution_proxy_count": len(low_pids),
            "pids": "|".join(spec.pids),
            "core_pids": "|".join(core_pids),
            "low_resolution_pids": "|".join(low_pids),
        }
        if len(core_pids) < reconstruction_config.nest.minimum_core_proxies:
            audit_rows.append({**base_row, "accepted": False, "reason": "too_few_core_proxies"})
            continue
        matrix = _slice_proxy_matrix(
            prepared_matrix, core_pids, spec.start, spec.end
        )
        local_train = global_train[
            (global_train >= spec.start)
            & (global_train <= spec.end)
            & np.isin(global_train, matrix.index)
        ]
        if len(local_train) < reconstruction_config.nest.minimum_calibration_years:
            audit_rows.append({**base_row, "accepted": False, "reason": "insufficient_calibration"})
            continue
        try:
            model = fit_native_pcr(
                matrix, target, local_train, pca_config, reconstruction_config
            )
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
                audit_rows.append(
                    {
                        **base_row,
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
                    }
                )
                continue
            prediction = model.predict(matrix)
            raw_prediction = model.predict_raw(matrix)
            resolution_pids = [
                pid for pid in low_pids
                if np.isfinite(records[pid].resolution)
                and records[pid].resolution
                <= reconstruction_config.nest.subnest_max_resolution_years
            ]
            slow_pids = [pid for pid in low_pids if pid not in resolution_pids]
            resolution_models, resolution_audit = fit_resolution_subnests(
                matrix,
                records,
                resolution_pids,
                target,
                local_train,
                spec.start,
                spec.end,
                pca_config,
                reconstruction_config,
            )
            if not resolution_audit.empty:
                for row in resolution_audit.to_dict("records"):
                    audit_rows.append({
                        "row_type": "resolution_subnest",
                        "nest_id": spec.nest_id,
                        "start": spec.start,
                        "end": spec.end,
                        **row,
                    })
            prediction, resolution_constraints = fuse_resolution_subnests(
                prediction, resolution_models, reconstruction_config
            )
            selected_multiresolution = reconstruction_config.multiresolution
            low_constraints = pd.DataFrame()
            low_observations = pd.DataFrame()
            if slow_pids and reconstruction_config.multiresolution.enabled:
                low_records = [records[pid] for pid in slow_pids]
                selected_multiresolution, _ = tune_multiresolution_config(
                    matrix,
                    target,
                    local_train,
                    pca_config,
                    reconstruction_config,
                    low_records,
                    n_components=model.n_components,
                    alpha=model.alpha,
                    regression_name=model.regression_name,
                )
                if selected_multiresolution.proxy_constraint_weight > 0:
                    adjustment: LowFrequencyResult = variational_low_frequency_adjustment(
                        prediction,
                        low_records,
                        target.reindex(local_train).dropna(),
                        config=selected_multiresolution,
                    )
                    prediction = adjustment.adjusted
                    low_constraints = adjustment.constraints.assign(nest_id=spec.nest_id)
                    low_observations = adjustment.observations.assign(nest_id=spec.nest_id)
            fitted.append(
                FittedNest(
                    spec=spec,
                    core_pids=tuple(core_pids),
                    low_resolution_pids=tuple(slow_pids),
                    model=model,
                    matrix=matrix,
                    resolution_models=tuple(resolution_models),
                    resolution_constraints=resolution_constraints.assign(
                        nest_id=spec.nest_id
                    ),
                    prediction=prediction,
                    raw_prediction=raw_prediction,
                    selected_multiresolution=selected_multiresolution,
                    low_frequency_constraints=low_constraints,
                    low_frequency_observations=low_observations,
                    internal_row=internal,
                )
            )
            audit_rows.append(
                {
                    **base_row,
                    "accepted": True,
                    "reason": "accepted",
                    "model_proxy_count": len(model.columns),
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
                    "finite_prediction_years": int(prediction.notna().sum()),
                    "accepted_resolution_subnest_count": len(resolution_models),
                    "low_frequency_weight": (
                        selected_multiresolution.proxy_constraint_weight
                        if slow_pids and reconstruction_config.multiresolution.enabled
                        else 0.0
                    ),
                    "lowpass_period_years": selected_multiresolution.lowpass_period_years,
                }
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            audit_rows.append(
                {
                    **base_row,
                    "accepted": False,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    return fitted, pd.DataFrame(audit_rows)


def combine_fitted_nests(
    fitted: Sequence[FittedNest],
    reconstruction_config: ReconstructionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine accepted NEST predictions without writing per-NEST matrices."""

    if not fitted:
        raise RuntimeError("no explicit NEST passed the internal NPCR thresholds")
    prediction_table = pd.concat(
        {member.spec.nest_id: member.prediction for member in fitted}, axis=1
    ).sort_index()
    raw_table = pd.concat(
        {member.spec.nest_id: member.raw_prediction for member in fitted}, axis=1
    ).sort_index()
    if reconstruction_config.nest.combination == "median":
        point = prediction_table.median(axis=1, skipna=True)
        raw = raw_table.median(axis=1, skipna=True)
    else:
        order = sorted(
            fitted,
            key=lambda member: (-len(member.core_pids), member.spec.start, -member.spec.end),
        )
        point = pd.Series(np.nan, index=prediction_table.index, dtype=float)
        raw = pd.Series(np.nan, index=raw_table.index, dtype=float)
        for member in reversed(order):
            point.update(member.prediction.dropna())
            raw.update(member.raw_prediction.dropna())
    reconstruction = pd.DataFrame(index=prediction_table.index)
    reconstruction["point_raw"] = raw
    reconstruction["point"] = point
    reconstruction["median_raw"] = raw
    reconstruction["median"] = point
    for name in ("q05", "q25", "q75", "q95"):
        reconstruction[name] = point
    reconstruction["ensemble_n"] = prediction_table.notna().sum(axis=1)
    availability = pd.DataFrame(index=prediction_table.index)
    availability["contributing_nest_count"] = prediction_table.notna().sum(axis=1)
    availability["maximum_core_proxy_count"] = 0
    availability["maximum_low_resolution_proxy_count"] = 0
    availability["maximum_resolution_subnest_proxy_count"] = 0
    for member in fitted:
        supported = member.prediction.notna().reindex(availability.index, fill_value=False)
        availability.loc[supported, "maximum_core_proxy_count"] = np.maximum(
            availability.loc[supported, "maximum_core_proxy_count"],
            len(member.core_pids),
        )
        availability.loc[supported, "maximum_low_resolution_proxy_count"] = np.maximum(
            availability.loc[supported, "maximum_low_resolution_proxy_count"],
            len(member.low_resolution_pids),
        )
        resolution_count = len({
            pid for layer in member.resolution_models for pid in layer.native_pids
        })
        availability.loc[supported, "maximum_resolution_subnest_proxy_count"] = np.maximum(
            availability.loc[supported, "maximum_resolution_subnest_proxy_count"],
            resolution_count,
        )
    availability["state_support"] = np.where(
        availability["contributing_nest_count"] > 0,
        "explicit_nest",
        "unconstrained",
    )
    return reconstruction, availability


def _validation_metrics(
    target: pd.Series,
    prediction: pd.Series,
    raw_prediction: pd.Series,
    train: np.ndarray,
    validation: np.ndarray,
    reconstruction_config: ReconstructionConfig,
) -> dict[str, object]:
    from .validation import (
        multiscale_reconstruction_metrics,
        reconstruction_metrics,
        spectral_reconstruction_metrics,
    )

    calibration_mean = float(target.reindex(train).mean())
    metrics: dict[str, object] = reconstruction_metrics(
        target.reindex(validation), prediction.reindex(validation),
        calibration_mean=calibration_mean,
    )
    raw = reconstruction_metrics(
        target.reindex(validation), raw_prediction.reindex(validation),
        calibration_mean=calibration_mean,
    )
    metrics.update({f"raw_{name}": value for name, value in raw.items()})
    metrics.update(multiscale_reconstruction_metrics(
        target.reindex(validation), prediction.reindex(validation),
        reconstruction_config.evaluation_lowpass_periods,
        calibration_mean=calibration_mean,
    ))
    metrics.update(spectral_reconstruction_metrics(
        target.reindex(validation), prediction.reindex(validation),
        reconstruction_config.evaluation_period_bands,
    ))
    return metrics


def explicit_nest_validation(
    records: Mapping[str, ProxyRecord],
    screening: pd.DataFrame,
    target: pd.Series,
    screening_config: ScreeningConfig,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
    *,
    rescreen: bool,
) -> pd.DataFrame:
    """Run early/late one-third holdouts with the explicit-NEST workflow."""

    from .reconstruction import _calibration_years, _holdout_position
    from .screening import screen_proxies
    from .validation import directed_edge_folds

    all_years = _calibration_years(target, reconstruction_config)
    folds = directed_edge_folds(
        all_years,
        reconstruction_config.external_validation_fraction,
        minimum_train_years=max(20, pca_config.max_components + 3),
    )
    rows: list[dict[str, object]] = []
    validation_config = replace(reconstruction_config, n_bootstrap=0)
    for fold_number, (train, validation) in enumerate(folds, start=1):
        try:
            fold_screening = screening
            if rescreen:
                fold_screening = screen_proxies(
                    records,
                    target.reindex(train).dropna(),
                    replace(screening_config, period=None),
                )
            selected = fold_screening.loc[
                fold_screening["selected"], "pid"
            ].astype(str).tolist()
            fitted, audit = fit_explicit_nests(
                records, selected, fold_screening, target, train,
                screening_config, pca_config, validation_config,
            )
            combined, _ = combine_fitted_nests(fitted, validation_config)
            metrics = _validation_metrics(
                target, combined["median"], combined["median_raw"],
                train, validation, reconstruction_config,
            )
            metrics.update({
                "fit_status": "completed",
                "validation_role": "sensitivity_only",
                "validation_mode": (
                    "rescreened_network" if rescreen else "full_proxy_network"
                ),
                "assessment_metric": "correlation",
                "fold": fold_number,
                "validation_start": int(validation.min()),
                "validation_end": int(validation.max()),
                "holdout_position": _holdout_position(validation, all_years),
                "selected_proxy_count": len(selected),
                "candidate_nest_count": int(
                    (audit["row_type"] == "coverage_nest").sum()
                ),
                "accepted_nest_count": int(
                    audit.loc[
                        audit["row_type"] == "coverage_nest", "accepted"
                    ].sum()
                ),
                "screening_refit": rescreen,
            })
            rows.append(metrics)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            rows.append({
                "fit_status": "unavailable",
                "fit_error": f"{type(error).__name__}: {error}",
                "validation_role": "sensitivity_only",
                "validation_mode": "rescreened_network" if rescreen else "full_proxy_network",
                "assessment_metric": "correlation",
                "fold": fold_number,
                "validation_start": int(validation.min()),
                "validation_end": int(validation.max()),
                "holdout_position": _holdout_position(validation, all_years),
                "screening_refit": rescreen,
                "r": np.nan,
                "n": 0,
            })
    return pd.DataFrame(rows)


def _bootstrap_explicit_nests(
    fitted: Sequence[FittedNest],
    records: Mapping[str, ProxyRecord],
    target: pd.Series,
    calibration_years: np.ndarray,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
) -> list[pd.Series]:
    """Refit accepted NESTs in memory and retain only combined replicates."""

    from .reconstruction import _moving_block_sample

    rng = np.random.default_rng(reconstruction_config.random_seed)
    bootstrap_fit_config = replace(reconstruction_config, auto_tune=False)
    replicates: list[pd.Series] = []
    for _ in range(reconstruction_config.n_bootstrap):
        members: list[FittedNest] = []
        for member in fitted:
            local = calibration_years[
                (calibration_years >= member.spec.start)
                & (calibration_years <= member.spec.end)
                & np.isin(calibration_years, member.matrix.index)
            ]
            if len(local) < reconstruction_config.nest.minimum_calibration_years:
                continue
            sampled = _moving_block_sample(
                local, reconstruction_config.bootstrap_block_years, rng
            )
            try:
                model = fit_native_pcr(
                    member.matrix, target, sampled, pca_config,
                    bootstrap_fit_config,
                    forced_n_components=member.model.n_components,
                    forced_alpha=member.model.alpha,
                    forced_regression=member.model.regression_name,
                    forced_amplitude=member.model.amplitude_config,
                )
                prediction = model.predict(member.matrix)
                resolution_models: list[ResolutionNestModel] = []
                calibration_set = set(int(year) for year in local)
                for layer in member.resolution_models:
                    window_train = np.asarray([
                        int(index)
                        for index, row in layer.windows.iterrows()
                        if set(
                            range(int(row.window_start), int(row.window_end) + 1)
                        ).issubset(calibration_set)
                        and np.isfinite(layer.target.loc[index])
                        and layer.matrix.loc[index, list(layer.native_pids)].notna().any()
                    ], dtype=int)
                    if len(window_train) < 4:
                        continue
                    sampled_windows = _moving_block_sample(
                        window_train,
                        max(
                            1,
                            reconstruction_config.bootstrap_block_years
                            // layer.resolution_years,
                        ),
                        rng,
                    )
                    try:
                        layer_model = fit_native_pcr(
                            layer.matrix,
                            layer.target,
                            sampled_windows,
                            pca_config,
                            bootstrap_fit_config,
                            forced_n_components=layer.model.n_components,
                            forced_alpha=layer.model.alpha,
                            forced_regression=layer.model.regression_name,
                            forced_amplitude=layer.model.amplitude_config,
                        )
                        resolution_models.append(replace(
                            layer,
                            model=layer_model,
                            prediction=layer_model.predict(layer.matrix).where(
                                layer.matrix[list(layer.native_pids)].notna().any(axis=1)
                            ),
                        ))
                    except (RuntimeError, ValueError, np.linalg.LinAlgError):
                        continue
                prediction, _ = fuse_resolution_subnests(
                    prediction, resolution_models, reconstruction_config
                )
                if (
                    member.low_resolution_pids
                    and reconstruction_config.multiresolution.enabled
                    and member.selected_multiresolution.proxy_constraint_weight > 0
                ):
                    adjustment = variational_low_frequency_adjustment(
                        prediction,
                        [records[pid] for pid in member.low_resolution_pids],
                        target.reindex(sampled).dropna(),
                        config=member.selected_multiresolution,
                    )
                    prediction = adjustment.adjusted
                members.append(replace(
                    member,
                    model=model,
                    prediction=prediction,
                    raw_prediction=model.predict_raw(member.matrix),
                ))
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
        if members:
            replicate, _ = combine_fitted_nests(members, reconstruction_config)
            replicates.append(replicate["median"])
    return replicates


def reconstruct_explicit_nests(
    records: Mapping[str, ProxyRecord],
    screening: pd.DataFrame,
    target: pd.Series,
    screening_config: ScreeningConfig,
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
):
    """Run coverage NEST -> per-NEST PCR -> CE/RE -> combination in memory."""

    from .reconstruction import (
        ReconstructionResult,
        _calibration_years,
        _longest_contiguous_annual_period,
        summarize_external_sensitivities,
    )

    selected = screening.loc[screening["selected"], "pid"].astype(str).tolist()
    if not selected:
        raise RuntimeError("no proxy passed screening")
    calibration_years = _calibration_years(target, reconstruction_config)
    fitted, nest_summary = fit_explicit_nests(
        records, selected, screening, target, calibration_years,
        screening_config, pca_config, reconstruction_config,
    )
    if not nest_summary.empty and "row_type" in nest_summary:
        nest_summary = nest_summary.assign(
            _row_order=nest_summary["row_type"].map(
                {"coverage_nest": 0, "resolution_subnest": 1}
            ).fillna(2)
        )
        sort_columns = ["nest_id", "_row_order"]
        if "resolution_years" in nest_summary:
            sort_columns.append("resolution_years")
        nest_summary = (
            nest_summary.sort_values(sort_columns, na_position="first")
            .drop(columns="_row_order")
            .reset_index(drop=True)
        )
    reconstruction, availability = combine_fitted_nests(
        fitted, reconstruction_config
    )

    replicates = _bootstrap_explicit_nests(
        fitted, records, target, calibration_years,
        pca_config, reconstruction_config,
    )
    required = int(np.ceil(
        reconstruction_config.n_bootstrap
        * reconstruction_config.minimum_bootstrap_success_fraction
    ))
    if reconstruction_config.n_bootstrap and len(replicates) < required:
        raise RuntimeError(
            f"only {len(replicates)}/{reconstruction_config.n_bootstrap} "
            f"explicit-NEST bootstrap replicates succeeded; required {required}"
        )
    if replicates:
        values = pd.concat(replicates, axis=1).reindex(reconstruction.index).to_numpy(float)
        finite = np.isfinite(values).sum(axis=1)
        usable = finite > 0
        for name, quantile in (
            ("median", 0.50), ("q05", 0.05), ("q25", 0.25),
            ("q75", 0.75), ("q95", 0.95),
        ):
            statistic = np.full(len(values), np.nan)
            statistic[usable] = np.nanquantile(values[usable], quantile, axis=1)
            reconstruction[name] = statistic
        reconstruction["ensemble_n"] = finite

    validation_tables: list[pd.DataFrame] = []
    if reconstruction_config.full_network_outer_validation:
        validation_tables.append(explicit_nest_validation(
            records, screening, target, screening_config, pca_config,
            reconstruction_config, rescreen=False,
        ))
    if reconstruction_config.rescreen_outer_folds:
        validation_tables.append(explicit_nest_validation(
            records, screening, target, screening_config, pca_config,
            reconstruction_config, rescreen=True,
        ))
    nonempty = [table for table in validation_tables if not table.empty]
    validation_folds = (
        pd.concat(nonempty, ignore_index=True, sort=False)
        if nonempty else pd.DataFrame()
    )

    if reconstruction_config.retain_longest_annual_segment:
        start, end = _longest_contiguous_annual_period(reconstruction["median"])
        reconstruction = reconstruction.loc[start:end]
        availability = availability.loc[start:end]

    selections: list[pd.DataFrame] = []
    weights: list[pd.DataFrame] = []
    multi: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    constraints: list[pd.DataFrame] = []
    observations: list[pd.DataFrame] = []
    for member in fitted:
        selections.append(
            member.model.selection_table.assign(
                nest_id=member.spec.nest_id,
                nest_layer="annual",
                resolution_years=1,
            )
        )
        weights.append(member.model.weight_table.assign(
            nest_id=member.spec.nest_id,
            nest_layer="annual",
            resolution_years=1,
        ))
        for layer in member.resolution_models:
            selections.append(layer.model.selection_table.assign(
                nest_id=member.spec.nest_id,
                nest_layer="resolution_subnest",
                resolution_years=layer.resolution_years,
                native_pids="|".join(layer.native_pids),
            ))
            weights.append(layer.model.weight_table.assign(
                nest_id=member.spec.nest_id,
                nest_layer="resolution_subnest",
                resolution_years=layer.resolution_years,
            ))
        audit = member.matrix.attrs.get("interpolation_audit", pd.DataFrame()).copy()
        if not audit.empty:
            audits.append(audit)
        if not member.low_frequency_constraints.empty:
            constraints.append(member.low_frequency_constraints)
        if not member.low_frequency_observations.empty:
            observations.append(member.low_frequency_observations)
        # Per-window rows remain in memory. Saving them once for every
        # overlapping coverage NEST would recreate the large intermediates
        # this implementation is designed to remove; compact counts live in
        # nest_summary.csv instead.
        multi.append(pd.DataFrame([{
            "nest_id": member.spec.nest_id,
            "selected": True,
            "proxy_constraint_weight": (
                member.selected_multiresolution.proxy_constraint_weight
                if member.low_resolution_pids
                and reconstruction_config.multiresolution.enabled
                else 0.0
            ),
            "lowpass_period_years": member.selected_multiresolution.lowpass_period_years,
        }]))

    reference = max(fitted, key=lambda member: len(member.core_pids))
    return ReconstructionResult(
        reconstruction=reconstruction.reset_index(),
        model=reference.model,
        validation_folds=validation_folds,
        validation_summary=summarize_external_sensitivities(
            validation_folds, reconstruction_config
        ),
        model_selection=pd.concat(selections, ignore_index=True, sort=False),
        proxy_weights=pd.concat(weights, ignore_index=True, sort=False),
        availability=availability.reset_index(),
        interpolation_audit=(
            pd.concat(audits, ignore_index=True, sort=False).drop_duplicates()
            if audits else pd.DataFrame()
        ),
        low_frequency_constraints=(
            pd.concat(constraints, ignore_index=True, sort=False)
            if constraints else pd.DataFrame()
        ),
        low_frequency_observations=(
            pd.concat(observations, ignore_index=True, sort=False)
            if observations else pd.DataFrame()
        ),
        multiresolution_selection=pd.concat(multi, ignore_index=True, sort=False),
        nest_summary=nest_summary,
    )

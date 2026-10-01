"""One-call screening, reconstruction, diagnostics, and compact output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy import signal

from .config import PipelineConfig, ProxyFilterConfig
from .data import load_observations, load_proxy_frame, load_proxy_workbook
from .deduplicate import deduplicate_frame
from .diagnostics import (
    observation_fit_table,
    plot_observation_diagnostics,
    primary_reconstruction_summary,
)
from .records import ProxyCollection, ProxyRecord
from .reconstruction import ReconstructionResult, reconstruct, split_resolution_roles
from .screening import screen_proxies
from .sensitivity import run_network_sensitivities


@dataclass
class PipelineResult:
    target: pd.Series
    duplicate_report: pd.DataFrame
    duplicate_groups: pd.DataFrame
    source_qc: pd.DataFrame
    screening: pd.DataFrame
    reconstruction: ReconstructionResult
    sensitivities: dict[str, pd.DataFrame]
    output_paths: dict[str, Path]


def _match(value: str, choices: tuple[str, ...]) -> bool:
    return not choices or str(value).strip().lower() in {choice.strip().lower() for choice in choices}


def filter_records_with_report(
    records: ProxyCollection, config: ProxyFilterConfig
) -> tuple[ProxyCollection, pd.DataFrame]:
    """Apply metadata filters and retain an auditable record-level decision."""

    selected: list[ProxyRecord] = []
    decisions: list[dict[str, object]] = []
    for record in records.values():
        reasons: list[str] = []
        if (
            config.maximum_plausible_year is not None
            and record.end > config.maximum_plausible_year
        ):
            reasons.append("end_after_maximum_plausible_year")
        if (
            config.maximum_resolution_years is not None
            and (
                not np.isfinite(record.resolution)
                or record.resolution > config.maximum_resolution_years
            )
        ):
            reasons.append("above_maximum_resolution")
        if config.latitude is not None and not (config.latitude[0] <= record.lat <= config.latitude[1]):
            reasons.append("outside_latitude_filter")
        if config.longitude is not None:
            start, end = config.longitude
            longitude = ((record.lon + 180.0) % 360.0) - 180.0
            start = ((start + 180.0) % 360.0) - 180.0
            end = ((end + 180.0) % 360.0) - 180.0
            inside = start <= longitude <= end if start <= end else longitude >= start or longitude <= end
            if not inside:
                reasons.append("outside_longitude_filter")
        if not _match(record.archive, config.archives):
            reasons.append("archive_filter")
        if not _match(record.proxy, config.proxies):
            reasons.append("proxy_type_filter")
        climate = str(record.metadata.get("climateInterpretation_variable", ""))
        if not _match(climate, config.climate_variables):
            reasons.append("climate_variable_filter")
        included = not reasons
        if included:
            selected.append(record)
        decisions.append(
            {
                "pid": record.pid,
                "included": included,
                "exclusion_reasons": "|".join(reasons),
                "archive": record.archive,
                "proxy": record.proxy,
                "climate_variable": climate,
                "lat": record.lat,
                "lon": record.lon,
                "start": record.start,
                "end": record.end,
                "native_resolution": record.resolution,
                "n_observations": record.time.size,
            }
        )
    return ProxyCollection(selected), pd.DataFrame(decisions)


def filter_records(records: ProxyCollection, config: ProxyFilterConfig) -> ProxyCollection:
    """Backward-compatible collection-only metadata filter."""

    return filter_records_with_report(records, config)[0]


def prepare_target(source, config: PipelineConfig) -> pd.Series:
    target = load_observations(
        source,
        year_col=config.target.year_column,
        value_col=config.target.value_column,
        months=config.target.months,
        season_year=config.target.season_year,
        minimum_month_fraction=config.target.minimum_month_fraction,
    )
    values = target.to_numpy(float)
    if config.target.detrend:
        finite = np.isfinite(values)
        values[finite] = signal.detrend(values[finite], type="linear")
    if config.target.lowpass_years is not None:
        finite = np.isfinite(values)
        if finite.sum() < max(12, config.target.lowpass_order * 3):
            raise ValueError("too few target years for the requested low-pass filter")
        normalized_cutoff = 2.0 / config.target.lowpass_years
        sos = signal.butter(
            config.target.lowpass_order,
            normalized_cutoff,
            btype="lowpass",
            output="sos",
        )
        values[finite] = signal.sosfiltfilt(sos, values[finite])
    return pd.Series(values, index=target.index, name=target.name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_proxy_map(screening: pd.DataFrame, path: str | Path) -> Path:
    """Write one compact selected-proxy map; no diagnostic plot cascade."""

    import matplotlib.pyplot as plt

    selected = screening[screening["selected"]].copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
    with plt.rc_context(font):
        figure = plt.figure(figsize=(10, 5.2))
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            axes = figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            try:
                axes.add_feature(cfeature.LAND, facecolor="#eeeeee")
                axes.coastlines(linewidth=0.6)
            except Exception:
                pass
            transform = ccrs.PlateCarree()
            for archive, group in selected.groupby("archive"):
                axes.scatter(group["lon"], group["lat"], s=30, alpha=0.8, label=str(archive), transform=transform)
            axes.set_global()
            axes.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        except Exception:
            axes = figure.add_subplot(1, 1, 1)
            for archive, group in selected.groupby("archive"):
                axes.scatter(group["lon"], group["lat"], s=30, alpha=0.8, label=str(archive))
            axes.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude")
            axes.grid(alpha=0.3)
        axes.set_title(f"Selected proxy network (n={len(selected)})")
        if not selected.empty:
            axes.legend(loc="best", fontsize=8, frameon=False)
        figure.tight_layout()
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
    return path


def save_pipeline_result(
    result: PipelineResult,
    config: PipelineConfig,
    proxy_source: str | Path,
    observation_source: str | Path,
) -> dict[str, Path]:
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if config.output.save_screening:
        paths["screening"] = output / "proxy_screening.csv"
        result.screening.to_csv(paths["screening"], index=False)
    if config.output.save_reconstruction:
        paths["reconstruction"] = output / "reconstruction.csv"
        result.reconstruction.reconstruction.to_csv(paths["reconstruction"], index=False)
    if config.output.save_metrics:
        if not result.duplicate_report.empty:
            paths["duplicate_report"] = output / "dropped_exact_duplicates.csv"
            result.duplicate_report.to_csv(paths["duplicate_report"], index=False)
        if not result.duplicate_groups.empty:
            paths["duplicate_groups"] = output / "exact_duplicate_groups.csv"
            result.duplicate_groups.to_csv(paths["duplicate_groups"], index=False)
        paths["source_qc"] = output / "source_qc.csv"
        result.source_qc.to_csv(paths["source_qc"], index=False)
        paths["validation"] = output / "validation_folds.csv"
        result.reconstruction.validation_folds.to_csv(paths["validation"], index=False)
        if "validation_mode" in result.reconstruction.validation_folds:
            for mode, table in result.reconstruction.validation_folds.groupby(
                "validation_mode", sort=False
            ):
                key = f"validation_{mode}"
                paths[key] = output / f"validation_{mode}.csv"
                table.to_csv(paths[key], index=False)
        paths["validation_summary"] = output / "validation_summary.csv"
        result.reconstruction.validation_summary.to_csv(paths["validation_summary"], index=False)
        paths["observation_fit"] = output / "observation_fit.csv"
        observation_fit_table(
            result.target,
            result.reconstruction.reconstruction,
            config.reconstruction,
        ).to_csv(paths["observation_fit"], index=False)
        paths["primary_summary"] = output / "primary_reconstruction_summary.csv"
        primary_reconstruction_summary(
            result.target,
            result.reconstruction.reconstruction,
            result.reconstruction.model_selection,
            config.reconstruction,
        ).to_csv(paths["primary_summary"], index=False)
        paths["model_selection"] = output / "model_selection.csv"
        result.reconstruction.model_selection.to_csv(paths["model_selection"], index=False)
        paths["proxy_weights"] = output / "proxy_weights.csv"
        result.reconstruction.proxy_weights.to_csv(paths["proxy_weights"], index=False)
        paths["availability"] = output / "proxy_availability.csv"
        result.reconstruction.availability.to_csv(paths["availability"], index=False)
        if not result.reconstruction.low_frequency_constraints.empty:
            paths["low_frequency_constraints"] = output / "low_frequency_constraints.csv"
            result.reconstruction.low_frequency_constraints.to_csv(paths["low_frequency_constraints"], index=False)
        if not result.reconstruction.low_frequency_observations.empty:
            paths["low_frequency_observations"] = output / "low_frequency_observations.csv"
            result.reconstruction.low_frequency_observations.to_csv(
                paths["low_frequency_observations"], index=False
            )
        if not result.reconstruction.multiresolution_selection.empty:
            paths["multiresolution_selection"] = output / "multiresolution_selection.csv"
            result.reconstruction.multiresolution_selection.to_csv(
                paths["multiresolution_selection"], index=False
            )
        for name, table in result.sensitivities.items():
            paths[f"sensitivity_{name}"] = output / f"sensitivity_{name}.csv"
            table.to_csv(paths[f"sensitivity_{name}"], index=False)
    if config.output.save_proxy_map:
        paths["proxy_map"] = output / f"proxy_map.{config.output.figure_format}"
        plot_proxy_map(result.screening, paths["proxy_map"])
    if config.output.save_observation_plot:
        paths["observation_plot"] = output / f"observation_comparison.{config.output.figure_format}"
        plot_observation_diagnostics(
            result.target,
            result.reconstruction.reconstruction,
            result.reconstruction.validation_folds,
            config.reconstruction,
            paths["observation_plot"],
            show_external_sensitivities=config.output.show_external_sensitivities,
        )
    proxy_path = Path(proxy_source)
    observation_path = Path(observation_source)
    selected_multiresolution = result.reconstruction.multiresolution_selection
    if not selected_multiresolution.empty and "selected" in selected_multiresolution:
        selected_rows = selected_multiresolution[
            selected_multiresolution["selected"].astype(bool)
        ]
    else:
        selected_rows = pd.DataFrame()
    selected_proxy_weight = config.reconstruction.multiresolution.proxy_constraint_weight
    selected_lowpass_period = config.reconstruction.multiresolution.lowpass_period_years
    if not config.reconstruction.multiresolution.enabled:
        selected_proxy_weight = 0.0
    if not selected_rows.empty:
        selected_proxy_weight = float(selected_rows.iloc[0]["proxy_constraint_weight"])
        selected_lowpass_period = float(selected_rows.iloc[0]["lowpass_period_years"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package": "npcrflow",
        "python": platform.python_version(),
        "proxy_source": str(proxy_path.resolve()),
        "proxy_sha256": _sha256(proxy_path),
        "observation_source": str(observation_path.resolve()),
        "observation_sha256": _sha256(observation_path),
        "config": asdict(config),
        "raw_proxy_count": len(result.source_qc) + len(result.duplicate_report),
        "exact_duplicate_count": len(result.duplicate_report),
        "loaded_proxy_count_after_deduplication": len(result.source_qc),
        "metadata_filter_proxy_count": int(result.source_qc["included"].sum()),
        "selected_proxy_count": int(result.screening["selected"].sum()),
        "model_proxy_count": len(result.reconstruction.model.columns),
        "n_components": result.reconstruction.model.n_components,
        "pca_eigenvalues": result.reconstruction.model.eigenvalues.tolist(),
        "kaiser_threshold": config.pca.kaiser_threshold,
        "kaiser_component_count": int(
            np.sum(
                result.reconstruction.model.eigenvalues
                > config.pca.kaiser_threshold
            )
        ),
        "regression": result.reconstruction.model.regression_name,
        "alpha": result.reconstruction.model.alpha,
        "amplitude_method": result.reconstruction.model.amplitude_calibrator.method,
        "amplitude_reference": result.reconstruction.model.amplitude_calibrator.reference,
        "amplitude_reference_overlap": result.reconstruction.model.amplitude_calibrator.reference_overlap,
        "amplitude_reference_min_proxy_count": result.reconstruction.model.amplitude_calibrator.reference_min_proxy_count,
        "amplitude_slope": result.reconstruction.model.amplitude_calibrator.slope,
        "amplitude_intercept": result.reconstruction.model.amplitude_calibrator.intercept,
        "amplitude_availability_knots": result.reconstruction.model.amplitude_calibrator.availability_knots,
        "amplitude_slope_knots": result.reconstruction.model.amplitude_calibrator.slope_knots,
        "weighted_proxy_count": float(result.reconstruction.model.proxy_weights.sum()),
        "downweighted_proxy_count": int((result.reconstruction.model.proxy_weights < 1.0).sum()),
        "external_validation_role": "sensitivity_only",
        "external_validation_assessment_metric": "correlation",
        "external_validation_modes": (
            result.reconstruction.validation_summary["validation_mode"].astype(str).tolist()
            if "validation_mode" in result.reconstruction.validation_summary
            else []
        ),
        "state_timestep_years": config.reconstruction.multiresolution.state_timestep_years,
        "selected_proxy_constraint_weight": selected_proxy_weight,
        "selected_lowpass_period_years": selected_lowpass_period,
        "saved_nests": False,
    }
    paths["manifest"] = output / "manifest.json"
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return paths


def run_pipeline(
    proxy_source: str | Path,
    observation_source: str | Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the complete workflow from raw files with one function call."""

    config = config or PipelineConfig()
    proxy_path = Path(proxy_source)
    duplicate_report = pd.DataFrame()
    duplicate_groups = pd.DataFrame()
    if proxy_path.suffix.lower() in {".xlsx", ".xls"}:
        loaded_records = load_proxy_workbook(proxy_path)
    else:
        raw_frame = pd.read_pickle(proxy_path)
        is_dod2k = {"year", "paleoData_values", "datasetId"}.issubset(raw_frame.columns)
        if config.proxy_filter.deduplicate_exact and is_dod2k:
            raw_frame, duplicate_report, duplicate_groups = deduplicate_frame(
                raw_frame,
                include_near=config.proxy_filter.deduplicate_near,
            )
        loaded_records = load_proxy_frame(raw_frame)
    records, source_qc = filter_records_with_report(loaded_records, config.proxy_filter)
    target = prepare_target(observation_source, config)
    screening = screen_proxies(records, target, config.screening)
    selected_pids = screening.loc[screening["selected"], "pid"].astype(str).tolist()
    core_pids, low_frequency_pids = split_resolution_roles(
        records,
        selected_pids,
        config.pca,
        config.reconstruction,
    )
    screening["reconstruction_role"] = "not_selected"
    screening.loc[screening["pid"].isin(core_pids), "reconstruction_role"] = "pcr_native"
    screening.loc[
        screening["pid"].isin(low_frequency_pids), "reconstruction_role"
    ] = "lowpass_native_window"
    reconstruction = reconstruct(
        records,
        screening,
        target,
        config.screening,
        config.pca,
        config.reconstruction,
    )
    sensitivities = run_network_sensitivities(
        records,
        screening,
        target,
        config.screening,
        config.pca,
        config.reconstruction,
        config.sensitivity,
    )
    result = PipelineResult(
        target=target,
        duplicate_report=duplicate_report,
        duplicate_groups=duplicate_groups,
        source_qc=source_qc,
        screening=screening,
        reconstruction=reconstruction,
        sensitivities=sensitivities,
        output_paths={},
    )
    result.output_paths = save_pipeline_result(result, config, proxy_source, observation_source)
    return result

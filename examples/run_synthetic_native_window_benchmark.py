"""Controlled native-window test of low-frequency cycle recovery.

The held-out target is never used in calibration or tuning. Sparse proxy
samples remain at their native timestamps and constrain window means; they are
never interpolated to annual values.
"""

from dataclasses import replace
from multiprocessing import Pool
import os
from pathlib import Path

import numpy as np
import pandas as pd

from npcrflow import MultiresolutionConfig, PCAConfig, ReconstructionConfig
from npcrflow.low_frequency import variational_low_frequency_adjustment
from npcrflow.model import fit_native_pcr
from npcrflow.records import ProxyRecord
from npcrflow.reconstruction import tune_multiresolution_config
from npcrflow.validation import (
    multiscale_reconstruction_metrics,
    reconstruction_metrics,
    spectral_reconstruction_metrics,
)


PROJECT = Path("/share/home/lrs/codex/projects/npcrflow")
ROOT = PROJECT / "results" / "synthetic_native_window_benchmark"
YEARS = np.arange(1800, 2001)
TRAIN = YEARS[YEARS <= 1899]
VALIDATION = YEARS[YEARS >= 1900]
REPEATS = 20


def _window_mean(values: np.ndarray, center: float, width: float) -> float:
    use = (YEARS >= center - width / 2.0) & (YEARS < center + width / 2.0)
    return float(np.mean(values[use]))


def _low_records(
    low_truth: np.ndarray,
    count: int,
    resolution: int,
    noise_fraction: float,
    rng: np.random.Generator,
) -> list[ProxyRecord]:
    records: list[ProxyRecord] = []
    for number in range(count):
        offset = (number + 0.5) * resolution / max(count, 1)
        times = np.arange(YEARS[0] + offset, YEARS[-1] + 1, resolution)
        native = np.asarray(
            [_window_mean(low_truth, time, resolution) for time in times],
            dtype=float,
        )
        scale = rng.uniform(0.7, 1.3) * (-1.0 if number % 2 else 1.0)
        intercept = rng.normal(0.0, 0.3)
        values = intercept + scale * native
        values += rng.normal(0.0, noise_fraction * np.std(native), len(values))
        records.append(ProxyRecord(f"low_{number + 1}", times, values))
    return records


def _metrics(observed: pd.Series, predicted: pd.Series, calibration_mean: float) -> dict[str, float]:
    result = reconstruction_metrics(observed, predicted, calibration_mean=calibration_mean)
    result.update(
        multiscale_reconstruction_metrics(
            observed,
            predicted,
            (10.0, 20.0),
            calibration_mean=calibration_mean,
        )
    )
    result.update(spectral_reconstruction_metrics(observed, predicted, ((10.0, 30.0),)))
    return result


def run_one(
    *,
    repeat: int,
    proxy_count: int,
    resolution: int,
    noise_fraction: float,
) -> dict[str, object]:
    # Pair every resolution/network/noise scenario to the same annual core for
    # a given repeat. A separate deterministic stream controls only the sparse
    # proxy system, so network-count contrasts cannot be caused by a different
    # core realization.
    core_rng = np.random.default_rng(20260926 + repeat * 1000)
    low_rng = np.random.default_rng(
        20260926 + repeat * 1000 + proxy_count * 100 + resolution * 10
    )
    step = np.arange(len(YEARS), dtype=float)
    low_truth = np.sin(2 * np.pi * step / 25.0) + 0.30 * np.sin(
        2 * np.pi * step / 60.0 + 0.4
    )
    high_truth = 0.35 * np.sin(2 * np.pi * step / 4.0 + 0.2)
    high_truth += 0.15 * np.sin(2 * np.pi * step / 7.0)
    target = pd.Series(low_truth + high_truth, index=YEARS, name="truth")

    # The annual core network resolves only high frequencies. It contains no
    # injected 25-year component, so any held-out cycle recovery must come
    # from the native-window low-resolution observations rather than from an
    # easier annual predictor.
    matrix = pd.DataFrame(
        {
            "annual_1": high_truth + core_rng.normal(0, 0.22, len(YEARS)),
            "annual_2": -0.8 * high_truth + core_rng.normal(0, 0.22, len(YEARS)),
            "annual_3": 0.5 * high_truth + core_rng.normal(0, 0.25, len(YEARS)),
        },
        index=YEARS,
    )
    matrix.loc[YEARS[::11], "annual_2"] = np.nan
    matrix.loc[YEARS[3::13], "annual_3"] = np.nan
    missing_before = int(matrix.isna().sum().sum())

    pca = PCAConfig(
        method="pairwise",
        selection="fixed",
        n_components=2,
        max_components=2,
        min_pairwise_overlap=20,
        min_proxies_per_year=1,
    )
    reconstruction = ReconstructionConfig(
        calibration_period=(int(TRAIN.min()), int(TRAIN.max())),
        interpolation="none",
        regression="ridge",
        ridge_alphas=(0.1,),
        auto_tune=False,
        validation_block_years=25,
        n_bootstrap=0,
        multiresolution=MultiresolutionConfig(
            enabled=True,
            regression_max_resolution_years=10.0,
            lowpass_period_years=10.0,
            smoothness_multiplier=1.0,
            core_anchor_weight=1.0,
            proxy_constraint_weight=1.0,
            auto_tune=True,
            proxy_constraint_weight_candidates=(0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
            lowpass_period_candidates=(10.0, 20.0),
            selection_lowpass_period_years=10.0,
            minimum_tuning_folds=2,
            minimum_calibration_overlap=4,
            maximum_support_multiplier=3.0,
            maximum_increment_ratio=2.0,
            preserve_calibration_mean=True,
        ),
    )
    model = fit_native_pcr(matrix, target, TRAIN, pca, reconstruction)
    core = model.predict(matrix)
    records = _low_records(low_truth, proxy_count, resolution, noise_fraction, low_rng)
    selected, _ = tune_multiresolution_config(
        matrix,
        target,
        TRAIN,
        pca,
        reconstruction,
        records,
        n_components=model.n_components,
        alpha=model.alpha,
        regression_name=model.regression_name,
    )
    adjustment = variational_low_frequency_adjustment(
        core,
        records,
        target.reindex(TRAIN),
        config=selected,
    )
    if int(matrix.isna().sum().sum()) != missing_before:
        raise RuntimeError("synthetic benchmark changed native missing values")
    if any(len(record.time) != len(record.value) for record in records):
        raise RuntimeError("synthetic benchmark changed a native proxy record")

    observed = target.reindex(VALIDATION)
    calibration_mean = float(target.reindex(TRAIN).mean())
    core_metrics = _metrics(observed, core.reindex(VALIDATION), calibration_mean)
    adjusted_metrics = _metrics(
        observed,
        adjustment.adjusted.reindex(VALIDATION),
        calibration_mean,
    )
    row: dict[str, object] = {
        "repeat": repeat,
        "proxy_count": proxy_count,
        "resolution_years": resolution,
        "noise_fraction": noise_fraction,
        "native_sample_count": int(sum(len(record.time) for record in records)),
        "selected_weight": selected.proxy_constraint_weight,
        "selected_period_years": selected.lowpass_period_years,
        "used_proxy_count": int(
            (adjustment.constraints.get("status", pd.Series(dtype=str)) == "used").sum()
        ),
    }
    for name, value in core_metrics.items():
        row[f"core_{name}"] = value
    for name, value in adjusted_metrics.items():
        row[f"adjusted_{name}"] = value
        if name in core_metrics and np.isscalar(value):
            try:
                row[f"delta_{name}"] = float(value) - float(core_metrics[name])
            except (TypeError, ValueError):
                pass
    core_peak_error = core_metrics["spectral_10_30y_peak_period_error"]
    adjusted_peak_error = adjusted_metrics["spectral_10_30y_peak_period_error"]
    row["peak_error_reduction"] = float(core_peak_error) - float(adjusted_peak_error)
    row["amplitude_error_reduction"] = abs(
        1.0 - float(core_metrics["spectral_10_30y_amplitude_ratio"])
    ) - abs(1.0 - float(adjusted_metrics["spectral_10_30y_amplitude_ratio"]))
    return row


def _run_task(task: tuple[int, int, int, float]) -> dict[str, object]:
    repeat, proxy_count, resolution, noise_fraction = task
    return run_one(
        repeat=repeat,
        proxy_count=proxy_count,
        resolution=resolution,
        noise_fraction=noise_fraction,
    )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [
        (repeat, proxy_count, resolution, noise_fraction)
        for resolution in (12, 20)
        for proxy_count in (1, 3, 5)
        for noise_fraction in (0.1, 0.5, 1.0)
        for repeat in range(1, REPEATS + 1)
    ]
    workers = max(1, int(os.environ.get("NPCR_SYNTH_WORKERS", "1")))
    if workers == 1:
        rows = [_run_task(task) for task in tasks]
    else:
        with Pool(processes=workers) as pool:
            rows = pool.map(_run_task, tasks)
    detail = pd.DataFrame(rows)
    detail.to_csv(ROOT / "replicates.csv", index=False)

    group = ["resolution_years", "proxy_count", "noise_fraction"]
    summary = detail.groupby(group, as_index=False).agg(
        repeats=("repeat", "count"),
        nonzero_weight_fraction=("selected_weight", lambda x: float(np.mean(x > 0))),
        median_selected_weight=("selected_weight", "median"),
        median_core_r=("core_r", "median"),
        median_adjusted_r=("adjusted_r", "median"),
        median_delta_ce=("delta_ce", "median"),
        median_delta_lowpass_10y_r=("delta_lowpass_10y_r", "median"),
        median_delta_lowpass_20y_r=("delta_lowpass_20y_r", "median"),
        median_delta_phase=("delta_spectral_10_30y_phase_similarity", "median"),
        median_core_phase=("core_spectral_10_30y_phase_similarity", "median"),
        median_adjusted_phase=("adjusted_spectral_10_30y_phase_similarity", "median"),
        median_core_amplitude_ratio=(
            "core_spectral_10_30y_amplitude_ratio", "median"
        ),
        median_adjusted_amplitude_ratio=(
            "adjusted_spectral_10_30y_amplitude_ratio", "median"
        ),
        median_core_sd_ratio=("core_sd_ratio", "median"),
        median_adjusted_sd_ratio=("adjusted_sd_ratio", "median"),
        phase_improvement_fraction=(
            "delta_spectral_10_30y_phase_similarity",
            lambda x: float(np.mean(x > 0)),
        ),
        median_peak_error_reduction=("peak_error_reduction", "median"),
        core_correct_peak_fraction=(
            "core_spectral_10_30y_peak_period_error", lambda x: float(np.mean(x <= 1.0))
        ),
        adjusted_correct_peak_fraction=(
            "adjusted_spectral_10_30y_peak_period_error",
            lambda x: float(np.mean(x <= 1.0)),
        ),
        peak_improvement_fraction=("peak_error_reduction", lambda x: float(np.mean(x > 0))),
        median_amplitude_error_reduction=("amplitude_error_reduction", "median"),
        amplitude_improvement_fraction=(
            "amplitude_error_reduction", lambda x: float(np.mean(x > 0))
        ),
    )
    summary.to_csv(ROOT / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""PDO ablation for the frozen automatic/dynamic/weighted NPCR method."""

from dataclasses import replace

import pandas as pd

from npcrflow import AmplitudeCalibrationConfig, ProxyWeightConfig, run_pipeline
from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config


VARIANTS = (
    ("auto_unweighted", "auto", False),
    ("dynamic_weighted", "dynamic_variance", True),
    ("auto_weighted", "auto", True),
)


def main() -> None:
    root = PROJECT / "results" / "pdo_v040_frozen_method"
    summaries: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for label, amplitude_method, weighted in VARIANTS:
        config = build_config(
            output_directory=root / label,
            amplitude_method="none",
            multiresolution_enabled=False,
        )
        amplitude = AmplitudeCalibrationConfig(
            method=amplitude_method,
            variance_reference="observation",
            minimum_overlap=20,
            slope_bounds=(0.25, 4.0),
            auto_candidates=(
                "none",
                "variance_observation",
                "variance_max_proxy_nest",
                "dynamic_variance",
            ),
            auto_sd_ratio_weight=0.10,
            dynamic_max_bins=4,
            dynamic_minimum_bin_years=15,
            dynamic_shrinkage_years=15.0,
        )
        proxy_weights = ProxyWeightConfig(
            enabled=weighted,
            redundancy_radius_km=250.0,
            redundancy_correlation_threshold=0.85,
            redundancy_min_overlap=20,
            same_archive_only=True,
            same_proxy_only=True,
            minimum_weight=0.10,
        )
        config = replace(
            config,
            reconstruction=replace(
                config.reconstruction,
                interpolation="none",
                retain_longest_annual_segment=False,
                n_bootstrap=0,
                amplitude=amplitude,
                proxy_weights=proxy_weights,
            ),
            output=replace(
                config.output,
                save_proxy_map=False,
                save_observation_plot=False,
            ),
        )
        result = run_pipeline(PROXIES, TARGET, config)
        summary = result.reconstruction.validation_summary.copy()
        summary.insert(0, "variant", label)
        summary["final_amplitude_method"] = result.reconstruction.model.amplitude_calibrator.method
        summary["final_amplitude_reference"] = result.reconstruction.model.amplitude_calibrator.reference
        summary["weighted_proxy_count"] = result.reconstruction.model.proxy_weights.sum()
        summary["downweighted_proxy_count"] = int(
            (result.reconstruction.model.proxy_weights < 1.0).sum()
        )
        summaries.append(summary)
        fold = result.reconstruction.validation_folds.copy()
        fold.insert(0, "variant", label)
        folds.append(fold)
        print(summary.to_string(index=False), flush=True)

    root.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(
        root / "comparison_summary.csv", index=False
    )
    pd.concat(folds, ignore_index=True).to_csv(
        root / "comparison_folds.csv", index=False
    )


if __name__ == "__main__":
    main()

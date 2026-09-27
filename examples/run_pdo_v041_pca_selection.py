"""PDO audit of legacy and validation-aware PCA dimension rules."""

from dataclasses import replace

import numpy as np
import pandas as pd

from npcrflow import PCAConfig, run_pipeline
from npcrflow.diagnostics import observation_fit_table
from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config


VARIANTS = (
    ("kaiser_legacy", "kaiser", 64),
    ("kaiser_cv", "kaiser_cv", 20),
    ("blocked_cv", "blocked_cv", 5),
    ("variance_90", "variance", 20),
)


def main() -> None:
    root = PROJECT / "results" / "pdo_v041_pca_selection"
    summaries: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for label, selection, maximum in VARIANTS:
        config = build_config(
            output_directory=root / label,
            amplitude_method="auto",
            multiresolution_enabled=False,
        )
        config = replace(
            config,
            pca=PCAConfig(
                method="pairwise",
                selection=selection,
                kaiser_threshold=1.0,
                variance_fraction=0.90,
                max_components=maximum,
                min_pairwise_overlap=10,
                min_proxies_per_year=2,
                score_ridge=0.10,
            ),
            reconstruction=replace(config.reconstruction, n_bootstrap=0),
            output=replace(
                config.output,
                save_proxy_map=False,
                save_observation_plot=False,
            ),
        )
        result = run_pipeline(PROXIES, TARGET, config)
        model = result.reconstruction.model
        apparent = observation_fit_table(
            result.target,
            result.reconstruction.reconstruction,
            config.reconstruction,
        ).iloc[0]
        summary = result.reconstruction.validation_summary.copy()
        summary.insert(0, "variant", label)
        summary["pca_selection"] = selection
        summary["final_n_components"] = model.n_components
        summary["full_kaiser_count"] = int(
            np.sum(model.eigenvalues > config.pca.kaiser_threshold)
        )
        summary["final_regression"] = model.regression_name
        summary["final_alpha"] = model.alpha
        summary["final_amplitude_method"] = model.amplitude_calibrator.method
        summary["apparent_r"] = float(apparent["r"])
        summary["apparent_sd_ratio"] = float(apparent["sd_ratio"])
        summary["apparent_re"] = float(apparent["re"])
        summary["apparent_ce"] = float(apparent["ce"])
        summaries.append(summary)
        fold = result.reconstruction.validation_folds.copy()
        fold.insert(0, "variant", label)
        folds.append(fold)
        print(
            summary[
                [
                    "variant", "pca_selection", "final_n_components",
                    "full_kaiser_count", "final_amplitude_method",
                    "apparent_r", "apparent_sd_ratio", "median_r",
                    "median_sd_ratio", "median_re", "median_ce",
                ]
            ].to_string(index=False),
            flush=True,
        )

    root.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(
        root / "comparison_summary.csv", index=False
    )
    pd.concat(
        [frame.dropna(axis=1, how="all") for frame in folds],
        ignore_index=True,
    ).to_csv(
        root / "comparison_folds.csv", index=False
    )


if __name__ == "__main__":
    main()

"""Compare fold-safe amplitude references on the unfiltered d18O PDO case."""

from dataclasses import replace

import pandas as pd

from npcrflow import AmplitudeCalibrationConfig, run_pipeline
from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config


VARIANTS = (
    ("none", "none", "observation"),
    ("variance_observation", "variance", "observation"),
    ("variance_max_proxy_nest", "variance", "max_proxy_nest"),
)


def main() -> None:
    result_root = PROJECT / "results" / "pdo_v035_amplitude_references"
    summaries: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for label, method, reference in VARIANTS:
        config = build_config(
            output_directory=result_root / label,
            amplitude_method=method,
            amplitude_variance_reference=reference,
            multiresolution_enabled=False,
        )
        config = replace(
            config,
            reconstruction=replace(config.reconstruction, n_bootstrap=0),
            output=replace(
                config.output,
                save_proxy_map=False,
                save_observation_plot=False,
            ),
        )
        result = run_pipeline(PROXIES, TARGET, config)
        summary = result.reconstruction.validation_summary.copy()
        summary.insert(0, "variant", label)
        summaries.append(summary)
        fold = result.reconstruction.validation_folds.copy()
        fold.insert(0, "variant", label)
        folds.append(fold)
        print(summary.to_string(index=False), flush=True)

    result_root.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(
        result_root / "comparison_summary.csv", index=False
    )
    pd.concat(folds, ignore_index=True).to_csv(
        result_root / "comparison_folds.csv", index=False
    )


if __name__ == "__main__":
    main()

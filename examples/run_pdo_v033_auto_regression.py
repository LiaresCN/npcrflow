"""Leakage-safe automatic-regression PDO test at two proxy resolutions.

The PDO target is unfiltered. Proxy records retain their native observation
years; no proxy interpolation is performed. Every reported holdout chooses
the regression family using only contiguous inner folds of its calibration
period.
"""

from dataclasses import replace

import pandas as pd

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import SensitivityConfig, run_pipeline
from npcrflow.diagnostics import observation_fit_table


ROOT = PROJECT / "results" / "pdo_v033_auto_regression"
EXPERIMENTS = (
    ("ridge", 1.5),
    ("auto", 1.5),
    ("ridge", 5.0),
    ("auto", 5.0),
)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    split_rows: list[pd.DataFrame] = []
    apparent_rows: list[pd.DataFrame] = []
    outer_rows: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []
    for requested_regression, maximum_resolution in EXPERIMENTS:
        label = f"{requested_regression}_to_{str(maximum_resolution).replace('.', 'p')}y"
        config = build_config(
            output_directory=ROOT / label,
            amplitude_method="none",
            multiresolution_enabled=False,
        )
        config = replace(
            config,
            proxy_filter=replace(
                config.proxy_filter,
                maximum_resolution_years=maximum_resolution,
            ),
            reconstruction=replace(
                config.reconstruction,
                regression=requested_regression,
                regression_candidates=("ridge", "pls", "elasticnet"),
                n_bootstrap=0,
            ),
            sensitivity=SensitivityConfig(
                split_periods=(
                    ((1900, 1949), (1950, 2000)),
                    ((1950, 2000), (1900, 1949)),
                ),
            ),
            output=replace(
                config.output,
                save_reconstruction=False,
                save_screening=False,
                save_metrics=False,
                save_proxy_map=False,
                save_observation_plot=False,
                save_nests=False,
            ),
        )
        result = run_pipeline(PROXIES, TARGET, config)

        split = result.sensitivities["split_period"].copy()
        split.insert(0, "maximum_resolution_years", maximum_resolution)
        split.insert(0, "requested_regression", requested_regression)
        split.insert(0, "experiment", label)
        split_rows.append(split)

        apparent = observation_fit_table(
            result.target,
            result.reconstruction.reconstruction,
            config.reconstruction,
        )
        apparent.insert(0, "selected_regression", result.reconstruction.model.regression_name)
        apparent.insert(0, "maximum_resolution_years", maximum_resolution)
        apparent.insert(0, "requested_regression", requested_regression)
        apparent.insert(0, "experiment", label)
        apparent_rows.append(apparent)

        outer = result.reconstruction.validation_folds.copy()
        outer.insert(0, "maximum_resolution_years", maximum_resolution)
        outer.insert(0, "requested_regression", requested_regression)
        outer.insert(0, "experiment", label)
        outer_rows.append(outer)

        selection = result.reconstruction.model_selection.copy()
        if not selection.empty:
            selection.insert(0, "maximum_resolution_years", maximum_resolution)
            selection.insert(0, "requested_regression", requested_regression)
            selection.insert(0, "experiment", label)
            selection_rows.append(selection)
        print(split.to_string(index=False), flush=True)

    pd.concat(split_rows, ignore_index=True).to_csv(
        ROOT / "bidirectional_split.csv", index=False
    )
    pd.concat(apparent_rows, ignore_index=True).to_csv(
        ROOT / "apparent_fit.csv", index=False
    )
    pd.concat(outer_rows, ignore_index=True).to_csv(
        ROOT / "outer_thirds.csv", index=False
    )
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_csv(
            ROOT / "full_calibration_model_selection.csv", index=False
        )


if __name__ == "__main__":
    main()

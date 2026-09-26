"""Bidirectional spectral comparison of linear statistical estimators."""

from dataclasses import replace

import pandas as pd

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import SensitivityConfig, run_pipeline
from npcrflow.diagnostics import observation_fit_table


ROOT = PROJECT / "results" / "pdo_v032_spectral_regression"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    split_rows: list[pd.DataFrame] = []
    apparent_rows: list[pd.DataFrame] = []
    for regression in ("ridge", "pls", "elasticnet"):
        config = build_config(
            output_directory=ROOT / regression,
            amplitude_method="none",
            multiresolution_enabled=False,
        )
        config = replace(
            config,
            proxy_filter=replace(config.proxy_filter, maximum_resolution_years=5.0),
            reconstruction=replace(
                config.reconstruction,
                regression=regression,
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
                save_proxy_map=False,
                save_observation_plot=False,
            ),
        )
        result = run_pipeline(PROXIES, TARGET, config)
        split = result.sensitivities["split_period"].copy()
        split.insert(0, "regression", regression)
        split_rows.append(split)
        apparent = observation_fit_table(
            result.target,
            result.reconstruction.reconstruction,
            config.reconstruction,
        )
        apparent.insert(0, "regression", regression)
        apparent_rows.append(apparent)
        print(split.to_string(index=False), flush=True)
    pd.concat(split_rows, ignore_index=True).to_csv(
        ROOT / "spectral_regression_split.csv", index=False
    )
    pd.concat(apparent_rows, ignore_index=True).to_csv(
        ROOT / "spectral_regression_apparent.csv", index=False
    )


if __name__ == "__main__":
    main()

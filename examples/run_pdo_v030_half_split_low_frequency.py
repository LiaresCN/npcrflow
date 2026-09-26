"""Bidirectional half-period validation for decadal and bidecadal PDO skill."""

from dataclasses import replace

import pandas as pd

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import SensitivityConfig, run_pipeline
from npcrflow.diagnostics import observation_fit_table


ROOT = PROJECT / "results" / "pdo_v030_half_split_low_frequency"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    apparent_rows: list[pd.DataFrame] = []
    for name, maximum_resolution in (
        ("native_to_1p5y", 1.5),
        ("native_to_5y", 5.0),
    ):
        config = build_config(
            output_directory=ROOT / name,
            amplitude_method="none",
            multiresolution_enabled=False,
        )
        config = replace(
            config,
            proxy_filter=replace(
                config.proxy_filter,
                maximum_resolution_years=maximum_resolution,
            ),
            reconstruction=replace(config.reconstruction, n_bootstrap=0),
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
        table = result.sensitivities["split_period"].copy()
        table.insert(0, "experiment", name)
        rows.append(table)
        apparent = observation_fit_table(
            result.target,
            result.reconstruction.reconstruction,
            config.reconstruction,
        )
        apparent.insert(0, "experiment", name)
        apparent_rows.append(apparent)
        print(table.to_string(index=False), flush=True)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(ROOT / "half_split_comparison.csv", index=False)
    pd.concat(apparent_rows, ignore_index=True).to_csv(
        ROOT / "apparent_spectral_comparison.csv", index=False
    )


if __name__ == "__main__":
    main()

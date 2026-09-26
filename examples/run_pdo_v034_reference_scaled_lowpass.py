"""Real PDO audit of reference-scaled native-window low-frequency assimilation."""

from dataclasses import replace

import pandas as pd

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import SensitivityConfig, run_pipeline
from npcrflow.diagnostics import observation_fit_table


ROOT = PROJECT / "results" / "pdo_v034_reference_scaled_lowpass"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    config = build_config(
        output_directory=ROOT,
        amplitude_method="none",
        multiresolution_enabled=True,
    )
    config = replace(
        config,
        reconstruction=replace(config.reconstruction, n_bootstrap=0),
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
    result.reconstruction.validation_folds.to_csv(ROOT / "outer_thirds.csv", index=False)
    result.reconstruction.multiresolution_selection.to_csv(
        ROOT / "full_calibration_lowpass_selection.csv", index=False
    )
    result.reconstruction.low_frequency_constraints.to_csv(
        ROOT / "full_calibration_constraints.csv", index=False
    )
    result.sensitivities["split_period"].to_csv(
        ROOT / "bidirectional_split.csv", index=False
    )
    apparent = observation_fit_table(
        result.target,
        result.reconstruction.reconstruction,
        config.reconstruction,
    )
    apparent.to_csv(ROOT / "apparent_fit.csv", index=False)
    print(result.reconstruction.validation_folds.to_string(index=False), flush=True)
    print(result.reconstruction.multiresolution_selection.to_string(index=False), flush=True)
    print(result.sensitivities["split_period"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

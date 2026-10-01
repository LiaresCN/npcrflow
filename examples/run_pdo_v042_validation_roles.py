"""Verify v0.4.2 main evidence and supplementary segment sensitivities."""

from dataclasses import replace

from npcrflow import run_pipeline
from npcrflow.diagnostics import primary_reconstruction_summary
from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config


def main() -> None:
    output = PROJECT / "results" / "pdo_v042_validation_roles"
    config = build_config(
        output_directory=output,
        amplitude_method="auto",
        multiresolution_enabled=False,
    )
    config = replace(
        config,
        reconstruction=replace(
            config.reconstruction,
            n_bootstrap=0,
            external_validation_fraction=1.0 / 3.0,
            full_network_outer_validation=True,
            rescreen_outer_folds=True,
        ),
        output=replace(
            config.output,
            save_proxy_map=False,
            save_observation_plot=False,
        ),
    )
    result = run_pipeline(PROXIES, TARGET, config)
    primary = primary_reconstruction_summary(
        result.target,
        result.reconstruction.reconstruction,
        result.reconstruction.model_selection,
        config.reconstruction,
    )
    print("PRIMARY", flush=True)
    print(primary.to_string(index=False), flush=True)
    print("EXTERNAL_SENSITIVITIES", flush=True)
    print(
        result.reconstruction.validation_summary[
            [
                "validation_mode", "validation_role", "assessment_metric",
                "fold_count", "median_r", "minimum_r", "maximum_r",
                "early_holdout_r", "late_holdout_r",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

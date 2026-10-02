"""Annual-only PDO ablation for the v0.6 explicit-NEST benchmark."""

from dataclasses import replace

from npcrflow import run_pipeline
from npcrflow.diagnostics import observation_fit_table
from run_pdo_from_raw_dod2k import PROXIES, TARGET, build_config


def main() -> None:
    config = build_config(
        amplitude_method="auto",
        multiresolution_enabled=False,
    )
    config = replace(
        config,
        proxy_filter=replace(
            config.proxy_filter,
            maximum_resolution_years=1.5,
        ),
        reconstruction=replace(
            config.reconstruction,
            method="explicit_nest",
            n_bootstrap=0,
            full_network_outer_validation=False,
            rescreen_outer_folds=False,
        ),
        output=replace(
            config.output,
            directory=config.output.directory.parent / "pdo_v060_explicit_nests_annual_only",
            save_proxy_map=False,
            save_observation_plot=False,
        ),
    )
    result = run_pipeline(PROXIES, TARGET, config)
    print(f"selected_proxies={int(result.screening['selected'].sum())}")
    print(
        observation_fit_table(
            result.target,
            result.reconstruction.reconstruction,
            config.reconstruction,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()

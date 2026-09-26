"""Training-only selection of the PDO native-window low-frequency influence."""

from dataclasses import replace

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import run_pipeline


def main() -> None:
    config = build_config(
        output_directory=PROJECT / "results" / "pdo_v031_auto_tuned_lowpass",
        amplitude_method="none",
        multiresolution_enabled=True,
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
    print(result.reconstruction.validation_summary.to_string(index=False), flush=True)
    print(result.reconstruction.validation_folds.to_string(index=False), flush=True)
    print(result.reconstruction.multiresolution_selection.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

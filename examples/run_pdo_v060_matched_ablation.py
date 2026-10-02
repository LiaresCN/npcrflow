"""Keep the mixed proxy coverage NEST grid but set resolution influence to zero."""

from dataclasses import replace
import logging

from npcrflow import run_pipeline
from run_pdo_from_raw_dod2k import PROXIES, TARGET, build_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = build_config(amplitude_method="auto", multiresolution_enabled=False)
    config = replace(
        config,
        reconstruction=replace(
            config.reconstruction,
            n_bootstrap=0, full_network_outer_validation=False, rescreen_outer_folds=False,
            nest=replace(config.reconstruction.nest, subnest_constraint_weight=0.0),
        ),
        output=replace(
            config.output,
            directory=config.output.directory.parent / "pdo_v060_matched_coverage_zero_resolution",
            save_proxy_map=False, save_observation_plot=False,
        ),
    )
    result = run_pipeline(PROXIES, TARGET, config)
    print(result.output_paths, flush=True)


if __name__ == "__main__":
    main()

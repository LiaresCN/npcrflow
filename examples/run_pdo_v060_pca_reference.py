"""Declared PCA references on the identical unfiltered d18O PDO network."""

import argparse
import logging
from dataclasses import replace

from npcrflow import run_pipeline
from run_pdo_from_raw_dod2k import PROXIES, TARGET, build_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", choices=("kaiser_cv", "kaiser", "variance"))
    args = parser.parse_args()
    config = build_config(amplitude_method="auto", multiresolution_enabled=False)
    config = replace(
        config,
        pca=replace(config.pca, selection=args.selection),
        reconstruction=replace(
            config.reconstruction, method="explicit_nest", n_bootstrap=0,
            full_network_outer_validation=False, rescreen_outer_folds=False,
        ),
        output=replace(
            config.output,
            directory=config.output.directory.parent / f"pdo_v060_pca_{args.selection}",
            save_proxy_map=False, save_observation_plot=False,
        ),
    )
    # Keep the same max_components=5 safety cap across all PCA references.
    # The Kaiser rule is exact within this declared cap, not an uncapped
    # reproduction of every dimension retained by the old notebooks.
    result = run_pipeline(PROXIES, TARGET, config)
    print(f"pca_reference={args.selection}", flush=True)
    print(f"selected_proxies={int(result.screening['selected'].sum())}", flush=True)
    print(result.output_paths, flush=True)


if __name__ == "__main__":
    main()

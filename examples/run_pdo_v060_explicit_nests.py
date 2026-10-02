"""PDO verification of explicit in-memory and multiresolution NESTs."""

from dataclasses import replace
import logging

from npcrflow import run_pipeline
from npcrflow.diagnostics import observation_fit_table
from run_pdo_from_raw_dod2k import build_config
from run_pdo_from_raw_dod2k import PROXIES, TARGET


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = build_config(
        output_directory=None,
        amplitude_method="auto",
        # Isolate the primary 2–10-year resolution-sub-NEST method. The
        # separate >10-year variational fallback remains an optional run.
        multiresolution_enabled=False,
    )
    config = replace(
        config,
        reconstruction=replace(
            config.reconstruction,
            method="explicit_nest",
            n_bootstrap=0,
            full_network_outer_validation=False,
            rescreen_outer_folds=False,
        ),
        output=replace(
            config.output,
            directory=config.output.directory.parent / "pdo_v060_explicit_nests_verified",
            save_proxy_map=False,
            save_observation_plot=False,
        ),
    )
    result = run_pipeline(PROXIES, TARGET, config)
    coverage = result.reconstruction.nest_summary
    coverage = coverage.loc[coverage["row_type"] == "coverage_nest"]
    resolution = result.reconstruction.nest_summary
    resolution = resolution.loc[resolution["row_type"] == "resolution_subnest"]
    print(f"selected_proxies={int(result.screening['selected'].sum())}")
    print(f"coverage_nests={len(coverage)} accepted={int(coverage['accepted'].sum())}")
    print(
        f"resolution_subnests={len(resolution)} "
        f"accepted={int(resolution['accepted'].sum())}"
    )
    print(
        observation_fit_table(
            result.target,
            result.reconstruction.reconstruction,
            config.reconstruction,
        )
        .to_string(index=False)
    )
    print(result.output_paths)


if __name__ == "__main__":
    main()

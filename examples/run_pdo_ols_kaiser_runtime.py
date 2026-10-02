"""Time the user-selected OLS/Kaiser PDO baseline, without optional tests."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter

from npcrflow import run_pipeline
from npcrflow.diagnostics import observation_fit_table
from run_pdo_from_raw_dod2k import PROXIES, TARGET, build_config


def build_runtime_config(output_directory: Path):
    base = build_config(amplitude_method="auto", multiresolution_enabled=False)
    return replace(
        base,
        pca=replace(base.pca, selection="kaiser", max_components=8),
        reconstruction=replace(
            base.reconstruction,
            regression="ols", regression_candidates=("ols",),
            ridge_alphas=(0.0,), auto_tune=True,
            min_ce=0.1, min_re=0.1, internal_ce_re_comparison="gt",
            n_bootstrap=0, full_network_outer_validation=False,
            rescreen_outer_folds=False,
            amplitude=replace(
                base.reconstruction.amplitude,
                auto_candidates=("variance_observation", "variance_max_proxy_nest"),
            ),
        ),
        output=replace(
            base.output, directory=output_directory,
            save_proxy_map=False, save_observation_plot=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path,
        default=Path("results/pdo_ols_kaiser8_strict010_two_variances_timing"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_directory}")
    config = build_runtime_config(args.output_directory)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    launch = {
        "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(), "git_revision": revision,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "config": asdict(config),
        "timing_scope": "full raw-input main pipeline, including compact output saves",
        "amplitude_scope": "two internal per-NEST reference candidates, not two final products",
    }
    runtime_path = args.output_directory / "runtime.json"
    runtime_path.write_text(json.dumps(launch, indent=2, default=str) + "\n")
    print(f"PID={launch['pid']} JOB_ID={launch['slurm_job_id']} REVISION={revision}", flush=True)
    started = perf_counter()
    try:
        result = run_pipeline(PROXIES, TARGET, config)
    except Exception as error:
        launch.update(status="failed", elapsed_seconds=perf_counter() - started,
                      error_type=type(error).__name__, error=str(error))
        runtime_path.write_text(json.dumps(launch, indent=2, default=str) + "\n")
        raise
    elapsed = perf_counter() - started
    nests = result.reconstruction.nest_summary
    coverage = nests.loc[nests.row_type.eq("coverage_nest")]
    resolution = nests.loc[nests.row_type.eq("resolution_subnest")]
    fit = observation_fit_table(result.target, result.reconstruction.reconstruction,
                                config.reconstruction)
    launch.update(
        status="completed", elapsed_seconds=elapsed,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        selected_proxies=int(result.screening.selected.sum()),
        coverage_nests=len(coverage), accepted_coverage_nests=int(coverage.accepted.sum()),
        resolution_subnests=len(resolution),
        accepted_resolution_subnests=int(resolution.accepted.sum()),
        observation_fit=fit.to_dict(orient="records"),
    )
    runtime_path.write_text(json.dumps(launch, indent=2, default=str) + "\n")
    print(f"PIPELINE_ELAPSED_SECONDS={elapsed:.6f}", flush=True)
    print(fit.to_string(index=False), flush=True)
    print(f"RUNTIME_AUDIT={runtime_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()

"""Full-network checks of explicit training-only amplitude calibration."""

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import run_pipeline


def main() -> None:
    for method in ("ols", "variance"):
        config = build_config(
            output_directory=PROJECT
            / "results"
            / f"pdo_v021_amplitude_{method}_bootstrap_corrected",
            amplitude_method=method,
            multiresolution_enabled=False,
        )
        result = run_pipeline(PROXIES, TARGET, config)
        summary = result.reconstruction.validation_summary.copy()
        summary.insert(0, "amplitude_method", method)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

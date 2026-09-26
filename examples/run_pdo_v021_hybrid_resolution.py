"""Complete-network PDO check with native sub-decadal PCR routing."""

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import run_pipeline


def main() -> None:
    config = build_config(
        output_directory=PROJECT
        / "results/pdo_v021_hybrid_native_regression_bootstrap_corrected",
        amplitude_method="ols",
        multiresolution_enabled=True,
    )
    result = run_pipeline(PROXIES, TARGET, config)
    print(result.reconstruction.validation_summary.to_string(index=False), flush=True)
    roles = result.screening.loc[result.screening["selected"], "reconstruction_role"]
    print(roles.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()

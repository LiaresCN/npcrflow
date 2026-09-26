"""Full-network check of the redesigned native-window low-pass layer."""

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import run_pipeline


def main() -> None:
    config = build_config(
        output_directory=PROJECT / "results/pdo_v020_lowpass_native_windows_capped",
        amplitude_method="ols",
        multiresolution_enabled=True,
    )
    result = run_pipeline(PROXIES, TARGET, config)
    print(result.reconstruction.validation_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Fair no-amplitude A/B check of native PCR versus hybrid resolution routing."""

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import run_pipeline


def main() -> None:
    for name, enabled in (("all_native_pcr", False), ("hybrid_10y", True)):
        config = build_config(
            output_directory=PROJECT
            / "results"
            / f"pdo_v021_{name}_no_amplitude_bootstrap_corrected",
            amplitude_method="none",
            multiresolution_enabled=enabled,
        )
        result = run_pipeline(PROXIES, TARGET, config)
        summary = result.reconstruction.validation_summary.copy()
        summary.insert(0, "experiment", name)
        print(summary.to_string(index=False), flush=True)
        roles = result.screening.loc[result.screening["selected"], "reconstruction_role"]
        print(roles.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()

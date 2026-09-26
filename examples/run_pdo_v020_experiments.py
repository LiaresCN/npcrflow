"""Engineering checks for NPCRFlow 0.2 amplitude and multiresolution options."""

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import run_pipeline


EXPERIMENTS = (
    ("amplitude_ols", "ols", False),
    ("amplitude_variance", "variance", False),
    ("lowpass_ols", "ols", True),
)


def main() -> None:
    for name, amplitude_method, multiresolution_enabled in EXPERIMENTS:
        output = PROJECT / "results" / f"pdo_v020_{name}"
        config = build_config(
            output_directory=output,
            amplitude_method=amplitude_method,
            multiresolution_enabled=multiresolution_enabled,
        )
        result = run_pipeline(PROXIES, TARGET, config)
        summary = result.reconstruction.validation_summary.copy()
        summary.insert(0, "experiment", name)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Validate controlled Wood/Coral interpolation and longest annual output."""

from dataclasses import replace

from npcrflow import run_pipeline
from npcrflow.diagnostics import primary_reconstruction_summary
from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config


def main() -> None:
    output = PROJECT / "results" / "pdo_v050_annual_longest"
    config = build_config(
        output_directory=output,
        amplitude_method="auto",
        multiresolution_enabled=False,
    )
    config = replace(
        config,
        reconstruction=replace(
            config.reconstruction,
            interpolation="archive_linear",
            interpolation_archives=("Wood", "Coral"),
            interpolation_max_gap_years=2,
            interpolation_max_resolution_years=2.0,
            retain_longest_annual_segment=True,
            n_bootstrap=0,
        ),
        output=replace(
            config.output,
            save_proxy_map=False,
            save_observation_plot=False,
        ),
    )
    result = run_pipeline(PROXIES, TARGET, config)
    reconstruction = result.reconstruction.reconstruction
    years = reconstruction["Year"].astype(int)
    primary = primary_reconstruction_summary(
        result.target,
        reconstruction,
        result.reconstruction.model_selection,
        config.reconstruction,
    )
    audit = result.reconstruction.interpolation_audit
    print("PRIMARY", flush=True)
    print(primary.to_string(index=False), flush=True)
    print("ANNUAL_OUTPUT", flush=True)
    print(
        {
            "start": int(years.min()),
            "end": int(years.max()),
            "year_count": len(years),
            "is_contiguous": bool((years.diff().dropna() == 1).all()),
            "finite_median_count": int(reconstruction["median"].notna().sum()),
            "interpolated_proxy_count": int((audit["interpolated_year_count"] > 0).sum()),
            "interpolated_year_value_count": int(audit["interpolated_year_count"].sum()),
        },
        flush=True,
    )
    print("INTERPOLATED_ARCHIVES", flush=True)
    print(
        audit.loc[audit["interpolated_year_count"] > 0]
        .groupby("archive")["interpolated_year_count"]
        .agg(["count", "sum"])
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()

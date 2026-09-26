"""WNPSM case starting from the same raw de-duplicated Dod2k database."""

from pathlib import Path

from npcrflow import (
    OutputConfig,
    PCAConfig,
    PipelineConfig,
    ProxyFilterConfig,
    ReconstructionConfig,
    ScreeningConfig,
    SensitivityConfig,
    TargetConfig,
    run_pipeline,
)


PROJECT = Path("/share/home/lrs/codex/projects/npcrflow")
PROXIES = Path("/share/home/lrs/Date collection/proxy数据库/Dod2k_V20260513_filteredproxy_lrs_use.pkl")
TARGET = Path("/share/home/lrs/WNPSM/返修/03_statistical_results/era5_wnpsm_jjas_1940_2025.csv")


def main() -> None:
    config = PipelineConfig(
        proxy_filter=ProxyFilterConfig(
            latitude=(-20, 50),
            # The source Dod2k longitudes are stored on -180..180.  This is
            # the actual subset produced by the legacy notebook's raw
            # ``60 <= lon <= 350`` comparison.
            longitude=(60, 180),
            archives=("Wood", "Coral"),
            proxies=("d18O", "ring width"),
            climate_variables=("moisture", "temperature", "temperature+moisture"),
            maximum_plausible_year=2100.0,
        ),
        target=TargetConfig(value_column="wnpsm_ref_1971_2000"),
        screening=ScreeningConfig(
            period=(1940, 2000),
            season_mode="fixed",
            months=(7, 8, 9, 10),
            min_overlap=20,
            p_threshold=0.10,
            r_threshold=0.20,
        ),
        pca=PCAConfig(
            method="pairwise",
            selection="blocked_cv",
            max_components=5,
            min_pairwise_overlap=10,
            min_proxies_per_year=2,
            score_ridge=0.10,
        ),
        reconstruction=ReconstructionConfig(
            calibration_period=(1940, 2000),
            interpolation="none",
            regression="ridge",
            validation_block_years=15,
            n_bootstrap=50,
            bootstrap_block_years=5,
            min_ce=0.0,
            min_re=0.0,
            skill_floor=0.0,
            strong_skill_threshold=0.5,
        ),
        sensitivity=SensitivityConfig(
            split_periods=(
                ((1940, 1970), (1971, 2000)),
                ((1971, 2000), (1940, 1970)),
            ),
            random_delete_fraction=0.20,
            random_delete_repeats=10,
        ),
        output=OutputConfig(
            directory=PROJECT / "results/wnpsm_raw_dod2k_final",
            save_proxy_map=True,
            save_nests=False,
        ),
    )
    result = run_pipeline(PROXIES, TARGET, config)
    print(result.reconstruction.validation_summary.to_string(index=False))
    print(f"selected={int(result.screening['selected'].sum())}")
    print(f"outputs={result.output_paths}")


if __name__ == "__main__":
    main()

"""Unfiltered PDO case starting directly from the raw Dod2k database."""

from pathlib import Path

from npcrflow import (
    AmplitudeCalibrationConfig,
    MultiresolutionConfig,
    OutputConfig,
    PCAConfig,
    PipelineConfig,
    ProxyFilterConfig,
    ProxyWeightConfig,
    ReconstructionConfig,
    ScreeningConfig,
    SensitivityConfig,
    TargetConfig,
    run_pipeline,
)


PROJECT = Path("/share/home/lrs/codex/projects/npcrflow")
PROXIES = Path("/share/home/lrs/Date collection/proxy数据库/Dod2k_V20260513_filteredproxy_lrs_use.pkl")
TARGET = Path("/share/home/lrs/jgra_recPDO_code/返修/ersstV5-obs-pdo.xlsx")


def build_config(
    *,
    output_directory: Path | None = None,
    amplitude_method: str = "auto",
    amplitude_variance_reference: str = "observation",
    proxy_weighting_enabled: bool = False,
    multiresolution_enabled: bool = False,
) -> PipelineConfig:
    """Return the complete PDO configuration with all method choices visible."""

    return PipelineConfig(
        proxy_filter=ProxyFilterConfig(
            deduplicate_exact=True,
            deduplicate_near=True,
            latitude=None,
            longitude=None,
            archives=("Wood", "Coral", "GlacierIce", "Speleothem"),
            proxies=("d18O",),
            climate_variables=("moisture", "temperature", "temperature+moisture"),
            maximum_plausible_year=2100.0,
            maximum_resolution_years=None,
        ),
        # Use the observed PDO index without temporal filtering. Set
        # ``lowpass_years`` only for an explicit filtering sensitivity test.
        target=TargetConfig(
            year_column="Year",
            value_column="data",
            months=tuple(range(1, 13)),
            season_year="end",
            minimum_month_fraction=0.75,
            detrend=False,
            lowpass_years=None,
            lowpass_order=4,
        ),
        screening=ScreeningConfig(
            period=(1900, 2000),
            season_mode="auto",
            months=tuple(range(1, 13)),
            season_year="end",
            minimum_month_fraction=0.75,
            min_overlap=20,
            low_resolution_min_overlap=8,
            # Records slower than 1.5 years may use the explicitly declared
            # low-resolution overlap rule; values are still never filled.
            low_resolution_cutoff_years=1.5,
            p_threshold=0.10,
            r_threshold=0.20,
            multiple_testing="holm",
            proxy_multiple_testing="none",
            archive_max_counts=(),
        ),
        pca=PCAConfig(
            method="pairwise",
            selection="blocked_cv",
            n_components=None,
            kaiser_threshold=1.0,
            variance_fraction=0.90,
            max_components=5,
            min_pairwise_overlap=10,
            min_proxies_per_year=2,
            score_ridge=0.10,
        ),
        reconstruction=ReconstructionConfig(
            calibration_period=(1900, 2000),
            reconstruction_period=(0, None),
            interpolation="none",
            detrend_proxies=False,
            regression="ridge",
            ridge_alphas=(0.0, 0.01, 0.1, 1.0, 10.0, 100.0),
            pls_components=2,
            random_forest_trees=500,
            random_forest_min_samples_leaf=5,
            random_forest_max_features=1.0,
            evaluation_lowpass_periods=(10.0, 20.0),
            evaluation_period_bands=((10.0, 30.0),),
            # 1900–2000 contains 101 years. A 34-year block yields three
            # contiguous outer folds, each withholding about one third.
            validation_block_years=34,
            n_bootstrap=50,
            bootstrap_block_years=5,
            minimum_bootstrap_success_fraction=0.80,
            random_seed=20260926,
            rescreen_outer_folds=True,
            min_ce=0.0,
            min_re=0.0,
            skill_floor=0.0,
            strong_skill_threshold=0.5,
            auto_tune=True,
            amplitude=AmplitudeCalibrationConfig(
                method=amplitude_method,
                variance_reference=amplitude_variance_reference,
                minimum_overlap=20,
                slope_bounds=(0.25, 4.0),
                auto_candidates=(
                    "none", "variance_observation",
                    "variance_max_proxy_nest", "dynamic_variance",
                ),
                auto_sd_ratio_weight=0.10,
                dynamic_max_bins=4,
                dynamic_minimum_bin_years=15,
                dynamic_shrinkage_years=15.0,
            ),
            proxy_weights=ProxyWeightConfig(
                enabled=proxy_weighting_enabled,
                redundancy_radius_km=250.0,
                redundancy_correlation_threshold=0.85,
                redundancy_min_overlap=20,
                same_archive_only=True,
                same_proxy_only=True,
                minimum_weight=0.10,
            ),
            # The annual state grid is explicit. When enabled, low-resolution
            # values constrain only the low-pass component over native windows.
            multiresolution=MultiresolutionConfig(
                enabled=multiresolution_enabled,
                state_timestep_years=1,
                # Native records up to and including 10-year resolution enter
                # pairwise PCR at observed years only; no values are filled.
                regression_max_resolution_years=10.0,
                lowpass_period_years=10.0,
                smoothness_multiplier=1.0,
                core_anchor_weight=1.0,
                proxy_constraint_weight=1.0,
                auto_tune=True,
                proxy_constraint_weight_candidates=(
                    0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0
                ),
                lowpass_period_candidates=(10.0, 20.0),
                selection_lowpass_period_years=10.0,
                minimum_tuning_folds=2,
                minimum_calibration_overlap=8,
                maximum_support_multiplier=3.0,
                maximum_increment_ratio=1.0,
                preserve_calibration_mean=True,
            ),
        ),
        sensitivity=SensitivityConfig(
            single_proxy=False,
            leave_one_proxy_out=False,
            # Optional experiments are retained but disabled in the normal
            # full-network run. Add directed periods here when required.
            split_periods=(),
            random_delete_fraction=0.20,
            random_delete_repeats=0,
        ),
        output=OutputConfig(
            directory=output_directory
            or PROJECT / "results/pdo_raw_dod2k_unfiltered_two_thirds_core",
            save_proxy_map=True,
            save_reconstruction=True,
            save_screening=True,
            save_metrics=True,
            save_observation_plot=True,
            save_nests=False,
            figure_format="png",
        ),
    )


def main() -> None:
    config = build_config()
    result = run_pipeline(PROXIES, TARGET, config)
    print(result.reconstruction.validation_summary.to_string(index=False))
    print(f"selected={int(result.screening['selected'].sum())}")
    print(f"outputs={result.output_paths}")


if __name__ == "__main__":
    main()

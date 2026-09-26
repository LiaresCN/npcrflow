"""Typed configuration for the one-call NPCR workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


DEFAULT_SEASONS: tuple[tuple[int, ...], ...] = (
    tuple(range(1, 13)),
    (12, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (9, 10, 11),
    (5, 6, 7, 8, 9),
    (6, 7, 8, 9, 10),
)


@dataclass(frozen=True)
class ProxyFilterConfig:
    """Metadata/domain filters applied before statistical screening."""

    deduplicate_exact: bool = True
    deduplicate_near: bool = True
    latitude: tuple[float, float] | None = None
    longitude: tuple[float, float] | None = None
    archives: tuple[str, ...] = ()
    proxies: tuple[str, ...] = ()
    climate_variables: tuple[str, ...] = ()
    maximum_plausible_year: float | None = 2100.0
    maximum_resolution_years: float | None = None

    def __post_init__(self) -> None:
        if self.latitude is not None and self.latitude[0] > self.latitude[1]:
            raise ValueError("latitude bounds must be increasing")
        if self.maximum_resolution_years is not None and self.maximum_resolution_years <= 0:
            raise ValueError("maximum_resolution_years must be positive or None")


@dataclass(frozen=True)
class TargetConfig:
    """Observation aggregation and optional target definition."""

    year_column: str = "Year"
    value_column: str | None = None
    months: tuple[int, ...] = tuple(range(1, 13))
    season_year: Literal["end", "start", "calendar"] = "end"
    minimum_month_fraction: float = 0.75
    detrend: bool = False
    lowpass_years: float | None = None
    lowpass_order: int = 4

    def __post_init__(self) -> None:
        if self.lowpass_years is not None and self.lowpass_years <= 2:
            raise ValueError("lowpass_years must exceed the 2-year Nyquist period")


@dataclass(frozen=True)
class ScreeningConfig:
    """Proxy/target screening settings.

    ``period=None`` uses the full proxy/target overlap.  In ``auto`` season
    mode, the best candidate season is chosen by absolute correlation and its
    effective-DOF p value is Holm-adjusted for the seasons actually tested.
    Annual or lower-resolution records are evaluated only once.
    """

    period: tuple[int, int] | None = None
    season_mode: Literal["fixed", "auto"] = "fixed"
    months: tuple[int, ...] = tuple(range(1, 13))
    season_candidates: tuple[tuple[int, ...], ...] = DEFAULT_SEASONS
    season_year: Literal["end", "start", "calendar"] = "end"
    minimum_month_fraction: float = 0.75
    min_overlap: int = 20
    low_resolution_min_overlap: int | None = None
    low_resolution_cutoff_years: float = 5.0
    p_threshold: float = 0.10
    r_threshold: float = 0.0
    detrend: bool = False
    multiple_testing: Literal["holm", "none"] = "holm"
    proxy_multiple_testing: Literal["fdr_bh", "holm", "none"] = "fdr_bh"
    archive_max_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.period is not None and self.period[0] > self.period[1]:
            raise ValueError("screening period must be increasing")
        if not 0 < self.p_threshold <= 1:
            raise ValueError("p_threshold must be in (0, 1]")
        if not 0 <= self.r_threshold <= 1:
            raise ValueError("r_threshold must be in [0, 1]")
        if self.min_overlap < 4:
            raise ValueError("min_overlap must be at least 4")
        if self.low_resolution_min_overlap is not None and self.low_resolution_min_overlap < 4:
            raise ValueError("low_resolution_min_overlap must be at least 4")
        if self.low_resolution_cutoff_years <= 0:
            raise ValueError("low_resolution_cutoff_years must be positive")
        if not 0 < self.minimum_month_fraction <= 1:
            raise ValueError("minimum_month_fraction must be in (0, 1]")
        for season in (self.months,) + self.season_candidates:
            if not season or any(month < 1 or month > 12 for month in season):
                raise ValueError(f"invalid month sequence: {season}")
        archive_names = [str(name).strip().lower() for name, _ in self.archive_max_counts]
        if len(archive_names) != len(set(archive_names)):
            raise ValueError("archive_max_counts contains duplicate archive names")
        if any(int(limit) < 1 for _, limit in self.archive_max_counts):
            raise ValueError("archive_max_counts limits must be positive")


@dataclass(frozen=True)
class PCAConfig:
    """Principal-component settings for complete or native-resolution data."""

    method: Literal["pairwise", "complete"] = "pairwise"
    selection: Literal["kaiser", "variance", "fixed", "blocked_cv"] = "blocked_cv"
    n_components: int | None = None
    variance_fraction: float = 0.90
    max_components: int = 10
    min_pairwise_overlap: int = 10
    min_proxies_per_year: int = 2
    score_ridge: float = 0.10

    def __post_init__(self) -> None:
        if self.selection == "fixed" and (self.n_components is None or self.n_components < 1):
            raise ValueError("fixed PCA selection requires n_components >= 1")
        if not 0 < self.variance_fraction <= 1:
            raise ValueError("variance_fraction must be in (0, 1]")
        if self.max_components < 1:
            raise ValueError("max_components must be positive")
        if self.score_ridge < 0:
            raise ValueError("score_ridge cannot be negative")


@dataclass(frozen=True)
class AmplitudeCalibrationConfig:
    """Optional, fold-safe calibration of reconstructed amplitude.

    ``ols`` estimates the intercept and slope mapping the raw reconstruction
    to the target. ``variance`` matches training-period means and standard
    deviations to ``variance_reference``. ``max_proxy_nest`` means the densest
    proxy-availability tier having at least ``minimum_overlap`` calibration
    years; it does not create or save legacy NEST files. All parameters are
    fitted only from the current training interval and are therefore part of
    the model rather than a post-validation correction.
    """

    method: Literal["none", "ols", "variance", "dynamic_variance", "auto"] = "none"
    variance_reference: Literal["observation", "max_proxy_nest"] = "observation"
    minimum_overlap: int = 20
    slope_bounds: tuple[float, float] = (0.25, 4.0)
    auto_candidates: tuple[str, ...] = (
        "none",
        "variance_observation",
        "variance_max_proxy_nest",
        "dynamic_variance",
    )
    auto_sd_ratio_weight: float = 0.10
    dynamic_max_bins: int = 4
    dynamic_minimum_bin_years: int = 15
    dynamic_shrinkage_years: float = 15.0

    def __post_init__(self) -> None:
        if self.minimum_overlap < 4:
            raise ValueError("amplitude minimum_overlap must be at least 4")
        lower, upper = self.slope_bounds
        if lower <= 0 or upper < lower:
            raise ValueError("amplitude slope_bounds must be positive and increasing")
        allowed = {
            "none", "ols", "variance_observation",
            "variance_max_proxy_nest", "dynamic_variance",
        }
        if not self.auto_candidates or any(item not in allowed for item in self.auto_candidates):
            raise ValueError("amplitude auto_candidates contains an unsupported choice")
        if self.auto_sd_ratio_weight < 0:
            raise ValueError("auto_sd_ratio_weight cannot be negative")
        if self.dynamic_max_bins < 2:
            raise ValueError("dynamic_max_bins must be at least 2")
        if self.dynamic_minimum_bin_years < 4:
            raise ValueError("dynamic_minimum_bin_years must be at least 4")
        if self.dynamic_shrinkage_years < 0:
            raise ValueError("dynamic_shrinkage_years cannot be negative")


@dataclass(frozen=True)
class ProxyWeightConfig:
    """Optional proxy-error and spatial-redundancy weights for weighted PCA."""

    enabled: bool = False
    error_sd_fields: tuple[str, ...] = (
        "paleoData_uncertainty",
        "paleoData_error",
        "measurement_error",
        "error_sd",
    )
    explicit_error_sd: tuple[tuple[str, float], ...] = ()
    redundancy_radius_km: float = 250.0
    redundancy_correlation_threshold: float = 0.85
    redundancy_min_overlap: int = 20
    same_archive_only: bool = True
    same_proxy_only: bool = True
    minimum_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.redundancy_radius_km < 0:
            raise ValueError("redundancy_radius_km cannot be negative")
        if not 0 <= self.redundancy_correlation_threshold <= 1:
            raise ValueError("redundancy_correlation_threshold must be in [0, 1]")
        if self.redundancy_min_overlap < 4:
            raise ValueError("redundancy_min_overlap must be at least 4")
        if not 0 < self.minimum_weight <= 1:
            raise ValueError("minimum_weight must be in (0, 1]")
        pids = [str(pid) for pid, _ in self.explicit_error_sd]
        if len(pids) != len(set(pids)):
            raise ValueError("explicit_error_sd contains duplicate proxy identifiers")
        if any(float(value) <= 0 for _, value in self.explicit_error_sd):
            raise ValueError("explicit proxy error standard deviations must be positive")


@dataclass(frozen=True)
class MultiresolutionConfig:
    """Native-support low-frequency assimilation into an annual latent state.

    Low-resolution observations are never interpolated. They constrain only
    window means of the low-pass component; the PCR high-frequency component
    is carried through unchanged. The current state grid is explicitly annual.
    An optional increment cap uses the larger low-frequency scale from the core
    and the current calibration-only target, never a held-out observation.
    """

    enabled: bool = False
    state_timestep_years: int = 1
    regression_max_resolution_years: float = 10.0
    lowpass_period_years: float = 10.0
    smoothness_multiplier: float = 1.0
    core_anchor_weight: float = 1.0
    proxy_constraint_weight: float = 1.0
    auto_tune: bool = True
    proxy_constraint_weight_candidates: tuple[float, ...] = (
        0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0
    )
    lowpass_period_candidates: tuple[float, ...] = (10.0, 20.0)
    selection_lowpass_period_years: float = 10.0
    minimum_tuning_folds: int = 2
    minimum_calibration_overlap: int = 8
    maximum_support_multiplier: float | None = 3.0
    maximum_increment_ratio: float | None = 1.0
    preserve_calibration_mean: bool = True

    def __post_init__(self) -> None:
        if self.state_timestep_years != 1:
            raise ValueError("the current latent reconstruction grid must be annual")
        if self.regression_max_resolution_years <= 0:
            raise ValueError("regression_max_resolution_years must be positive")
        if self.lowpass_period_years <= 2:
            raise ValueError("lowpass_period_years must exceed the 2-year Nyquist period")
        if self.smoothness_multiplier < 0:
            raise ValueError("smoothness_multiplier cannot be negative")
        if self.core_anchor_weight < 0 or self.proxy_constraint_weight < 0:
            raise ValueError("multiresolution weights cannot be negative")
        if not self.proxy_constraint_weight_candidates:
            raise ValueError("proxy_constraint_weight_candidates cannot be empty")
        if any(weight < 0 for weight in self.proxy_constraint_weight_candidates):
            raise ValueError("proxy constraint weight candidates cannot be negative")
        if not self.lowpass_period_candidates or any(
            period <= 2 for period in self.lowpass_period_candidates
        ):
            raise ValueError("lowpass period candidates must all exceed 2 years")
        if self.selection_lowpass_period_years <= 2:
            raise ValueError("selection_lowpass_period_years must exceed 2 years")
        if self.minimum_tuning_folds < 1:
            raise ValueError("minimum_tuning_folds must be positive")
        if self.minimum_calibration_overlap < 4:
            raise ValueError("minimum_calibration_overlap must be at least 4")
        if self.maximum_support_multiplier is not None and self.maximum_support_multiplier <= 0:
            raise ValueError("maximum_support_multiplier must be positive or None")
        if self.maximum_increment_ratio is not None and self.maximum_increment_ratio <= 0:
            raise ValueError("maximum_increment_ratio must be positive or None")


@dataclass(frozen=True)
class ReconstructionConfig:
    """Reconstruction and validation settings.

    ``interpolation='none'`` is mandatory and never fills a proxy value.
    ``auto_tune=False`` uses
    the first admissible PC count and first configured regularization value;
    the default performs nested blocked tuning.
    """

    calibration_period: tuple[int, int] | None = None
    reconstruction_period: tuple[int | None, int | None] | None = None
    interpolation: Literal["none"] = "none"
    detrend_proxies: bool = False
    regression: Literal[
        "auto", "ols", "ridge", "pls", "elasticnet", "random_forest"
    ] = "ridge"
    regression_candidates: tuple[str, ...] = ("ridge", "pls", "elasticnet")
    ridge_alphas: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
    pls_components: int = 2
    random_forest_trees: int = 500
    random_forest_min_samples_leaf: int = 5
    random_forest_max_features: float = 1.0
    evaluation_lowpass_periods: tuple[float, ...] = (10.0, 20.0)
    evaluation_period_bands: tuple[tuple[float, float], ...] = ((10.0, 30.0),)
    validation_block_years: int = 20
    n_bootstrap: int = 200
    bootstrap_block_years: int = 5
    minimum_bootstrap_success_fraction: float = 0.80
    random_seed: int = 20260926
    rescreen_outer_folds: bool = True
    min_ce: float | None = 0.0
    min_re: float | None = 0.0
    skill_floor: float = 0.0
    strong_skill_threshold: float = 0.5
    auto_tune: bool = True
    amplitude: AmplitudeCalibrationConfig = field(default_factory=AmplitudeCalibrationConfig)
    proxy_weights: ProxyWeightConfig = field(default_factory=ProxyWeightConfig)
    multiresolution: MultiresolutionConfig = field(default_factory=MultiresolutionConfig)

    def __post_init__(self) -> None:
        if self.interpolation != "none":
            raise ValueError("proxy interpolation is disabled; native missing values must be retained")
        if self.calibration_period is not None and self.calibration_period[0] > self.calibration_period[1]:
            raise ValueError("calibration period must be increasing")
        if self.reconstruction_period is not None:
            start, end = self.reconstruction_period
            if start is not None and end is not None and start > end:
                raise ValueError("reconstruction period must be increasing")
        if self.validation_block_years < 2:
            raise ValueError("validation_block_years must be at least 2")
        if self.n_bootstrap < 0:
            raise ValueError("n_bootstrap cannot be negative")
        if not 0 < self.minimum_bootstrap_success_fraction <= 1:
            raise ValueError("minimum_bootstrap_success_fraction must be in (0, 1]")
        if self.strong_skill_threshold < 0:
            raise ValueError("strong_skill_threshold cannot be negative")
        if self.pls_components < 1:
            raise ValueError("pls_components must be positive")
        allowed_regressions = {"ols", "ridge", "pls", "elasticnet", "random_forest"}
        if not self.regression_candidates:
            raise ValueError("regression_candidates cannot be empty")
        if len(set(self.regression_candidates)) != len(self.regression_candidates):
            raise ValueError("regression_candidates contains duplicates")
        if any(name not in allowed_regressions for name in self.regression_candidates):
            raise ValueError("regression_candidates contains an unsupported model")
        if self.random_forest_trees < 1:
            raise ValueError("random_forest_trees must be positive")
        if self.random_forest_min_samples_leaf < 1:
            raise ValueError("random_forest_min_samples_leaf must be positive")
        if not 0 < self.random_forest_max_features <= 1:
            raise ValueError("random_forest_max_features must be in (0, 1]")
        if any(period <= 2 for period in self.evaluation_lowpass_periods):
            raise ValueError("evaluation low-pass periods must exceed 2 years")
        for minimum, maximum in self.evaluation_period_bands:
            if minimum <= 2 or maximum <= minimum:
                raise ValueError(
                    "evaluation period bands must be increasing and exceed 2 years"
                )


@dataclass(frozen=True)
class SensitivityConfig:
    single_proxy: bool = False
    leave_one_proxy_out: bool = False
    split_periods: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
    random_delete_fraction: float = 0.20
    random_delete_repeats: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.random_delete_fraction < 1:
            raise ValueError("random_delete_fraction must be in [0, 1)")
        if self.random_delete_repeats < 0:
            raise ValueError("random_delete_repeats cannot be negative")


@dataclass(frozen=True)
class OutputConfig:
    directory: Path = Path("results")
    save_reconstruction: bool = True
    save_screening: bool = True
    save_metrics: bool = True
    save_proxy_map: bool = True
    save_observation_plot: bool = True
    save_nests: bool = False
    figure_format: Literal["png", "pdf", "svg"] = "png"


@dataclass(frozen=True)
class PipelineConfig:
    proxy_filter: ProxyFilterConfig = field(default_factory=ProxyFilterConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

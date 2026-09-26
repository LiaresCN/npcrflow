"""Training-only amplitude calibration for reconstruction models."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import AmplitudeCalibrationConfig


@dataclass(frozen=True)
class AmplitudeCalibrator:
    """Affine map fitted from a model's training-period predictions."""

    method: str = "none"
    intercept: float = 0.0
    slope: float = 1.0
    overlap: int = 0
    unclipped_slope: float = 1.0
    clipped: bool = False
    reference: str = "none"
    reference_overlap: int = 0
    reference_min_proxy_count: int | None = None
    source_center: float = 0.0
    target_center: float = 0.0
    availability_knots: tuple[float, ...] = ()
    slope_knots: tuple[float, ...] = ()

    def apply(
        self,
        values: pd.Series,
        effective_availability: pd.Series | None = None,
    ) -> pd.Series:
        if not self.availability_knots:
            result = self.intercept + self.slope * values.astype(float)
            return result.rename(values.name)
        if effective_availability is None:
            raise ValueError("dynamic variance calibration requires effective availability")
        if effective_availability.index.equals(values.index):
            availability = effective_availability.to_numpy(float)
        else:
            availability = effective_availability.reindex(values.index).to_numpy(float)
        local_slope = np.interp(
            availability,
            np.asarray(self.availability_knots, dtype=float),
            np.asarray(self.slope_knots, dtype=float),
        )
        local_slope[~np.isfinite(availability)] = self.slope
        result = self.target_center + local_slope * (
            values.to_numpy(float) - self.source_center
        )
        result = pd.Series(result, index=values.index, name=values.name)
        return result.rename(values.name)


def fit_amplitude_calibrator(
    observed: pd.Series,
    predicted: pd.Series,
    config: AmplitudeCalibrationConfig,
    availability: pd.Series | None = None,
    effective_availability: pd.Series | None = None,
) -> AmplitudeCalibrator:
    """Fit an affine amplitude map without using any withheld observations."""

    if observed.index.equals(predicted.index):
        # Moving-block bootstrap samples legitimately repeat year labels. Their
        # two series are already positionally aligned, while pandas concat
        # refuses to reindex duplicate labels.
        joined = pd.DataFrame(
            {
                "observed": observed.to_numpy(float),
                "predicted": predicted.to_numpy(float),
            }
        )
        if availability is not None:
            if availability.index.equals(predicted.index):
                joined["availability"] = availability.to_numpy(float)
            else:
                joined["availability"] = availability.reindex(predicted.index).to_numpy(float)
        if effective_availability is not None:
            if effective_availability.index.equals(predicted.index):
                joined["effective_availability"] = effective_availability.to_numpy(float)
            else:
                joined["effective_availability"] = effective_availability.reindex(predicted.index).to_numpy(float)
        joined = joined.dropna(subset=["observed", "predicted"])
    else:
        joined = pd.concat(
            [observed.rename("observed"), predicted.rename("predicted")],
            axis=1,
            join="inner",
        ).dropna(subset=["observed", "predicted"])
        if availability is not None:
            joined["availability"] = availability.reindex(joined.index).to_numpy(float)
        if effective_availability is not None:
            joined["effective_availability"] = effective_availability.reindex(joined.index).to_numpy(float)
    if config.method == "none" or len(joined) < config.minimum_overlap:
        return AmplitudeCalibrator(method="none", overlap=len(joined))
    x = joined["predicted"].to_numpy(float)
    y = joined["observed"].to_numpy(float)
    variance = float(np.var(x, ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        return AmplitudeCalibrator(method="none", overlap=len(joined))
    if config.method == "auto":
        raise ValueError("automatic amplitude choice must be resolved by model selection")
    if config.method == "ols":
        slope = float(np.cov(x, y, ddof=1)[0, 1] / variance)
    elif config.method == "variance":
        reference_overlap = len(joined)
        reference_min_proxy_count = None
        if config.variance_reference == "observation":
            reference_values = y
        elif config.variance_reference == "max_proxy_nest":
            if "availability" not in joined:
                raise ValueError(
                    "max_proxy_nest variance calibration requires proxy availability"
                )
            usable = joined.dropna(subset=["availability"])
            reference_values = np.empty(0, dtype=float)
            for threshold in sorted(usable["availability"].unique(), reverse=True):
                candidate = usable.loc[
                    usable["availability"] >= threshold, "predicted"
                ].to_numpy(float)
                if len(candidate) >= config.minimum_overlap:
                    reference_values = candidate
                    reference_min_proxy_count = int(threshold)
                    break
            if len(reference_values) < config.minimum_overlap:
                return AmplitudeCalibrator(method="none", overlap=len(joined))
            reference_overlap = len(reference_values)
        else:  # guarded by the typed configuration
            raise ValueError(
                f"unknown variance reference: {config.variance_reference}"
            )
        target_sd = float(np.std(reference_values, ddof=1))
        predicted_sd = float(np.std(x, ddof=1))
        if not np.isfinite(target_sd) or target_sd <= 0:
            return AmplitudeCalibrator(method="none", overlap=len(joined))
        slope = target_sd / predicted_sd
    elif config.method == "dynamic_variance":
        availability_name = (
            "effective_availability"
            if "effective_availability" in joined
            else "availability"
        )
        if availability_name not in joined:
            raise ValueError("dynamic variance calibration requires proxy availability")
        usable = joined.dropna(subset=[availability_name]).copy()
        maximum_bins = min(
            config.dynamic_max_bins,
            len(usable) // config.dynamic_minimum_bin_years,
        )
        if maximum_bins < 2:
            fallback = replace(
                config,
                method="variance",
                variance_reference="observation",
            )
            return fit_amplitude_calibrator(
                observed,
                predicted,
                fallback,
                availability=availability,
                effective_availability=effective_availability,
            )
        usable["bin"] = pd.qcut(
            usable[availability_name],
            q=maximum_bins,
            duplicates="drop",
        )
        global_slope = float(np.std(y, ddof=1) / np.std(x, ddof=1))
        knots: list[float] = []
        slopes: list[float] = []
        for _, group in usable.groupby("bin", observed=True):
            if len(group) < config.dynamic_minimum_bin_years:
                continue
            predicted_sd = float(group["predicted"].std(ddof=1))
            observed_sd = float(group["observed"].std(ddof=1))
            if not np.isfinite(predicted_sd) or predicted_sd <= 0 or not np.isfinite(observed_sd):
                continue
            local = observed_sd / predicted_sd
            shrink = len(group) / (len(group) + config.dynamic_shrinkage_years)
            shrunk = float(
                np.exp(shrink * np.log(max(local, 1e-12)) + (1.0 - shrink) * np.log(max(global_slope, 1e-12)))
            )
            knots.append(float(group[availability_name].median()))
            slopes.append(shrunk)
        if len(knots) < 2:
            fallback = replace(
                config,
                method="variance",
                variance_reference="observation",
            )
            return fit_amplitude_calibrator(
                observed,
                predicted,
                fallback,
                availability=availability,
                effective_availability=effective_availability,
            )
        order = np.argsort(knots)
        knots_array = np.asarray(knots, dtype=float)[order]
        slopes_array = np.asarray(slopes, dtype=float)[order]
        lower, upper = config.slope_bounds
        slopes_array = np.clip(slopes_array, lower, upper)
        training_availability = joined[availability_name].to_numpy(float)
        local_training_slopes = np.interp(
            training_availability, knots_array, slopes_array
        )
        source_center = float(np.mean(x))
        target_center = float(
            np.mean(y) - np.mean(local_training_slopes * (x - source_center))
        )
        representative_slope = float(np.median(local_training_slopes))
        return AmplitudeCalibrator(
            method="dynamic_variance",
            intercept=target_center - representative_slope * source_center,
            slope=representative_slope,
            overlap=len(joined),
            unclipped_slope=global_slope,
            clipped=bool(np.any(np.asarray(slopes) != slopes_array)),
            reference="observation_dynamic",
            reference_overlap=len(joined),
            source_center=source_center,
            target_center=target_center,
            availability_knots=tuple(float(value) for value in knots_array),
            slope_knots=tuple(float(value) for value in slopes_array),
        )
    else:  # guarded by the typed configuration, retained for direct calls
        raise ValueError(f"unknown amplitude calibration method: {config.method}")
    lower, upper = config.slope_bounds
    clipped_slope = float(np.clip(slope, lower, upper))
    target_mean = float(
        np.mean(reference_values) if config.method == "variance" else np.mean(y)
    )
    intercept = float(target_mean - clipped_slope * np.mean(x))
    return AmplitudeCalibrator(
        method=config.method,
        intercept=intercept,
        slope=clipped_slope,
        overlap=len(joined),
        unclipped_slope=slope,
        clipped=not np.isclose(slope, clipped_slope),
        reference=(
            config.variance_reference if config.method == "variance" else "observation"
        ),
        reference_overlap=(
            reference_overlap if config.method == "variance" else len(joined)
        ),
        reference_min_proxy_count=(
            reference_min_proxy_count if config.method == "variance" else None
        ),
    )

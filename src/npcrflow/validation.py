"""Leakage-aware temporal validation and paleoclimate skill metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .low_frequency import lowpass_annual_state
from .screening import correlation_with_effective_dof


def reconstruction_metrics(
    observed: pd.Series,
    predicted: pd.Series,
    *,
    calibration_mean: float | None = None,
) -> dict[str, float]:
    """Compute standard validation statistics.

    RE uses the calibration-period mean as its reference.  CE uses the mean of
    the independent validation observations.  This is intentionally different
    from the legacy notebooks' variance and median denominators.
    """

    joined = pd.concat(
        [observed.rename("observed"), predicted.rename("predicted")],
        axis=1,
        join="inner",
    ).dropna()
    if joined.empty:
        return {
            key: np.nan
            for key in (
                "n", "r", "p_effective", "n_eff", "rmse", "mae", "bias",
                "observed_sd", "predicted_sd", "sd_ratio", "variance_ratio",
                "r2_skill", "re", "ce",
            )
        }
    y = joined["observed"].to_numpy(float)
    y_hat = joined["predicted"].to_numpy(float)
    residual = y - y_hat
    sse = float(np.sum(residual**2))
    validation_mean = float(np.mean(y))
    calibration_mean = validation_mean if calibration_mean is None else float(calibration_mean)
    ce_denominator = float(np.sum((y - validation_mean) ** 2))
    re_denominator = float(np.sum((y - calibration_mean) ** 2))
    r, p_value, n_eff, n = correlation_with_effective_dof(
        joined["observed"], joined["predicted"]
    )
    observed_sd = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    predicted_sd = float(np.std(y_hat, ddof=1)) if len(y_hat) > 1 else np.nan
    sd_ratio = predicted_sd / observed_sd if observed_sd > 0 else np.nan
    return {
        "n": int(n),
        "r": r,
        "p_effective": p_value,
        "n_eff": n_eff,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(y_hat - y)),
        "observed_sd": observed_sd,
        "predicted_sd": predicted_sd,
        "sd_ratio": sd_ratio,
        "variance_ratio": sd_ratio**2 if np.isfinite(sd_ratio) else np.nan,
        "r2_skill": 1.0 - sse / ce_denominator if ce_denominator > 0 else np.nan,
        "re": 1.0 - sse / re_denominator if re_denominator > 0 else np.nan,
        "ce": 1.0 - sse / ce_denominator if ce_denominator > 0 else np.nan,
    }


def multiscale_reconstruction_metrics(
    observed: pd.Series,
    predicted: pd.Series,
    periods: Iterable[float],
    *,
    calibration_mean: float | None = None,
) -> dict[str, float]:
    """Evaluate low-pass agreement without altering either fitted input.

    Filtering is applied only to the already-produced observation and
    reconstruction series for diagnosis. It never fills or interpolates a
    proxy. At least two requested low-frequency periods are required so a
    short validation block is reported as unavailable rather than persuasive.
    """

    joined = pd.concat(
        [observed.rename("observed"), predicted.rename("predicted")],
        axis=1,
        join="inner",
    ).dropna()
    result: dict[str, float] = {}
    for period in periods:
        label = f"{float(period):g}".replace(".", "p")
        prefix = f"lowpass_{label}y_"
        minimum = max(8, int(np.ceil(2.0 * float(period))))
        if len(joined) < minimum:
            result.update(
                {
                    f"{prefix}n": float(len(joined)),
                    f"{prefix}r": np.nan,
                    f"{prefix}sd_ratio": np.nan,
                    f"{prefix}variance_ratio": np.nan,
                    f"{prefix}rmse": np.nan,
                    f"{prefix}re": np.nan,
                    f"{prefix}ce": np.nan,
                }
            )
            continue
        observed_low = lowpass_annual_state(joined["observed"], float(period))
        predicted_low = lowpass_annual_state(joined["predicted"], float(period))
        metrics = reconstruction_metrics(
            observed_low,
            predicted_low,
            calibration_mean=calibration_mean,
        )
        for name in ("n", "r", "sd_ratio", "variance_ratio", "rmse", "re", "ce"):
            result[f"{prefix}{name}"] = metrics[name]
    return result


def spectral_reconstruction_metrics(
    observed: pd.Series,
    predicted: pd.Series,
    period_bands: Iterable[tuple[float, float]],
    *,
    minimum_cycles: float = 1.5,
) -> dict[str, float]:
    """Compare frequency-band amplitude and phase on a common native grid.

    The longest genuinely consecutive common annual segment is used; gaps are
    never interpolated. A band is unavailable unless the segment spans at
    least ``minimum_cycles`` of its slowest period. Spectra are calculated
    after linear detrending and a shared Hann taper.
    """

    joined = pd.concat(
        [observed.rename("observed"), predicted.rename("predicted")],
        axis=1,
        join="inner",
    ).dropna().sort_index()
    if not joined.empty:
        years = joined.index.to_numpy(int)
        split_points = np.flatnonzero(np.diff(years) != 1) + 1
        segments = np.split(np.arange(len(joined)), split_points)
        longest = max(segments, key=len) if segments else np.array([], dtype=int)
        joined = joined.iloc[longest]
    result: dict[str, float] = {}
    for minimum, maximum in period_bands:
        low_label = f"{float(minimum):g}".replace(".", "p")
        high_label = f"{float(maximum):g}".replace(".", "p")
        prefix = f"spectral_{low_label}_{high_label}y_"
        unavailable = {
            f"{prefix}n": float(len(joined)),
            f"{prefix}frequency_bins": 0.0,
            f"{prefix}observed_peak_period": np.nan,
            f"{prefix}predicted_peak_period": np.nan,
            f"{prefix}peak_period_error": np.nan,
            f"{prefix}amplitude_ratio": np.nan,
            f"{prefix}phase_similarity": np.nan,
            f"{prefix}magnitude_similarity": np.nan,
        }
        if len(joined) < int(np.ceil(minimum_cycles * maximum)):
            result.update(unavailable)
            continue
        count = len(joined)
        coordinate = np.arange(count, dtype=float)

        def detrended(values: np.ndarray) -> np.ndarray:
            slope, intercept = np.polyfit(coordinate, values, 1)
            return values - (slope * coordinate + intercept)

        taper = np.hanning(count)
        observed_values = detrended(joined["observed"].to_numpy(float)) * taper
        predicted_values = detrended(joined["predicted"].to_numpy(float)) * taper
        frequency = np.fft.rfftfreq(count, d=1.0)
        observed_fft = np.fft.rfft(observed_values)
        predicted_fft = np.fft.rfft(predicted_values)
        band = (frequency >= 1.0 / maximum) & (frequency <= 1.0 / minimum)
        band[0] = False
        if int(band.sum()) < 2:
            result.update(unavailable)
            continue
        observed_band = observed_fft[band]
        predicted_band = predicted_fft[band]
        observed_power = float(np.sum(np.abs(observed_band) ** 2))
        predicted_power = float(np.sum(np.abs(predicted_band) ** 2))
        denominator = np.sqrt(observed_power * predicted_power)
        if observed_power <= 0 or predicted_power <= 0 or denominator <= 0:
            result.update(unavailable)
            continue
        cross = np.sum(np.conj(observed_band) * predicted_band)
        band_frequency = frequency[band]
        observed_peak = float(
            1.0 / band_frequency[int(np.argmax(np.abs(observed_band) ** 2))]
        )
        predicted_peak = float(
            1.0 / band_frequency[int(np.argmax(np.abs(predicted_band) ** 2))]
        )
        result.update(
            {
                f"{prefix}n": float(count),
                f"{prefix}frequency_bins": float(band.sum()),
                f"{prefix}observed_peak_period": observed_peak,
                f"{prefix}predicted_peak_period": predicted_peak,
                f"{prefix}peak_period_error": abs(predicted_peak - observed_peak),
                f"{prefix}amplitude_ratio": float(
                    np.sqrt(predicted_power / observed_power)
                ),
                f"{prefix}phase_similarity": float(np.real(cross) / denominator),
                f"{prefix}magnitude_similarity": float(np.abs(cross) / denominator),
            }
        )
    return result


def contiguous_folds(
    years: Iterable[int],
    block_years: int,
    *,
    minimum_train_years: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return rolling, non-random validation blocks over the available years."""

    unique = np.asarray(sorted(set(int(year) for year in years)), dtype=int)
    if unique.size < minimum_train_years + 2:
        return []
    # Permit a requested one-third block to use ceil(n / 3). With 101 years,
    # floor division would silently turn a 34-year request into 33 years and
    # create an unintended fourth fold of only two years.
    maximum_three_fold_block = (unique.size + 2) // 3
    block_years = min(block_years, max(2, maximum_three_fold_block))
    blocks = [unique[start : start + block_years] for start in range(0, unique.size, block_years)]
    # Do not silently lose a one-year remainder (for example 1900–2000 with
    # 20-year blocks). Merge it into the preceding late block so every
    # observation year is withheld exactly once.
    if len(blocks) > 1 and blocks[-1].size < 2:
        blocks[-2] = np.concatenate([blocks[-2], blocks[-1]])
        blocks.pop()
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for validation in blocks:
        train = unique[~np.isin(unique, validation)]
        if validation.size >= 2 and train.size >= minimum_train_years:
            folds.append((train, validation))
    return folds


def summarize_fold_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    result: dict[str, float] = {"fold_count": int(len(frame))}
    for column in ("r", "rmse", "mae", "bias", "r2_skill", "re", "ce"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        result[f"median_{column}"] = float(values.median())
        result[f"q05_{column}"] = float(values.quantile(0.05))
        result[f"q95_{column}"] = float(values.quantile(0.95))
    return result

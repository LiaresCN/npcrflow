"""Legacy-style repeated random calibration/validation inside each NEST.

PCA is built once on that NEST's full proxy matrix. This is an
internal regression ensemble, not a fold-independent PCA validation or a
with-replacement bootstrap. No member is rejected for its individual CE/RE.
"""

from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd

from .amplitude import fit_amplitude_calibrator


def random_holdout_indices(n, count, fraction, seed):
    n_validation = int(np.ceil(n * fraction))
    if n_validation < 3 or n - n_validation < 4:
        raise RuntimeError("too few samples for random 2/3 calibration and 1/3 validation")
    rng = np.random.default_rng(seed)
    indices = np.asarray([rng.permutation(n) for _ in range(count)])
    return indices[:, n_validation:], indices[:, :n_validation]


def _adjust_predictions(raw, observed, train, availability, effective, amplitude):
    """Vectorized affine maps, numerically equivalent to amplitude.py."""
    if amplitude.method == "none":
        return raw.copy(), np.ones(raw.shape[1])
    x = raw[train, np.arange(raw.shape[1])[:, None]]
    y = observed[train]
    sx = np.std(x, axis=1, ddof=1)
    center_x = x.mean(axis=1)
    reference = y
    center_y = y.mean(axis=1)
    usable = (sx > 0) & np.isfinite(sx) & (train.shape[1] >= amplitude.minimum_overlap)
    if amplitude.method == "variance" and amplitude.variance_reference == "observation":
        slope = np.std(y, axis=1, ddof=1) / np.maximum(sx, 1e-300)
        usable &= np.std(y, axis=1, ddof=1) > 0
    elif amplitude.method == "ols":
        covariance = np.sum((x - center_x[:, None]) * (y - center_y[:, None]), axis=1)
        slope = covariance / np.maximum((train.shape[1] - 1) * sx**2, 1e-300)
    elif amplitude.method == "variance" and amplitude.variance_reference == "max_proxy_nest":
        slope = np.ones(raw.shape[1])
        for run in range(raw.shape[1]):
            reference = np.empty(0)
            for threshold in sorted(np.unique(availability[train[run]]), reverse=True):
                candidate = x[run, availability[train[run]] >= threshold]
                if len(candidate) >= amplitude.minimum_overlap:
                    reference = candidate
                    break
            if len(reference) < amplitude.minimum_overlap or np.std(reference, ddof=1) <= 0:
                usable[run] = False
            else:
                slope[run] = np.std(reference, ddof=1) / max(sx[run], 1e-300)
                center_y[run] = reference.mean()
    else:
        # Keep the declared dynamic-amplitude option without approximating it.
        result = raw.copy()
        slopes = np.ones(raw.shape[1])
        for run in range(raw.shape[1]):
            indices = train[run]
            calibrator = fit_amplitude_calibrator(
                pd.Series(observed[indices]), pd.Series(raw[indices, run]), amplitude,
                pd.Series(availability[indices]), pd.Series(effective[indices]),
            )
            result[:, run] = calibrator.apply(
                pd.Series(raw[:, run]), pd.Series(effective),
            ).to_numpy()
            slopes[run] = calibrator.slope
        return result, slopes
    slopes = np.where(usable, np.clip(slope, *amplitude.slope_bounds), 1.0)
    intercepts = np.where(usable, center_y - slopes * center_x, 0.0)
    return raw * slopes[None, :] + intercepts[None, :], slopes


def _selection_row(prediction, raw, observed, train, validation, slopes,
                   regression, count, alpha, label, amplitude, stage, config):
    from .model import _passes_internal_ce_re

    y = observed[validation]
    p = prediction[validation, np.arange(prediction.shape[1])[:, None]]
    original = raw[validation, np.arange(raw.shape[1])[:, None]]
    residual = y - p
    sse = np.sum(residual**2, axis=1)
    centered_y = y - y.mean(axis=1, keepdims=True)
    ce_den = np.sum(centered_y**2, axis=1)
    re_den = np.sum((y - observed[train].mean(axis=1)[:, None])**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ce = 1 - sse / np.where(ce_den > 0, ce_den, np.nan)
        re = 1 - sse / np.where(re_den > 0, re_den, np.nan)
        centered_p = p - p.mean(axis=1, keepdims=True)
        r = np.sum(centered_y * centered_p, axis=1) / np.sqrt(
            ce_den * np.sum(centered_p**2, axis=1))
        sd = np.std(p, axis=1, ddof=1) / np.std(y, axis=1, ddof=1)
        raw_sse = np.sum((y - original)**2, axis=1)
        raw_ce = 1 - raw_sse / np.where(ce_den > 0, ce_den, np.nan)
        raw_re = 1 - raw_sse / np.where(re_den > 0, re_den, np.nan)
    med = lambda values: float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
    mce, mre, mr = med(ce), med(re), med(r)
    cet = config.skill_floor if config.min_ce is None else config.min_ce
    ret = config.skill_floor if config.min_re is None else config.min_re
    passes = _passes_internal_ce_re(mce, mre, cet, ret, config.internal_ce_re_comparison)
    error = abs(np.log(max(med(sd), 1e-12)))
    return {
        "selection_stage": stage, "selected": False, "regression": regression,
        "n_components": count, "alpha": alpha, "amplitude_candidate": label,
        "amplitude_method": amplitude.method, "amplitude_reference": amplitude.variance_reference,
        "folds": len(train), "internal_validation_method": "random_holdout",
        "pca_fit_scope": "full_nest_proxy_matrix_once",
        "validation_fraction": config.bootstrap_validation_fraction,
        "median_ce": mce, "median_re": mre, "median_r": mr,
        "minimum_ce": float(np.nanmin(ce)), "minimum_re": float(np.nanmin(re)),
        "median_rmse": med(np.sqrt(np.mean(residual**2, axis=1))),
        "median_sd_ratio": med(sd), "median_variance_ratio": med(sd**2),
        "median_raw_ce": med(raw_ce), "median_raw_re": med(raw_re),
        "median_amplitude_slope": med(slopes), "robust_skill": min(mce, mre),
        "minimum_skill": min(float(np.nanmin(ce)), float(np.nanmin(re))),
        "internal_ce_threshold": cet, "internal_re_threshold": ret,
        "internal_ce_re_comparison": config.internal_ce_re_comparison,
        "passes_internal_ce_re": passes, "passes_skill_floor": passes,
        "sd_ratio_error": error, "amplitude_penalty": config.amplitude.auto_sd_ratio_weight * error,
        "objective": min(mce, mre) + .25 * mr - .001 * count,
        "calibration_samples_per_run": train.shape[1], "validation_samples_per_run": validation.shape[1],
    }


def fit_random_holdout_pcr(matrix, target, train_years, pca_config, config,
                           basis, structural_grid, amplitude_options):
    """Fit repeat regressions with one reusable NEST-local PCA score matrix."""
    from .model import _fit_candidate, _regressor, _score_rows

    columns, means, scales, loadings, _, intercepts, trends, origin, weights, _ = basis
    observed = target.reindex(matrix.index).to_numpy(float)
    # Stable per-NEST seed independent of iteration order or added NESTs.
    identity = repr((columns, tuple(int(y) for y in train_years))).encode()
    digest = int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")
    seed = np.random.SeedSequence([config.random_seed, digest])
    caches = {}
    rows = []
    none = replace(config.amplitude, method="none")
    availability = matrix.reindex(columns=columns).notna().sum(axis=1).to_numpy(float)
    effective = matrix.reindex(columns=columns).notna().mul(weights, axis=1).sum(axis=1).to_numpy(float)
    for regression, requested_count, alpha in structural_grid:
        count = min(requested_count, loadings.shape[1])
        scores = _score_rows(matrix, columns, means, scales, intercepts, trends, origin,
                             loadings, count, pca_config.min_proxies_per_year,
                             pca_config.score_ridge, weights)
        values = scores.to_numpy(float)
        finite = np.isfinite(values).all(axis=1)
        eligible = np.flatnonzero(finite & np.isfinite(observed) & matrix.index.isin(train_years))
        tr, va = random_holdout_indices(len(eligible), config.n_bootstrap,
                                       config.bootstrap_validation_fraction, seed)
        train, validation = eligible[tr], eligible[va]
        if train.shape[1] <= count + 2:
            raise RuntimeError("too few random-calibration samples for retained PCs")
        raw = np.full((len(matrix), config.n_bootstrap), np.nan)
        successes = []
        for run in range(config.n_bootstrap):
            indices = train[run]
            try:
                x, y = values[indices], observed[indices]
                if regression == "ols" or (regression == "ridge" and alpha == 0):
                    # Centered least squares matches sklearn's intercept fit.
                    xm, ym = x.mean(axis=0), y.mean()
                    beta = np.linalg.lstsq(x - xm, y - ym, rcond=None)[0]
                    raw[finite, run] = (values[finite] - xm) @ beta + ym
                else:
                    regressor = _regressor(regression, alpha, count, config)
                    regressor.fit(x, y)
                    raw[finite, run] = np.asarray(regressor.predict(values[finite])).reshape(-1)
                successes.append(run)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
        required = int(np.ceil(config.n_bootstrap * config.minimum_bootstrap_success_fraction))
        if len(successes) < required:
            raise RuntimeError(f"only {len(successes)}/{config.n_bootstrap} random holdout fits succeeded")
        raw, train, validation = raw[:, successes], train[successes], validation[successes]
        row = _selection_row(raw, raw, observed, train, validation, np.ones(len(successes)),
                             regression, count, alpha, "none", none, "structure", config)
        row["regression_order"] = [item[0] for item in structural_grid].index(regression)
        rows.append(row)
        caches[(regression, count, alpha)] = (raw, train, validation, scores, successes)
    structural = pd.DataFrame(rows)
    pool = structural.loc[structural.passes_internal_ce_re]
    if pool.empty:
        pool = structural
    best = pool.sort_values(["objective", "median_rmse", "regression_order"],
                            ascending=[False, True, True]).iloc[0]
    structural.loc[best.name, "selected"] = True
    name, count, alpha = str(best.regression), int(best.n_components), float(best.alpha)
    raw, train, validation, scores, successes = caches[(name, count, alpha)]
    adjusted = {}
    amplitude_rows = []
    for label, amplitude in amplitude_options:
        predictions, slopes = _adjust_predictions(raw, observed, train, availability, effective, amplitude)
        adjusted[label] = predictions
        amplitude_rows.append(_selection_row(predictions, raw, observed, train, validation,
                                             slopes, name, count, alpha, label, amplitude,
                                             "amplitude", config))
    amplitudes = pd.DataFrame(amplitude_rows)
    pool = amplitudes.loc[amplitudes.passes_internal_ce_re]
    if pool.empty:
        pool = amplitudes
    selected = pool.sort_values(["robust_skill", "median_r", "sd_ratio_error", "median_rmse"],
                                ascending=[False, False, True, True]).iloc[0]
    amplitudes.loc[selected.name, "selected"] = True
    label = str(selected.amplitude_candidate)
    amplitude = dict(amplitude_options)[label]
    model = _fit_candidate(matrix, target, train_years, pca_config, config, count, alpha,
                           regression_name=name, basis=basis, amplitude_config=amplitude,
                           precomputed_scores=scores)
    model.selection_table = pd.concat([structural, amplitudes], ignore_index=True)
    model.ensemble_predictions = pd.DataFrame(adjusted[label], index=matrix.index, columns=successes)
    model.ensemble_raw_predictions = pd.DataFrame(raw, index=matrix.index, columns=successes)
    return model

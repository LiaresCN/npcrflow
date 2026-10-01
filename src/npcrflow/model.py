"""Missing-data PCR and regularized regression for the prepared proxy matrix."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge

from .amplitude import AmplitudeCalibrator, fit_amplitude_calibrator
from .config import (
    AmplitudeCalibrationConfig,
    PCAConfig,
    ProxyWeightConfig,
    ReconstructionConfig,
)
from .validation import contiguous_folds, reconstruction_metrics
from .weighting import compute_proxy_weights


@dataclass
class NativePCRModel:
    columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    trend_intercepts: np.ndarray
    trend_slopes: np.ndarray
    trend_origin: float
    loadings: np.ndarray
    eigenvalues: np.ndarray
    n_components: int
    alpha: float
    regression_name: str
    regressor: Any
    min_proxies_per_year: int
    score_ridge: float
    proxy_weights: np.ndarray
    weight_table: pd.DataFrame
    amplitude_config: AmplitudeCalibrationConfig
    amplitude_calibrator: AmplitudeCalibrator
    selection_table: pd.DataFrame

    def scores(self, matrix: pd.DataFrame) -> pd.DataFrame:
        aligned = matrix.reindex(columns=self.columns)
        raw = aligned.to_numpy(float)
        years = aligned.index.to_numpy(float) - self.trend_origin
        trends = self.trend_intercepts + years[:, None] * self.trend_slopes
        values = (raw - trends - self.means) / self.scales
        values = values * np.sqrt(self.proxy_weights)[None, :]
        loadings = self.loadings[:, : self.n_components]
        score_values = np.full((len(aligned), self.n_components), np.nan)
        required = self.min_proxies_per_year
        for row_number, row in enumerate(values):
            observed = np.isfinite(row)
            if observed.sum() < required:
                continue
            design = loadings[observed, :]
            if np.linalg.matrix_rank(design) == self.n_components:
                score_values[row_number] = np.linalg.lstsq(design, row[observed], rcond=None)[0]
            else:
                gram = design.T @ design + self.score_ridge * np.eye(self.n_components)
                score_values[row_number] = np.linalg.solve(gram, design.T @ row[observed])
        return pd.DataFrame(
            score_values,
            index=aligned.index,
            columns=[f"PC{i + 1}" for i in range(self.n_components)],
        )

    def predict_raw(self, matrix: pd.DataFrame) -> pd.Series:
        scores = self.scores(matrix)
        finite = np.all(np.isfinite(scores.to_numpy(float)), axis=1)
        result = pd.Series(np.nan, index=matrix.index, name="reconstruction", dtype=float)
        if finite.any():
            predicted = np.asarray(self.regressor.predict(scores.loc[finite])).reshape(-1)
            result.loc[finite] = predicted
        return result

    def predict(self, matrix: pd.DataFrame) -> pd.Series:
        """Predict with the declared training-only amplitude calibration."""

        effective_availability = (
            matrix.reindex(columns=self.columns)
            .notna()
            .mul(self.proxy_weights, axis=1)
            .sum(axis=1)
            .rename("effective_availability")
        )
        return self.amplitude_calibrator.apply(
            self.predict_raw(matrix), effective_availability
        )


def _fit_pairwise_basis(
    matrix: pd.DataFrame,
    fit_years: Iterable[int],
    config: PCAConfig,
    detrend: bool,
    weight_config: ProxyWeightConfig,
) -> tuple:
    fit = matrix.loc[list(fit_years)]
    minimum = max(4, config.min_pairwise_overlap)
    valid_columns = [
        column
        for column in fit.columns
        if fit[column].notna().sum() >= minimum and fit[column].std(skipna=True, ddof=1) > 0
    ]
    if not valid_columns:
        raise RuntimeError("no proxy has enough finite calibration values")
    fit = fit[valid_columns]
    weight_matrix = matrix.reindex(columns=valid_columns).copy()
    weight_matrix.attrs = matrix.attrs.copy()
    weight_table = compute_proxy_weights(weight_matrix, fit_years, weight_config)
    weight_lookup = weight_table.set_index("pid")["weight"]
    proxy_weights = weight_lookup.reindex(valid_columns).to_numpy(float)
    raw = fit.to_numpy(float)
    trend_origin = float(np.mean(fit.index.to_numpy(float)))
    centered_years = fit.index.to_numpy(float) - trend_origin
    trend_intercepts = np.zeros(len(valid_columns), dtype=float)
    trend_slopes = np.zeros(len(valid_columns), dtype=float)
    if detrend:
        for column in range(len(valid_columns)):
            finite = np.isfinite(raw[:, column])
            if finite.sum() >= 3:
                slope, intercept = np.polyfit(centered_years[finite], raw[finite, column], 1)
                trend_intercepts[column] = intercept
                trend_slopes[column] = slope
    residual = raw - trend_intercepts - centered_years[:, None] * trend_slopes
    means = np.nanmean(residual, axis=0)
    scales = np.nanstd(residual, axis=0, ddof=1)
    standardized = (residual - means) / scales
    count = len(valid_columns)
    correlation = np.eye(count)
    for first in range(count):
        for second in range(first + 1, count):
            overlap = np.isfinite(standardized[:, first]) & np.isfinite(standardized[:, second])
            if overlap.sum() >= config.min_pairwise_overlap:
                value = float(np.corrcoef(standardized[overlap, first], standardized[overlap, second])[0, 1])
                if np.isfinite(value):
                    correlation[first, second] = correlation[second, first] = value
    correlation *= np.sqrt(np.outer(proxy_weights, proxy_weights))
    # Pairwise correlations need not form a positive-semidefinite matrix.  A
    # clipped eigendecomposition is the nearest stable spectral basis needed
    # for score estimation; proxy observations themselves remain untouched.
    eigenvalues, eigenvectors = np.linalg.eigh((correlation + correlation.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    loadings = eigenvectors[:, order]
    return (
        tuple(valid_columns), means, scales, loadings, eigenvalues,
        trend_intercepts, trend_slopes, trend_origin, proxy_weights, weight_table,
    )


def _fit_complete_basis(
    matrix: pd.DataFrame,
    fit_years: Iterable[int],
    config: PCAConfig,
    detrend: bool,
    weight_config: ProxyWeightConfig,
) -> tuple:
    fit = matrix.loc[list(fit_years)]
    valid_columns = [column for column in fit if fit[column].notna().all() and fit[column].std(ddof=1) > 0]
    if not valid_columns:
        raise RuntimeError("no complete proxy column is available in calibration")
    fit = fit[valid_columns]
    weight_matrix = matrix.reindex(columns=valid_columns).copy()
    weight_matrix.attrs = matrix.attrs.copy()
    weight_table = compute_proxy_weights(weight_matrix, fit_years, weight_config)
    weight_lookup = weight_table.set_index("pid")["weight"]
    proxy_weights = weight_lookup.reindex(valid_columns).to_numpy(float)
    raw = fit.to_numpy(float)
    trend_origin = float(np.mean(fit.index.to_numpy(float)))
    centered_years = fit.index.to_numpy(float) - trend_origin
    trend_intercepts = np.zeros(len(valid_columns), dtype=float)
    trend_slopes = np.zeros(len(valid_columns), dtype=float)
    if detrend:
        for column in range(len(valid_columns)):
            slope, intercept = np.polyfit(centered_years, raw[:, column], 1)
            trend_intercepts[column] = intercept
            trend_slopes[column] = slope
    residual = raw - trend_intercepts - centered_years[:, None] * trend_slopes
    means = residual.mean(axis=0)
    scales = residual.std(axis=0, ddof=1)
    standardized = (residual - means) / scales
    standardized = standardized * np.sqrt(proxy_weights)[None, :]
    _, singular, vectors = np.linalg.svd(standardized, full_matrices=False)
    eigenvalues = singular**2 / max(len(fit) - 1, 1)
    return (
        tuple(valid_columns), means, scales, vectors.T, eigenvalues,
        trend_intercepts, trend_slopes, trend_origin, proxy_weights, weight_table,
    )


def _component_candidates(eigenvalues: np.ndarray, config: PCAConfig) -> list[int]:
    maximum = min(config.max_components, len(eigenvalues))
    if config.selection == "fixed":
        return [min(int(config.n_components or 1), maximum)]
    if config.selection in {"kaiser", "kaiser_cv"}:
        count = max(
            1,
            min(int(np.sum(eigenvalues > config.kaiser_threshold)), maximum),
        )
        # ``kaiser`` reproduces the legacy eigenvalue-threshold rule exactly.
        # ``kaiser_cv`` treats it as a physically interpretable upper bound and
        # lets contiguous validation choose among the retained dimensions.
        return [count] if config.selection == "kaiser" else list(range(1, count + 1))
    if config.selection == "variance":
        total = eigenvalues.sum()
        if total <= 0:
            return [1]
        count = int(np.searchsorted(np.cumsum(eigenvalues) / total, config.variance_fraction) + 1)
        return [max(1, min(count, maximum))]
    return list(range(1, maximum + 1))


def _regressor(
    name: str,
    alpha: float,
    n_components: int,
    config: ReconstructionConfig,
) -> Any:
    if name == "ols":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=alpha)
    if name == "pls":
        return PLSRegression(
            n_components=max(1, min(config.pls_components, n_components)),
            scale=False,
        )
    if name == "elasticnet":
        return ElasticNet(alpha=max(alpha, 1e-8), l1_ratio=0.5, max_iter=20000)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=config.random_forest_trees,
            min_samples_leaf=config.random_forest_min_samples_leaf,
            max_features=config.random_forest_max_features,
            random_state=config.random_seed,
            n_jobs=1,
        )
    raise ValueError(f"unsupported regression model: {name}")


def _score_rows(
    matrix: pd.DataFrame,
    columns: tuple[str, ...],
    means: np.ndarray,
    scales: np.ndarray,
    trend_intercepts: np.ndarray,
    trend_slopes: np.ndarray,
    trend_origin: float,
    loadings: np.ndarray,
    n_components: int,
    min_proxies: int,
    score_ridge: float,
    proxy_weights: np.ndarray,
) -> pd.DataFrame:
    temporary = NativePCRModel(
        columns=columns,
        means=means,
        scales=scales,
        trend_intercepts=trend_intercepts,
        trend_slopes=trend_slopes,
        trend_origin=trend_origin,
        loadings=loadings,
        eigenvalues=np.empty(0),
        n_components=n_components,
        alpha=0.0,
        regression_name="temporary",
        regressor=None,
        min_proxies_per_year=min_proxies,
        score_ridge=score_ridge,
        proxy_weights=proxy_weights,
        weight_table=pd.DataFrame(),
        amplitude_config=AmplitudeCalibrationConfig(),
        amplitude_calibrator=AmplitudeCalibrator(),
        selection_table=pd.DataFrame(),
    )
    return temporary.scores(matrix)


def _fit_candidate(
    matrix: pd.DataFrame,
    target: pd.Series,
    train_years: Iterable[int],
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
    n_components: int,
    alpha: float,
    regression_name: str | None = None,
    basis: tuple | None = None,
    amplitude_config: AmplitudeCalibrationConfig | None = None,
) -> NativePCRModel:
    basis_function = _fit_pairwise_basis if pca_config.method == "pairwise" else _fit_complete_basis
    if basis is None:
        basis = basis_function(
            matrix, train_years, pca_config, reconstruction_config.detrend_proxies,
            reconstruction_config.proxy_weights,
        )
    (
        columns, means, scales, loadings, eigenvalues,
        trend_intercepts, trend_slopes, trend_origin, proxy_weights, weight_table,
    ) = basis
    n_components = min(n_components, loadings.shape[1])
    scores = _score_rows(
        matrix,
        columns,
        means,
        scales,
        trend_intercepts,
        trend_slopes,
        trend_origin,
        loadings,
        n_components,
        pca_config.min_proxies_per_year,
        pca_config.score_ridge,
        proxy_weights,
    )
    years = [year for year in train_years if year in target.index and year in scores.index]
    joined = scores.loc[years].copy()
    joined["target"] = target.reindex(years).to_numpy(float)
    joined = joined.dropna()
    if len(joined) <= n_components + 2:
        raise RuntimeError("too few finite calibration years for regression")
    regression_name = regression_name or reconstruction_config.regression
    if regression_name == "auto":
        raise ValueError("a concrete regression model is required for candidate fitting")
    regressor = _regressor(
        regression_name,
        alpha,
        n_components,
        reconstruction_config,
    )
    regressor.fit(joined.iloc[:, :n_components], joined["target"])
    amplitude_config = amplitude_config or reconstruction_config.amplitude
    if amplitude_config.method == "auto":
        raise ValueError("automatic amplitude choice requires inner model selection")
    model = NativePCRModel(
        columns=columns,
        means=means,
        scales=scales,
        trend_intercepts=trend_intercepts,
        trend_slopes=trend_slopes,
        trend_origin=trend_origin,
        loadings=loadings,
        eigenvalues=eigenvalues,
        n_components=n_components,
        alpha=alpha,
        regression_name=regression_name,
        regressor=regressor,
        min_proxies_per_year=pca_config.min_proxies_per_year,
        score_ridge=pca_config.score_ridge,
        proxy_weights=proxy_weights,
        weight_table=weight_table,
        amplitude_config=amplitude_config,
        amplitude_calibrator=AmplitudeCalibrator(),
        selection_table=pd.DataFrame(),
    )
    raw_training_prediction = model.predict_raw(matrix.reindex(joined.index))
    training_availability = (
        matrix.reindex(joined.index)
        .reindex(columns=columns)
        .notna()
        .sum(axis=1)
        .rename("availability")
    )
    training_effective_availability = (
        matrix.reindex(joined.index)
        .reindex(columns=columns)
        .notna()
        .mul(proxy_weights, axis=1)
        .sum(axis=1)
        .rename("effective_availability")
    )
    model.amplitude_calibrator = fit_amplitude_calibrator(
        target.reindex(joined.index),
        raw_training_prediction,
        amplitude_config,
        availability=training_availability,
        effective_availability=training_effective_availability,
    )
    return model


def fit_native_pcr(
    matrix: pd.DataFrame,
    target: pd.Series,
    train_years: Iterable[int],
    pca_config: PCAConfig,
    reconstruction_config: ReconstructionConfig,
    *,
    forced_n_components: int | None = None,
    forced_alpha: float | None = None,
    forced_regression: str | None = None,
    forced_amplitude: AmplitudeCalibrationConfig | None = None,
) -> NativePCRModel:
    """Fit missing-data PCR and select model family/hyperparameters by blocks."""

    train_years = np.asarray(list(train_years), dtype=int)
    basis_function = _fit_pairwise_basis if pca_config.method == "pairwise" else _fit_complete_basis
    full_basis = basis_function(
        matrix, train_years, pca_config, reconstruction_config.detrend_proxies,
        reconstruction_config.proxy_weights,
    )
    eigenvalues = full_basis[4]
    component_candidates = (
        [forced_n_components]
        if forced_n_components is not None
        else _component_candidates(eigenvalues, pca_config)
    )
    regression_candidates = (
        [forced_regression]
        if forced_regression is not None
        else list(
            reconstruction_config.regression_candidates
            if reconstruction_config.regression == "auto"
            else (reconstruction_config.regression,)
        )
    )
    if not reconstruction_config.auto_tune:
        component_candidates = component_candidates[:1]
        regression_candidates = regression_candidates[:1]

    def alphas_for(regression_name: str) -> list[float]:
        if forced_alpha is not None:
            return [float(forced_alpha)]
        values = (
            reconstruction_config.ridge_alphas
            if regression_name in {"ridge", "elasticnet"}
            else (0.0,)
        )
        result = [float(value) for value in values]
        return result[:1] if not reconstruction_config.auto_tune else result

    def amplitude_candidates() -> list[tuple[str, AmplitudeCalibrationConfig]]:
        configured = reconstruction_config.amplitude
        if forced_amplitude is not None:
            return [(forced_amplitude.method, forced_amplitude)]
        if configured.method != "auto":
            label = configured.method
            if configured.method == "variance":
                label = f"variance_{configured.variance_reference}"
            return [(label, configured)]
        result: list[tuple[str, AmplitudeCalibrationConfig]] = []
        for name in configured.auto_candidates:
            if name == "variance_observation":
                concrete = replace(
                    configured, method="variance", variance_reference="observation"
                )
            elif name == "variance_max_proxy_nest":
                concrete = replace(
                    configured, method="variance", variance_reference="max_proxy_nest"
                )
            else:
                concrete = replace(configured, method=name)
            result.append((name, concrete))
        return result

    amplitude_options = amplitude_candidates()
    if not reconstruction_config.auto_tune:
        amplitude_options = amplitude_options[:1]
    structural_grid = [
        (str(regression_name), int(n_components), float(alpha))
        for regression_name in regression_candidates
        for n_components in component_candidates
        for alpha in alphas_for(str(regression_name))
    ]
    if not reconstruction_config.auto_tune:
        regression_name, n_components, alpha = structural_grid[0]
        _, amplitude = amplitude_options[0]
        return _fit_candidate(
            matrix,
            target,
            train_years,
            pca_config,
            reconstruction_config,
            n_components,
            alpha,
            regression_name=regression_name,
            basis=full_basis,
            amplitude_config=amplitude,
        )
    target_years = [year for year in sorted(set(train_years)) if year in target.index]
    folds = contiguous_folds(
        target_years,
        reconstruction_config.validation_block_years,
        minimum_train_years=max(20, pca_config.max_components + 3),
    )
    # PCA/covariance depends on the fold's training data but not on PC count or
    # regression alpha. Cache it once per fold instead of recomputing it for
    # every hyperparameter combination.
    fold_bases = [
        basis_function(
            matrix,
            fold_train,
            pca_config,
            reconstruction_config.detrend_proxies,
            reconstruction_config.proxy_weights,
        )
        for fold_train, _ in folds
    ]
    regression_order = {
        name: position for position, name in enumerate(regression_candidates)
    }
    no_amplitude = replace(reconstruction_config.amplitude, method="none")

    def evaluate_candidate(
        regression_name: str,
        n_components: int,
        alpha: float,
        amplitude_label: str,
        amplitude: AmplitudeCalibrationConfig,
        selection_stage: str,
    ) -> dict[str, object] | None:
        fold_metrics: list[dict[str, float]] = []
        for (fold_train, fold_validation), fold_basis in zip(folds, fold_bases):
            try:
                fold_model = _fit_candidate(
                    matrix,
                    target,
                    fold_train,
                    pca_config,
                    reconstruction_config,
                    int(n_components),
                    float(alpha),
                    regression_name=str(regression_name),
                    basis=fold_basis,
                    amplitude_config=amplitude,
                )
                prediction = fold_model.predict(matrix.loc[fold_validation])
                metrics = reconstruction_metrics(
                    target.reindex(fold_validation),
                    prediction,
                    calibration_mean=float(target.reindex(fold_train).mean()),
                )
                raw_metrics = reconstruction_metrics(
                    target.reindex(fold_validation),
                    fold_model.predict_raw(matrix.loc[fold_validation]),
                    calibration_mean=float(target.reindex(fold_train).mean()),
                )
                metrics.update({f"raw_{key}": value for key, value in raw_metrics.items()})
                metrics["amplitude_slope"] = fold_model.amplitude_calibrator.slope
                fold_metrics.append(metrics)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
        metrics = pd.DataFrame(fold_metrics)
        if metrics.empty:
            return None
        median_ce = float(metrics["ce"].median())
        median_re = float(metrics["re"].median())
        median_r = float(metrics["r"].median())
        robust_skill = min(median_ce, median_re)
        minimum_ce = float(metrics["ce"].min())
        minimum_re = float(metrics["re"].min())
        minimum_skill = min(minimum_ce, minimum_re)
        ce_threshold = (
            reconstruction_config.skill_floor
            if reconstruction_config.min_ce is None
            else reconstruction_config.min_ce
        )
        re_threshold = (
            reconstruction_config.skill_floor
            if reconstruction_config.min_re is None
            else reconstruction_config.min_re
        )
        passes_internal_ce_re = bool(
            median_ce >= ce_threshold and median_re >= re_threshold
        )
        median_sd_ratio = float(metrics["sd_ratio"].median())
        sd_ratio_error = abs(np.log(max(median_sd_ratio, 1e-12)))
        return {
            "selection_stage": selection_stage,
            "selected": False,
            "regression": str(regression_name),
            "regression_order": regression_order[str(regression_name)],
            "n_components": int(n_components),
            "alpha": float(alpha),
            "amplitude_candidate": amplitude_label,
            "amplitude_method": amplitude.method,
            "amplitude_reference": (
                amplitude.variance_reference
                if amplitude.method == "variance"
                else (
                    "observation_dynamic"
                    if amplitude.method == "dynamic_variance"
                    else "observation" if amplitude.method == "ols" else "none"
                )
            ),
            "folds": len(metrics),
            "median_ce": median_ce,
            "median_re": median_re,
            "minimum_ce": minimum_ce,
            "minimum_re": minimum_re,
            "median_r": median_r,
            "median_rmse": float(metrics["rmse"].median()),
            "median_sd_ratio": median_sd_ratio,
            "median_variance_ratio": float(metrics["variance_ratio"].median()),
            "median_raw_ce": float(metrics["raw_ce"].median()),
            "median_raw_re": float(metrics["raw_re"].median()),
            "median_amplitude_slope": float(metrics["amplitude_slope"].median()),
            "robust_skill": robust_skill,
            "minimum_skill": minimum_skill,
            "internal_ce_threshold": ce_threshold,
            "internal_re_threshold": re_threshold,
            "passes_internal_ce_re": passes_internal_ce_re,
            # Backward-compatible column name retained for v0.4.1 readers.
            "passes_skill_floor": passes_internal_ce_re,
            # Kept as an audit diagnostic. It is intentionally excluded from
            # structural selection so amplitude cannot change PC count.
            "sd_ratio_error": sd_ratio_error,
            "amplitude_penalty": (
                reconstruction_config.amplitude.auto_sd_ratio_weight
                * sd_ratio_error
            ),
            "objective": robust_skill + 0.25 * median_r - 0.001 * int(n_components),
        }

    structural_rows: list[dict[str, object]] = []
    for regression_name, n_components, alpha in structural_grid:
        row = evaluate_candidate(
            regression_name,
            n_components,
            alpha,
            "none",
            no_amplitude,
            "structure",
        )
        if row is not None:
            structural_rows.append(row)
    structural_selection = pd.DataFrame(structural_rows)
    if structural_selection.empty:
        # Short calibration samples cannot support an inner CV.  Fall back to
        # the simplest configured model and make the absence explicit.
        best_regression, best_components, best_alpha = structural_grid[0]
    else:
        pool = structural_selection[structural_selection["passes_skill_floor"]]
        if pool.empty:
            pool = structural_selection
        best = pool.sort_values(
            ["objective", "median_rmse", "regression_order"],
            ascending=[False, True, True],
        ).iloc[0]
        best_regression = str(best["regression"])
        best_components = int(best["n_components"])
        best_alpha = float(best["alpha"])
        structural_selection.loc[best.name, "selected"] = True

    amplitude_rows: list[dict[str, object]] = []
    for amplitude_label, amplitude in amplitude_options:
        row = evaluate_candidate(
            best_regression,
            best_components,
            best_alpha,
            amplitude_label,
            amplitude,
            "amplitude",
        )
        if row is not None:
            amplitude_rows.append(row)
    amplitude_selection = pd.DataFrame(amplitude_rows)
    if amplitude_selection.empty:
        _, best_amplitude = amplitude_options[0]
    elif len(amplitude_options) == 1:
        best = amplitude_selection.iloc[0]
        best_amplitude = amplitude_options[0][1]
        amplitude_selection.loc[best.name, "selected"] = True
    else:
        pool = amplitude_selection[amplitude_selection["passes_skill_floor"]]
        if pool.empty:
            pool = amplitude_selection
        # Skill is primary. Correlation breaks skill ties; amplitude fidelity
        # is only the third criterion and can no longer buy a worse PCR core.
        best = pool.sort_values(
            ["robust_skill", "median_r", "sd_ratio_error", "median_rmse"],
            ascending=[False, False, True, True],
        ).iloc[0]
        best_label = str(best["amplitude_candidate"])
        best_amplitude = next(
            amplitude for label, amplitude in amplitude_options if label == best_label
        )
        amplitude_selection.loc[best.name, "selected"] = True
    selection = pd.concat(
        [structural_selection, amplitude_selection], ignore_index=True
    )
    model = _fit_candidate(
        matrix,
        target,
        train_years,
        pca_config,
        reconstruction_config,
        best_components,
        best_alpha,
        regression_name=best_regression,
        basis=full_basis,
        amplitude_config=best_amplitude,
    )
    model.selection_table = selection
    return model

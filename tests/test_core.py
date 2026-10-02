from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from npcrflow.config import (
    AmplitudeCalibrationConfig,
    ExplicitNestConfig,
    MultiresolutionConfig,
    PCAConfig,
    OutputConfig,
    PipelineConfig,
    ProxyWeightConfig,
    ProxyFilterConfig,
    ReconstructionConfig,
    ScreeningConfig,
    SensitivityConfig,
    TargetConfig,
)
from npcrflow.amplitude import fit_amplitude_calibrator
from npcrflow.data import annualize_record, load_observations, load_proxy_database
from npcrflow.deduplicate import deduplicate_frame
from npcrflow.model import (
    NativePCRModel, _component_candidates, _fit_candidate,
    _pairwise_correlation, _score_rows, fit_native_pcr,
)
from npcrflow.pipeline import filter_records_with_report, run_pipeline
from npcrflow.nesting import explicit_nest_validation
from npcrflow.records import ProxyCollection, ProxyRecord
from npcrflow.low_frequency import _native_support_widths, variational_low_frequency_adjustment
from npcrflow.reconstruction import blocked_pipeline_validation, blocked_validation
from npcrflow.reconstruction import (
    _longest_contiguous_annual_period,
    build_proxy_matrix,
    prestandardize_proxy_matrix,
    reconstruct,
    split_resolution_roles,
    tune_multiresolution_config,
)
from npcrflow.resolution_nest import build_resolution_matrix, fit_resolution_subnests
from npcrflow.screening import _fdr_bh, correlation_with_effective_dof, screen_proxies
from npcrflow.sensitivity import run_network_sensitivities, summarize_sensitivity
from npcrflow.validation import (
    contiguous_folds,
    directed_edge_folds,
    multiscale_reconstruction_metrics,
    reconstruction_metrics,
    spectral_reconstruction_metrics,
)
from npcrflow.weighting import compute_proxy_weights


class DataTests(unittest.TestCase):
    def test_raw_dod2k_loader_parses_string_year_array(self):
        frame = pd.DataFrame(
            {
                "datasetId": ["a"],
                "year": ["[2000, 2001, 2002]"],
                "paleoData_values": [np.array([1.0, 2.0, 3.0])],
                "yearUnits": ["CE"],
                "geo_meanLat": [1.0],
                "geo_meanLon": [2.0],
                "geo_meanElev": [3.0],
                "archiveType": ["Wood"],
                "paleoData_proxy": ["ring width"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.pkl"
            frame.to_pickle(path)
            records = load_proxy_database(path)
        np.testing.assert_array_equal(records["a"].time, [2000, 2001, 2002])

    def test_proxy_filter_defaults_to_exact_deduplication(self):
        self.assertTrue(ProxyFilterConfig().deduplicate_exact)
        reconstruction = ReconstructionConfig()
        self.assertFalse(OutputConfig().show_external_sensitivities)
        self.assertTrue(reconstruction.full_network_outer_validation)
        self.assertTrue(reconstruction.rescreen_outer_folds)
        self.assertEqual(reconstruction.min_ce, 0.0)
        self.assertEqual(reconstruction.min_re, 0.0)
        self.assertEqual(reconstruction.strong_skill_threshold, 0.5)

    def test_djf_annualization_uses_end_year_without_interpolation(self):
        times = []
        values = []
        for year in (1999, 2000, 2001):
            for month in range(1, 13):
                times.append(year + (month - 1) / 12)
                values.append(float(month))
        record = ProxyRecord("monthly", np.asarray(times), np.asarray(values))
        result = annualize_record(record, months=(12, 1, 2), season_year="end", minimum_month_fraction=1.0)
        self.assertAlmostEqual(result.loc[2000], (12 + 1 + 2) / 3)
        self.assertAlmostEqual(result.loc[2001], (12 + 1 + 2) / 3)
        self.assertNotIn(1999, result.index)

    def test_monthly_observation_table(self):
        frame = pd.DataFrame(
            {
                "Year": [2000, 2001],
                "Jan": [1.0, 2.0],
                "Feb": [3.0, 4.0],
                "Mar": [5.0, 6.0],
            }
        )
        result = load_observations(frame, months=(1, 2, 3), minimum_month_fraction=1.0)
        self.assertEqual(result.loc[2000], 3.0)
        self.assertEqual(result.loc[2001], 4.0)


class DeduplicationTests(unittest.TestCase):
    def test_exact_duplicate_is_removed_and_reported(self):
        frame = pd.DataFrame(
            {
                "datasetId": ["less", "more", "other"],
                "year": [np.array([1, 2, 3]), np.array([3, 2, 1]), np.array([1, 2, 3])],
                "paleoData_values": [np.array([4, 5, 6]), np.array([6, 5, 4]), np.array([4, 5, 7])],
                "originalDataURL": [np.nan, "https://example.test", "https://other.test"],
                "archiveType": ["Wood", "Wood", "Wood"],
            }
        )
        cleaned, report, groups = deduplicate_frame(frame)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(len(report), 1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(report.iloc[0]["kept_dataset_id"], "more")

    def test_rounding_equivalent_cross_database_copy_is_removed(self):
        time = np.array([1900.0, 1911.6, 1923.2, 1934.8])
        first = np.array([-35.123456, -34.876543, -36.000049, -35.500001])
        second = np.round(first, 4)
        frame = pd.DataFrame(
            {
                "datasetId": ["iso2k_record", "pages2k_record"],
                "year": [time, time.copy()],
                "paleoData_values": [first, second],
                "archiveType": ["GlacierIce", "GlacierIce"],
                "paleoData_proxy": ["d18O", "d18O"],
                "paleoData_variableName": ["d18O", "d18O"],
                "geo_meanLat": [-72.82, -72.82],
                "geo_meanLon": [159.18, 159.18],
            }
        )
        cleaned, report, groups = deduplicate_frame(frame)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.loc[0, "reason"], "rounding_equivalent_time_value_duplicate")
        self.assertEqual(groups.loc[0, "group_type"], "rounding_equivalent_time_value_duplicate")

    def test_near_values_at_different_sites_are_retained(self):
        time = np.array([1900.0, 1910.0, 1920.0, 1930.0])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        frame = pd.DataFrame(
            {
                "datasetId": ["site_a", "site_b"],
                "year": [time, time.copy()],
                "paleoData_values": [values, values + 1e-6],
                "archiveType": ["GlacierIce", "GlacierIce"],
                "paleoData_proxy": ["d18O", "d18O"],
                "paleoData_variableName": ["d18O", "d18O"],
                "geo_meanLat": [-72.82, -71.82],
                "geo_meanLon": [159.18, 159.18],
            }
        )
        cleaned, report, _ = deduplicate_frame(frame)
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(report.empty)

    def test_pipeline_reads_original_table_and_writes_duplicate_audit(self):
        years = np.arange(1900, 1960)
        latent = np.sin(np.arange(len(years)) / 6.0)
        rows = []
        for number in range(4):
            rows.append(
                {
                    "datasetId": f"p{number}",
                    "year": years,
                    "paleoData_values": latent * (1 + 0.05 * number),
                    "yearUnits": "CE",
                    "geo_meanLat": 20.0,
                    "geo_meanLon": 120.0,
                    "archiveType": "Wood",
                    "paleoData_proxy": "ring width",
                    "climateInterpretation_variable": "temperature",
                }
            )
        rows.append(rows[0] | {"datasetId": "duplicate_p0"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proxy_path = root / "raw.pkl"
            target_path = root / "target.csv"
            pd.DataFrame(rows).to_pickle(proxy_path)
            pd.DataFrame({"Year": years, "target": latent}).to_csv(target_path, index=False)
            result = run_pipeline(
                proxy_path,
                target_path,
                PipelineConfig(
                    target=TargetConfig(value_column="target"),
                    screening=ScreeningConfig(min_overlap=15, p_threshold=1.0),
                    pca=PCAConfig(
                        selection="fixed", n_components=1, max_components=1,
                        min_proxies_per_year=1,
                    ),
                    reconstruction=ReconstructionConfig(
                        regression="ridge", ridge_alphas=(0.1,), validation_block_years=15,
                        n_bootstrap=0, auto_tune=False,
                    ),
                    output=OutputConfig(
                        directory=root / "output",
                        save_proxy_map=False,
                        save_observation_plot=False,
                    ),
                ),
            )
            self.assertEqual(len(result.duplicate_report), 1)
            self.assertEqual(len(result.source_qc), 4)
            self.assertTrue((root / "output" / "dropped_exact_duplicates.csv").exists())
            self.assertTrue((root / "output" / "observation_fit.csv").exists())
            primary = pd.read_csv(root / "output" / "primary_reconstruction_summary.csv")
            self.assertEqual(
                primary.loc[0, "validity_basis"],
                "observation_correlation_and_internal_npcr_ce_re",
            )
            self.assertFalse(bool(primary.loc[0, "external_sensitivities_decide_main_result"]))
            self.assertEqual(len(result.reconstruction.model.columns), 4)
            self.assertIn("state_support", result.reconstruction.availability)
            modes = set(result.reconstruction.validation_folds["validation_mode"])
            self.assertEqual(modes, {"full_proxy_network", "rescreened_network"})
            fixed = result.reconstruction.validation_folds.query(
                "validation_mode == 'full_proxy_network'"
            )
            rescreened = result.reconstruction.validation_folds.query(
                "validation_mode == 'rescreened_network'"
            )
            self.assertFalse(fixed["screening_refit"].any())
            self.assertTrue(rescreened["screening_refit"].all())
            self.assertEqual(
                set(result.reconstruction.validation_summary["validation_role"]),
                {"sensitivity_only"},
            )
            self.assertEqual(
                set(result.reconstruction.validation_summary["assessment_metric"]),
                {"correlation"},
            )
            self.assertNotIn("skill_class", result.reconstruction.validation_summary)
            self.assertTrue((root / "output" / "validation_full_proxy_network.csv").exists())
            self.assertTrue((root / "output" / "validation_rescreened_network.csv").exists())


class ScreeningTests(unittest.TestCase):
    def test_bh_fdr_adjusts_across_proxy_candidates(self):
        adjusted = _fdr_bh([0.01, 0.04, 0.03, np.nan])
        np.testing.assert_allclose(adjusted[:3], [0.03, 0.04, 0.04])
        self.assertTrue(np.isnan(adjusted[3]))

    def test_archive_cap_keeps_best_ranked_records(self):
        years = np.arange(1900, 1950)
        target = pd.Series(np.sin(np.arange(len(years)) / 6.0), index=years)
        records = {
            f"wood_{number}": ProxyRecord(
                f"wood_{number}",
                years,
                target.to_numpy() + number * 0.01 * np.cos(np.arange(len(years))),
                archive="Wood",
            )
            for number in range(4)
        }
        screened = screen_proxies(
            records,
            target,
            ScreeningConfig(
                min_overlap=20,
                p_threshold=0.1,
                proxy_multiple_testing="none",
                archive_max_counts=(("Wood", 2),),
            ),
        )
        self.assertEqual(int(screened["selected_before_archive_cap"].sum()), 4)
        self.assertEqual(int(screened["selected"].sum()), 2)
        self.assertEqual(int(screened["excluded_by_archive_cap"].sum()), 2)

    def test_effective_sample_correlation(self):
        years = pd.Index(np.arange(1900, 1950), name="Year")
        x = pd.Series(np.sin(np.arange(50) / 7), index=years)
        y = x * 2
        r, p_value, n_eff, n = correlation_with_effective_dof(x, y)
        self.assertGreater(r, 0.999)
        self.assertLess(p_value, 1e-6)
        self.assertLessEqual(n_eff, n)

    def test_annual_record_is_not_multiplied_by_auto_seasons(self):
        years = np.arange(1900, 1950)
        record = ProxyRecord("annual", years, np.sin(years / 5))
        target = pd.Series(np.sin(years / 5), index=years)
        result = screen_proxies(
            {"annual": record},
            target,
            ScreeningConfig(season_mode="auto", min_overlap=20),
        )
        self.assertEqual(result.loc[0, "n_seasons_tested"], 1)
        self.assertTrue(bool(result.loc[0, "selected"]))

    def test_low_resolution_record_can_use_separate_overlap_threshold(self):
        years = np.arange(1900, 2000, 10)
        target_years = np.arange(1900, 2000)
        target = pd.Series(np.sin((target_years - 1900) / 10), index=target_years)
        record = ProxyRecord("decadal", years, target.reindex(years).to_numpy())
        result = screen_proxies(
            {"decadal": record},
            target,
            ScreeningConfig(
                min_overlap=20,
                low_resolution_min_overlap=8,
                low_resolution_cutoff_years=5,
                p_threshold=1.0,
            ),
        )
        self.assertEqual(result.loc[0, "minimum_overlap_required"], 8)
        self.assertTrue(bool(result.loc[0, "selected"]))

    def test_metadata_filter_reports_all_exclusion_reasons(self):
        valid = ProxyRecord(
            "valid", [1900, 1901], [1, 2], lat=20, lon=120,
            archive="Wood", proxy="ring width",
            metadata={"climateInterpretation_variable": "temperature"},
        )
        excluded = ProxyRecord(
            "excluded", [1900, 2200], [1, 2], lat=80, lon=120,
            archive="LakeSediment", proxy="temperature",
            metadata={"climateInterpretation_variable": "moisture"},
        )
        selected, report = filter_records_with_report(
            ProxyCollection([valid, excluded]),
            ProxyFilterConfig(
                latitude=(0, 60), archives=("Wood",), proxies=("ring width",),
                climate_variables=("temperature",), maximum_plausible_year=2100,
            ),
        )
        self.assertEqual(list(selected), ["valid"])
        reasons = report.set_index("pid").loc["excluded", "exclusion_reasons"]
        self.assertIn("end_after_maximum_plausible_year", reasons)
        self.assertIn("outside_latitude_filter", reasons)
        self.assertIn("archive_filter", reasons)

    def test_metadata_filter_can_cap_native_resolution_for_ablation(self):
        annual = ProxyRecord("annual", [1900, 1901, 1902], [1, 2, 3])
        decadal = ProxyRecord("decadal", [1900, 1910, 1920], [1, 2, 3])
        selected, report = filter_records_with_report(
            ProxyCollection([annual, decadal]),
            ProxyFilterConfig(maximum_resolution_years=5.0),
        )
        self.assertEqual(list(selected), ["annual"])
        reason = report.set_index("pid").loc["decadal", "exclusion_reasons"]
        self.assertIn("above_maximum_resolution", reason)


class ModelTests(unittest.TestCase):
    def test_contiguous_folds_include_single_year_remainder(self):
        years = np.arange(1900, 2001)
        folds = contiguous_folds(years, 20)
        held_out = np.concatenate([validation for _, validation in folds])
        np.testing.assert_array_equal(np.sort(held_out), years)
        self.assertEqual((folds[-1][1].min(), folds[-1][1].max()), (1980, 2000))

    def test_contiguous_thirds_leave_about_two_thirds_for_training(self):
        years = np.arange(1900, 2001)
        folds = contiguous_folds(years, 34)
        self.assertEqual(len(folds), 3)
        self.assertEqual([len(validation) for _, validation in folds], [34, 34, 33])
        self.assertEqual([len(training) for training, _ in folds], [67, 67, 68])

    def test_directed_edge_folds_use_two_thirds_in_both_directions(self):
        years = np.arange(1900, 2001)
        folds = directed_edge_folds(years)
        self.assertEqual(len(folds), 2)
        self.assertEqual([len(validation) for _, validation in folds], [34, 34])
        self.assertEqual([len(training) for training, _ in folds], [67, 67])
        self.assertEqual((folds[0][1].min(), folds[0][1].max()), (1900, 1933))
        self.assertEqual((folds[1][1].min(), folds[1][1].max()), (1967, 2000))

    def test_reconstruction_period_can_fix_start_and_infer_end(self):
        records = {"p": ProxyRecord("p", [1900, 1901, 1902], [1, 2, 3])}
        matrix = build_proxy_matrix(
            records,
            ["p"],
            ScreeningConfig(),
            reconstruction_period=(0, None),
        )
        self.assertEqual(matrix.index.min(), 0)
        self.assertEqual(matrix.index.max(), 1902)

    def test_pairwise_pcr_handles_missing_values_without_filling_matrix(self):
        rng = np.random.default_rng(42)
        years = np.arange(1900, 2000)
        latent = np.sin(np.arange(100) / 8) + 0.25 * np.cos(np.arange(100) / 3)
        matrix = pd.DataFrame(index=pd.Index(years, name="Year"))
        for number in range(4):
            matrix[f"p{number}"] = latent * (1 + number / 10) + rng.normal(0, 0.12, len(years))
        matrix.loc[years[:20], "p0"] = np.nan
        matrix.loc[years[::4], "p1"] = np.nan
        matrix.loc[years[1::5], "p2"] = np.nan
        target = pd.Series(latent + rng.normal(0, 0.08, len(years)), index=years)
        missing_before = int(matrix.isna().sum().sum())
        model = fit_native_pcr(
            matrix,
            target,
            years,
            PCAConfig(method="pairwise", selection="blocked_cv", max_components=2, min_pairwise_overlap=8),
            ReconstructionConfig(regression="ridge", ridge_alphas=(0.0, 0.1, 1.0), validation_block_years=20, n_bootstrap=0),
        )
        prediction = model.predict(matrix)
        correlation = prediction.corr(target)
        self.assertGreater(correlation, 0.8)
        self.assertEqual(int(matrix.isna().sum().sum()), missing_before)

    def test_training_score_optimization_preserves_full_reconstruction(self):
        rng = np.random.default_rng(123)
        years = np.arange(1800, 2000)
        latent = np.sin((years - 1800) / 8)
        matrix = pd.DataFrame({
            f"p{i}": latent + rng.normal(0, 0.15, len(years))
            for i in range(3)
        }, index=years)
        matrix.loc[years[::4], "p1"] = np.nan
        target = pd.Series(latent, index=years)
        # Duplicate sampled years exercise the bootstrap alignment too.
        train = np.repeat(years[-60:], 2)
        pca = PCAConfig(selection="fixed", n_components=2, max_components=2)
        config = ReconstructionConfig(
            auto_tune=False, regression="ridge", ridge_alphas=(0.1,),
            amplitude=AmplitudeCalibrationConfig(method="variance"), n_bootstrap=0,
        )
        optimized = fit_native_pcr(matrix, target, train, pca, config)

        def legacy_scores(_matrix, *args):
            return _score_rows(matrix, *args)

        with patch("npcrflow.model._score_rows", side_effect=legacy_scores):
            full_matrix_fit = fit_native_pcr(matrix, target, train, pca, config)
        np.testing.assert_allclose(
            optimized.predict(matrix), full_matrix_fit.predict(matrix),
            atol=1e-12, rtol=1e-12, equal_nan=True,
        )

    def test_fold_score_cache_preserves_selection_and_full_prediction(self):
        rng = np.random.default_rng(71)
        years = np.arange(1800, 2001)
        latent = np.sin((years - 1800) / 8)
        matrix = pd.DataFrame({
            f"p{i}": latent + rng.normal(0, 0.3, len(years))
            for i in range(6)
        }, index=years)
        matrix.loc[years[::4], "p1"] = np.nan
        matrix.loc[years[1::5], "p2"] = np.nan
        target = pd.Series(latent + rng.normal(0, 0.1, len(years)), index=years)
        pca = PCAConfig(selection="blocked_cv", max_components=3)
        config = ReconstructionConfig(
            regression="ridge", ridge_alphas=(0.0, 0.1, 1.0),
            amplitude=AmplitudeCalibrationConfig(method="auto"),
            validation_block_years=20, n_bootstrap=0,
        )
        with patch("npcrflow.model._score_rows", wraps=_score_rows) as calls:
            cached = fit_native_pcr(matrix, target, years[-101:], pca, config)
            cached_calls = calls.call_count

        def uncached_candidate(*args, **kwargs):
            kwargs["precomputed_scores"] = None
            return _fit_candidate(*args, **kwargs)

        # Independently recalculate both candidate training and validation
        # scores; no cache is involved in their fitted coefficients/metrics.
        original_predict_scores = NativePCRModel._predict_scores

        def recalculated_prediction(model, values, _scores):
            return original_predict_scores(model, values, model.scores(values))

        with patch("npcrflow.model._fit_candidate", side_effect=uncached_candidate), \
                patch.object(NativePCRModel, "_predict_scores", recalculated_prediction), \
                patch("npcrflow.model._score_rows", wraps=_score_rows) as calls:
            uncached = fit_native_pcr(matrix, target, years[-101:], pca, config)
            uncached_calls = calls.call_count
        self.assertLess(cached_calls, uncached_calls / 3)
        pd.testing.assert_frame_equal(
            cached.selection_table, uncached.selection_table,
            check_exact=False, atol=1e-12, rtol=1e-12,
        )
        np.testing.assert_allclose(
            cached.predict(matrix), uncached.predict(matrix),
            atol=1e-12, rtol=1e-12, equal_nan=True,
        )

    def test_vectorized_pairwise_pca_matches_finite_overlap_pearson(self):
        rng = np.random.default_rng(68)
        values = rng.normal(size=(120, 7))
        values[rng.random(values.shape) < 0.25] = np.nan
        values[:115, -1] = np.nan
        expected = np.eye(values.shape[1])
        for first in range(values.shape[1]):
            for second in range(first + 1, values.shape[1]):
                overlap = np.isfinite(values[:, first]) & np.isfinite(values[:, second])
                if overlap.sum() >= 8:
                    expected[first, second] = expected[second, first] = np.corrcoef(
                        values[overlap, first], values[overlap, second]
                    )[0, 1]
        np.testing.assert_allclose(_pairwise_correlation(values, 8), expected, atol=1e-14)

    def test_pairwise_constant_overlap_does_not_invent_correlation(self):
        values = np.full((60, 2), np.nan)
        values[:30, 0] = 0.3
        values[:30, 1] = 0.33
        values[30:45, 0] = np.arange(15)
        values[45:, 1] = np.arange(15)
        result = _pairwise_correlation(values, 8)
        self.assertEqual(result[0, 1], 0.0)
        self.assertEqual(result[1, 0], 0.0)

    def test_vectorized_pca_preserves_native_full_period_predictions(self):
        rng = np.random.default_rng(69)
        years = np.arange(1800, 2000)
        truth = np.sin((years - 1800) / 8.0)
        matrix = pd.DataFrame({
            f"p{i}": truth + rng.normal(0, 0.2, len(years))
            for i in range(8)
        }, index=years)
        matrix.mask(rng.random(matrix.shape) < 0.1, inplace=True)
        target = pd.Series(truth, index=years)
        pca = PCAConfig(selection="fixed", n_components=2, max_components=2)
        config = ReconstructionConfig(auto_tune=False, ridge_alphas=(0.1,), n_bootstrap=0)
        fast = fit_native_pcr(matrix, target, years[-80:], pca, config)

        def legacy_correlation(values, minimum):
            result = np.eye(values.shape[1])
            for first in range(values.shape[1]):
                for second in range(first + 1, values.shape[1]):
                    finite = np.isfinite(values[:, first]) & np.isfinite(values[:, second])
                    if finite.sum() >= minimum:
                        value = np.corrcoef(values[finite, first], values[finite, second])[0, 1]
                        if np.isfinite(value):
                            result[first, second] = result[second, first] = value
            return result

        with patch("npcrflow.model._pairwise_correlation", side_effect=legacy_correlation):
            legacy = fit_native_pcr(matrix, target, years[-80:], pca, config)
        np.testing.assert_allclose(
            fast.predict(matrix), legacy.predict(matrix),
            atol=1e-12, rtol=1e-12, equal_nan=True,
        )

    def test_fixed_prestandardization_uses_only_available_training_years(self):
        years = np.arange(1900, 1911)
        matrix = pd.DataFrame(
            {
                "p1": np.arange(len(years), dtype=float),
                "p2": 10.0 + 2.0 * np.arange(len(years), dtype=float),
            },
            index=years,
        )
        # The requested reference extends through 1910, but the final three
        # years are held out. They must not influence the common-period z-score.
        train_years = years[:8]
        standardized = prestandardize_proxy_matrix(
            matrix,
            (1905, 1910),
            allowed_years=train_years,
        )
        np.testing.assert_allclose(standardized.loc[1906], [0.0, 0.0])
        audit = standardized.attrs["standardization_audit"].set_index("pid")
        np.testing.assert_allclose(
            audit.loc[["p1", "p2"], "standardization_mean"], [6.0, 22.0]
        )
        self.assertEqual(set(audit["standardization_reference_count"]), {3})

    def test_standardization_period_must_increase(self):
        with self.assertRaisesRegex(ValueError, "standardization period"):
            ReconstructionConfig(standardization_period=(2000, 1950))

    def test_random_forest_regression_uses_same_native_score_path(self):
        rng = np.random.default_rng(5)
        years = np.arange(1900, 1970)
        latent = np.sin(np.arange(len(years)) / 7.0)
        matrix = pd.DataFrame(
            {
                "p1": latent + rng.normal(0, 0.05, len(years)),
                "p2": latent**2 + rng.normal(0, 0.05, len(years)),
            },
            index=years,
        )
        matrix.loc[years[::5], "p2"] = np.nan
        target = pd.Series(latent + 0.4 * latent**2, index=years)
        missing_before = int(matrix.isna().sum().sum())
        model = fit_native_pcr(
            matrix,
            target,
            years,
            PCAConfig(
                method="pairwise", selection="fixed", n_components=2,
                max_components=2, min_pairwise_overlap=8, min_proxies_per_year=1,
            ),
            ReconstructionConfig(
                regression="random_forest", validation_block_years=15,
                n_bootstrap=0, auto_tune=False, random_forest_trees=50,
                random_forest_min_samples_leaf=2,
            ),
        )
        self.assertGreater(model.predict(matrix).corr(target), 0.8)
        self.assertEqual(int(matrix.isna().sum().sum()), missing_before)

    def test_auto_regression_compares_linear_families_without_filling(self):
        rng = np.random.default_rng(51)
        years = np.arange(1900, 2000)
        latent = np.sin(np.arange(len(years)) / 8.0) + 0.2 * np.cos(
            np.arange(len(years)) / 3.0
        )
        matrix = pd.DataFrame(
            {
                "p1": latent + rng.normal(0, 0.12, len(years)),
                "p2": 0.8 * latent + rng.normal(0, 0.15, len(years)),
                "p3": -0.5 * latent + rng.normal(0, 0.18, len(years)),
            },
            index=years,
        )
        matrix.loc[years[::5], "p2"] = np.nan
        matrix.loc[years[2::7], "p3"] = np.nan
        missing_before = int(matrix.isna().sum().sum())
        target = pd.Series(latent + rng.normal(0, 0.08, len(years)), index=years)
        model = fit_native_pcr(
            matrix,
            target,
            years,
            PCAConfig(
                method="pairwise", selection="blocked_cv", max_components=2,
                min_pairwise_overlap=8, min_proxies_per_year=1,
            ),
            ReconstructionConfig(
                regression="auto",
                regression_candidates=("ridge", "pls", "elasticnet"),
                ridge_alphas=(0.0, 0.1),
                validation_block_years=20,
                n_bootstrap=0,
            ),
        )
        self.assertEqual(
            set(model.selection_table["regression"]),
            {"ridge", "pls", "elasticnet"},
        )
        self.assertIn(model.regression_name, {"ridge", "pls", "elasticnet"})
        self.assertNotEqual(model.regression_name, "auto")
        self.assertEqual(int(matrix.isna().sum().sum()), missing_before)

    def test_lowpass_diagnostic_does_not_change_input_or_fill_proxy_data(self):
        years = np.arange(1900, 1960)
        observed = pd.Series(np.sin(np.arange(len(years)) / 8.0), index=years)
        predicted = observed * 0.7
        original = predicted.copy()
        metrics = multiscale_reconstruction_metrics(observed, predicted, (10.0, 40.0))
        self.assertGreater(metrics["lowpass_10y_r"], 0.99)
        self.assertAlmostEqual(metrics["lowpass_10y_sd_ratio"], 0.7)
        self.assertTrue(np.isnan(metrics["lowpass_40y_r"]))
        pd.testing.assert_series_equal(predicted, original)

    def test_spectral_diagnostic_recovers_period_amplitude_and_phase(self):
        years = np.arange(1900, 2000)
        observed = pd.Series(
            np.sin(2 * np.pi * np.arange(len(years)) / 20.0),
            index=years,
        )
        predicted = observed * 0.5
        original = predicted.copy()
        metrics = spectral_reconstruction_metrics(
            observed,
            predicted,
            ((10.0, 30.0),),
        )
        self.assertAlmostEqual(metrics["spectral_10_30y_observed_peak_period"], 20.0)
        self.assertAlmostEqual(metrics["spectral_10_30y_predicted_peak_period"], 20.0)
        self.assertAlmostEqual(metrics["spectral_10_30y_amplitude_ratio"], 0.5)
        self.assertAlmostEqual(metrics["spectral_10_30y_phase_similarity"], 1.0)
        pd.testing.assert_series_equal(predicted, original)

        gapped = predicted.copy()
        gapped.loc[1930:1969] = np.nan
        unavailable = spectral_reconstruction_metrics(
            observed,
            gapped,
            ((10.0, 30.0),),
        )
        self.assertEqual(unavailable["spectral_10_30y_n"], 30.0)
        self.assertTrue(np.isnan(unavailable["spectral_10_30y_amplitude_ratio"]))

    def test_subdecadal_proxy_is_routed_to_resolution_subnest_without_interpolation(self):
        years = np.arange(1900, 1931)
        records = {
            "annual": ProxyRecord("annual", years, np.sin(years)),
            "three_year": ProxyRecord(
                "three_year", years[::3], np.sin(years[::3])
            ),
            "eleven_year": ProxyRecord(
                "eleven_year", years[::11], np.sin(years[::11])
            ),
        }
        pca = PCAConfig(min_proxies_per_year=1)
        reconstruction = ReconstructionConfig(
            multiresolution=MultiresolutionConfig(
                enabled=True,
                regression_max_resolution_years=10.0,
                minimum_calibration_overlap=4,
            )
        )
        core, low = split_resolution_roles(
            records,
            list(records),
            pca,
            reconstruction,
        )
        self.assertEqual(core, ["annual"])
        self.assertEqual(low, ["three_year", "eleven_year"])
        matrix = build_proxy_matrix(
            records,
            ["three_year"],
            ScreeningConfig(),
            interpolation="none",
        )
        self.assertEqual(int(matrix["three_year"].notna().sum()), len(years[::3]))
        self.assertTrue(np.isnan(matrix.loc[1901, "three_year"]))

    def test_three_year_proxy_participates_in_window_pca_and_regression(self):
        years = np.arange(1900, 2000)
        latent = np.sin(np.arange(len(years)) / 7.0) + 0.3 * np.cos(
            np.arange(len(years)) / 19.0
        )
        three_years = years[1::3]
        records = {
            "tree_a": ProxyRecord("tree_a", years, latent, archive="Wood"),
            "tree_b": ProxyRecord(
                "tree_b", years, 0.9 * latent + 0.05 * np.cos(years),
                archive="Wood",
            ),
            "stalagmite": ProxyRecord(
                "stalagmite",
                three_years,
                pd.Series(latent, index=years).reindex(three_years).to_numpy(),
                archive="Speleothem",
            ),
        }
        annual_matrix = pd.DataFrame(
            {"tree_a": records["tree_a"].value, "tree_b": records["tree_b"].value},
            index=pd.Index(years, name="Year"),
        )
        target = pd.Series(latent, index=years, name="target")
        config = ReconstructionConfig(
            auto_tune=False,
            n_bootstrap=0,
            full_network_outer_validation=False,
            rescreen_outer_folds=False,
            nest=ExplicitNestConfig(
                minimum_subnest_calibration_windows=8
            ),
        )
        models, audit = fit_resolution_subnests(
            annual_matrix,
            records,
            ["stalagmite"],
            target,
            years,
            int(years.min()),
            int(years.max()),
            PCAConfig(
                selection="fixed", n_components=1, max_components=1,
                min_proxies_per_year=2,
            ),
            config,
        )
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].resolution_years, 3)
        self.assertIn("stalagmite", models[0].model.columns)
        self.assertIn("tree_a", models[0].model.columns)
        self.assertTrue(bool(audit.iloc[0]["accepted"]))

    def test_resolution_subnest_prestandardizes_native_values_without_interpolation(self):
        years = np.arange(1900, 1910)
        record = ProxyRecord(
            "stalagmite",
            [1900, 1903, 1906, 1909],
            [10.0, 13.0, 16.0, 19.0],
            archive="Speleothem",
        )
        matrix, _, _ = build_resolution_matrix(
            pd.DataFrame({"tree": np.zeros(len(years))}, index=years),
            {
                "tree": ProxyRecord("tree", years, np.zeros(len(years)), archive="Wood"),
                "stalagmite": record,
            },
            ["stalagmite"],
            pd.Series(np.zeros(len(years)), index=years),
            1900,
            1909,
            3,
            standardization_period=(1900, 1906),
            standardization_years=years,
        )
        # The three observed reference values have mean 13 and sample SD 3;
        # no annual values are manufactured between the four native samples.
        np.testing.assert_allclose(
            matrix["stalagmite"].to_numpy(), [-1.0, 0.0, 1.0, 2.0]
        )

    def test_native_sample_on_window_boundary_is_used_once(self):
        record = ProxyRecord("native", [1902.5], [7.0], archive="Speleothem")
        matrix, _, _ = build_resolution_matrix(
            pd.DataFrame(index=np.arange(1900, 1906)),
            {"native": record},
            ["native"],
            pd.Series(0.0, index=np.arange(1900, 1906)),
            1900, 1905, 3,
        )
        self.assertEqual(int(matrix["native"].notna().sum()), 1)
        self.assertEqual(matrix.loc[1903, "native"], 7.0)

    def test_window_grouping_matches_observed_means_with_gaps_and_partial_end(self):
        years = np.arange(1900, 1934)
        rng = np.random.default_rng(70)
        annual = pd.DataFrame({"tree": rng.normal(size=len(years))}, index=years)
        annual.loc[1905:1907, "tree"] = np.nan
        times = np.array([1900.0, 1904.5, 1910.0, 1915.2, 1920.0, 1924.5, 1933.0])
        native = ProxyRecord("native", times, np.arange(len(times), dtype=float), archive="Speleothem")
        records = {"tree": ProxyRecord("tree", years, annual["tree"].to_numpy()), "native": native}
        target = pd.Series(rng.normal(size=len(years)), index=years)
        target.loc[1915:1918] = np.nan
        matrix, window_target, windows = build_resolution_matrix(
            annual, records, ["native"], target, 1900, 1933, 5,
        )
        for row in windows.itertuples():
            expected = annual.loc[row.window_start:row.window_end, "tree"].mean()
            np.testing.assert_allclose(matrix.loc[row.Index, "tree"], expected, equal_nan=True)
            np.testing.assert_allclose(
                window_target.loc[row.Index], target.loc[row.window_start:row.window_end].mean(),
                equal_nan=True,
            )
            native_in_window = (
                (times >= row.window_start - 0.5) & (times < row.window_end + 0.5)
            )
            expected_native = native.value[native_in_window].mean() if native_in_window.any() else np.nan
            np.testing.assert_allclose(matrix.loc[row.Index, "native"], expected_native, equal_nan=True)

    def test_resolution_layer_requires_native_proxy_in_fitted_pca(self):
        years = np.arange(1900, 1990)
        latent = np.sin((years - 1900) / 7.0)
        records = {
            "tree_a": ProxyRecord("tree_a", years, latent, archive="Wood"),
            "tree_b": ProxyRecord("tree_b", years, latent + 0.1 * np.cos(years), archive="Wood"),
            "native": ProxyRecord("native", years[::3], np.ones(30), archive="Speleothem"),
        }
        models, audit = fit_resolution_subnests(
            pd.DataFrame({pid: records[pid].value for pid in ("tree_a", "tree_b")}, index=years),
            records,
            ["native"],
            pd.Series(latent, index=years),
            years,
            1900, 1989,
            PCAConfig(selection="fixed", n_components=1, max_components=1),
            ReconstructionConfig(auto_tune=False, n_bootstrap=0),
        )
        self.assertEqual(models, [])
        self.assertEqual(audit.iloc[0]["reason"], "no_native_proxy_in_fitted_pca")

    def test_explicit_nest_end_to_end_records_resolution_subnest(self):
        years = np.arange(1900, 2000)
        latent = np.sin((years - 1900) / 8.0)
        three_years = years[1::3]
        records = ProxyCollection([
            ProxyRecord("tree_a", years, latent, archive="Wood"),
            ProxyRecord(
                "tree_b", years, latent + 0.04 * np.cos(years), archive="Wood"
            ),
            ProxyRecord(
                "stalagmite",
                three_years,
                pd.Series(latent, index=years).reindex(three_years).to_numpy(),
                archive="Speleothem",
            ),
        ])
        screening = pd.DataFrame({
            "pid": list(records),
            "selected": [True, True, True],
            "best_months": ["1,2,3,4,5,6,7,8,9,10,11,12"] * 3,
        })
        result = reconstruct(
            records,
            screening,
            pd.Series(latent, index=years, name="target"),
            ScreeningConfig(),
            PCAConfig(
                selection="fixed", n_components=1, max_components=1,
                min_proxies_per_year=2,
            ),
            ReconstructionConfig(
                method="explicit_nest",
                auto_tune=True,
                regression="ridge",
                ridge_alphas=(0.1,),
                n_bootstrap=2,
                minimum_bootstrap_success_fraction=0.5,
                full_network_outer_validation=False,
                rescreen_outer_folds=False,
                nest=ExplicitNestConfig(
                    minimum_subnest_calibration_windows=8,
                    require_internal_ce_re=False,
                ),
            ),
        )
        rows = result.nest_summary
        self.assertTrue((rows["row_type"] == "coverage_nest").any())
        resolution = rows.loc[rows["row_type"] == "resolution_subnest"]
        self.assertTrue((resolution["resolution_years"] == 3).any())
        self.assertTrue(bool(resolution.loc[
            resolution["resolution_years"] == 3, "accepted"
        ].iloc[0]))
        self.assertIn("resolution_subnest", set(result.model_selection["nest_layer"]))
        self.assertEqual(result.reconstruction["Year"].diff().dropna().unique().tolist(), [1])
        self.assertGreater(int(result.reconstruction["ensemble_n"].max()), 0)

    def test_explicit_mixed_resolution_directed_holdouts_and_common_reference(self):
        years = np.arange(1900, 2001)
        latent = np.sin((years - 1900) / 7.0)
        records = {
            "tree_a": ProxyRecord("tree_a", years, latent, archive="Wood"),
            "tree_b": ProxyRecord("tree_b", years, latent + 0.02 * np.cos(years), archive="Wood"),
            "native": ProxyRecord("native", years[::3], latent[::3], archive="Speleothem"),
        }
        screening = pd.DataFrame({"pid": list(records), "selected": True})
        config = ReconstructionConfig(
            standardization_period=(1950, 2000), n_bootstrap=0,
            ridge_alphas=(0.1,), validation_block_years=20,
            nest=ExplicitNestConfig(require_internal_ce_re=False),
        )
        for rescreen in (False, True):
            table = explicit_nest_validation(
                records, screening, pd.Series(latent, index=years),
                ScreeningConfig(min_overlap=8),
                PCAConfig(selection="fixed", n_components=1, max_components=1, min_pairwise_overlap=4),
                config, rescreen=rescreen,
            )
            self.assertEqual(len(table), 2)
            self.assertEqual(set(table["fit_status"]), {"completed"})
            self.assertEqual(set(table["validation_start"]), {1900, 1967})
            self.assertEqual(set(table["validation_end"]), {1933, 2000})
            self.assertTrue((table["r"] > 0.8).all())
            self.assertEqual(set(table["assessment_metric"]), {"correlation"})
            self.assertEqual(set(table["validation_role"]), {"sensitivity_only"})

    def test_unavailable_explicit_holdouts_are_reported(self):
        years = np.arange(1900, 2001)
        table = explicit_nest_validation(
            {}, pd.DataFrame(columns=["pid", "selected"]),
            pd.Series(np.sin(years), index=years),
            ScreeningConfig(), PCAConfig(), ReconstructionConfig(n_bootstrap=0),
            rescreen=False,
        )
        self.assertEqual(len(table), 2)
        self.assertEqual(set(table["fit_status"]), {"unavailable"})
        self.assertTrue(table["r"].isna().all())
        self.assertTrue(table["fit_error"].notna().all())

    def test_unsupported_interpolation_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "interpolation must be"):
            ReconstructionConfig(interpolation="linear")  # type: ignore[arg-type]

    def test_archive_interpolation_fills_only_short_internal_wood_coral_gaps(self):
        years = [1900, 1901, 1904, 1905, 1910, 1911]
        records = {
            "wood": ProxyRecord(
                "wood", years, [0, 1, 4, 5, 10, 11], archive="Wood"
            ),
            "ice": ProxyRecord(
                "ice", [1899, *years, 1912], [-1, 0, 1, 4, 5, 10, 11, 12],
                archive="GlacierIce",
            ),
        }
        matrix = build_proxy_matrix(
            records,
            ["wood", "ice"],
            ScreeningConfig(),
            interpolation="archive_linear",
            interpolation_archives=("Wood", "Coral"),
            interpolation_max_gap_years=2,
            interpolation_max_resolution_years=2.0,
        )
        self.assertAlmostEqual(float(matrix.loc[1902, "wood"]), 2.0)
        self.assertAlmostEqual(float(matrix.loc[1903, "wood"]), 3.0)
        self.assertTrue(np.isnan(matrix.loc[1906, "wood"]))
        self.assertTrue(np.isnan(matrix.loc[1902, "ice"]))
        self.assertTrue(np.isnan(matrix.loc[1899, "wood"]))
        self.assertTrue(np.isnan(matrix.loc[1912, "wood"]))
        audit = matrix.attrs["interpolation_audit"].set_index("pid")
        self.assertEqual(int(audit.loc["wood", "interpolated_year_count"]), 2)
        self.assertEqual(int(audit.loc["ice", "interpolated_year_count"]), 0)

    def test_longest_contiguous_annual_period_prefers_earliest_tie(self):
        values = pd.Series(
            [1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0],
            index=[1900, 1901, 1902, 1903, 1904, 1905, 1906],
        )
        self.assertEqual(_longest_contiguous_annual_period(values), (1900, 1901))

    def test_standard_re_and_ce_denominators(self):
        observed = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
        predicted = pd.Series([1.0, 2.0, 2.0], index=[1, 2, 3])
        metrics = reconstruction_metrics(observed, predicted, calibration_mean=0.0)
        self.assertAlmostEqual(metrics["ce"], 0.5)
        self.assertAlmostEqual(metrics["re"], 1 - 1 / 14)

    def test_sensitivity_summary_reports_skill_range(self):
        table = pd.DataFrame({"median_r": [0.2, 0.5, 0.8], "median_ce": [-0.1, 0.1, 0.3]})
        summary = summarize_sensitivity(table).set_index("metric")
        self.assertEqual(summary.loc["median_r", "n"], 3)
        self.assertAlmostEqual(summary.loc["median_r", "median"], 0.5)
        self.assertAlmostEqual(summary.loc["median_ce", "minimum"], -0.1)

    def test_low_frequency_operator_uses_native_support_without_interpolation(self):
        years = pd.Index(np.arange(1900, 1951), name="Year")
        target = pd.Series(np.linspace(-1, 1, len(years)), index=years)
        times = np.arange(1902.5, 1950, 5.0)
        values = np.interp(times, years, target)
        low_resolution = ProxyRecord("five_year", times, values)
        core = pd.Series(0.2 * np.sin(np.arange(len(years)) * 2.0), index=years)
        result = variational_low_frequency_adjustment(
            core,
            [low_resolution],
            target,
            config=MultiresolutionConfig(
                enabled=True,
                lowpass_period_years=10.0,
                minimum_calibration_overlap=5,
                maximum_increment_ratio=None,
            ),
        )
        self.assertEqual(low_resolution.time.size, len(times))
        self.assertEqual(result.constraints.loc[0, "status"], "used")
        self.assertEqual(len(result.observations), len(times))
        self.assertTrue((result.observations["annual_state_count"] >= 4).all())
        self.assertEqual(result.adjusted.index.to_list(), years.to_list())
        np.testing.assert_allclose(
            result.adjusted - result.adjusted_low_frequency,
            result.core_high_frequency,
            atol=1e-10,
        )
        self.assertGreater(result.adjusted.corr(target), core.corr(target))

    def test_native_support_does_not_bridge_a_long_hiatus(self):
        record = ProxyRecord("irregular", [0, 3, 100, 103], [0, 1, 2, 3])
        uncapped = _native_support_widths(record)
        capped = _native_support_widths(record, maximum_multiplier=3.0)
        self.assertGreater(float(uncapped.max()), 40.0)
        self.assertLessEqual(float(capped.max()), 9.0)

    def test_low_frequency_anchor_uses_calibration_scale_when_core_lacks_cycle(self):
        years = np.arange(1800, 2001)
        step = np.arange(len(years), dtype=float)
        low = np.sin(2 * np.pi * step / 25.0)
        high = 0.35 * np.sin(2 * np.pi * step / 4.0 + 0.2)
        target = pd.Series(low + high, index=years)
        core = pd.Series(high, index=years)
        records = []
        for number in range(3):
            times = np.arange(1800 + (number + 0.5) * 4.0, 2001, 12.0)
            values = [
                float(np.mean(low[(years >= time - 6.0) & (years < time + 6.0)]))
                for time in times
            ]
            records.append(ProxyRecord(f"low_{number}", times, values))
        result = variational_low_frequency_adjustment(
            core,
            records,
            target.loc[1800:1899],
            config=MultiresolutionConfig(
                enabled=True,
                proxy_constraint_weight=30.0,
                auto_tune=False,
                minimum_calibration_overlap=4,
                maximum_increment_ratio=1.0,
            ),
        )
        audit = result.constraints.iloc[0]
        self.assertGreater(
            audit["calibration_target_low_frequency_scale"],
            10 * audit["core_low_frequency_scale"],
        )
        self.assertAlmostEqual(
            audit["increment_reference_scale"],
            audit["calibration_target_low_frequency_scale"],
        )
        validation = target.loc[1900:2000]
        self.assertGreater(
            result.adjusted.loc[1900:2000].corr(validation),
            core.loc[1900:2000].corr(validation) + 0.5,
        )

    def test_multiresolution_tuner_can_select_zero_or_nonzero_without_interpolation(self):
        years = np.arange(1900, 2001)
        step = np.arange(len(years))
        high = np.sin(2 * np.pi * step / 4.0)
        low = 1.2 * np.sin(2 * np.pi * step / 35.0)
        rng = np.random.default_rng(1)
        pca = PCAConfig(
            selection="fixed", n_components=1, max_components=1,
            min_proxies_per_year=1, min_pairwise_overlap=8,
        )
        reconstruction = ReconstructionConfig(
            regression="ridge", ridge_alphas=(0.1,), auto_tune=False,
            n_bootstrap=0, validation_block_years=25,
            multiresolution=MultiresolutionConfig(
                enabled=True,
                auto_tune=True,
                proxy_constraint_weight_candidates=(0.0, 3.0),
                lowpass_period_candidates=(10.0,),
                minimum_calibration_overlap=8,
                maximum_increment_ratio=None,
            ),
        )
        matrix = pd.DataFrame(
            {
                "p1": high + rng.normal(0, 0.1, len(years)),
                "p2": high + rng.normal(0, 0.1, len(years)),
            },
            index=years,
        )
        matrix.loc[years[1::7], "p2"] = np.nan
        missing_before = int(matrix.isna().sum().sum())
        target = pd.Series(high + low, index=years)
        low_times = years[::5]
        useful = ProxyRecord("useful", low_times, target.reindex(low_times).to_numpy())
        selected, table = tune_multiresolution_config(
            matrix, target, years, pca, reconstruction, [useful],
            n_components=1, alpha=0.1, regression_name="ridge",
        )
        self.assertEqual(selected.proxy_constraint_weight, 3.0)
        self.assertEqual(int(table["selected"].sum()), 1)
        self.assertEqual(int(matrix.isna().sum().sum()), missing_before)

        perfect_target = pd.Series(high, index=years)
        perfect_matrix = pd.DataFrame({"p1": high, "p2": high * 1.01}, index=years)
        noise = ProxyRecord("noise", low_times, rng.normal(size=len(low_times)))
        rejected, _ = tune_multiresolution_config(
            perfect_matrix, perfect_target, years, pca, reconstruction, [noise],
            n_components=1, alpha=0.1, regression_name="ridge",
        )
        self.assertEqual(rejected.proxy_constraint_weight, 0.0)

    def test_amplitude_calibration_is_explicit_and_affine(self):
        years = np.arange(1900, 1950)
        predicted = pd.Series(np.linspace(-1, 1, len(years)), index=years)
        observed = 0.4 + 2.0 * predicted
        calibrator = fit_amplitude_calibrator(
            observed,
            predicted,
            AmplitudeCalibrationConfig(method="ols", minimum_overlap=20),
        )
        self.assertAlmostEqual(calibrator.slope, 2.0)
        self.assertAlmostEqual(calibrator.intercept, 0.4)
        np.testing.assert_allclose(calibrator.apply(predicted), observed)

    def test_amplitude_calibration_accepts_bootstrap_duplicate_years(self):
        years = pd.Index([1900, 1901, 1900, 1901, 1902, 1902])
        predicted = pd.Series([-1.0, 0.0, -1.0, 0.0, 1.0, 1.0], index=years)
        observed = 0.5 + 1.5 * predicted
        calibrator = fit_amplitude_calibrator(
            observed,
            predicted,
            AmplitudeCalibrationConfig(method="ols", minimum_overlap=4),
        )
        self.assertAlmostEqual(calibrator.slope, 1.5)
        self.assertAlmostEqual(calibrator.intercept, 0.5)

    def test_variance_calibration_can_use_densest_proxy_network(self):
        years = np.arange(1900, 1910)
        predicted = pd.Series(np.arange(10, dtype=float), index=years)
        observed = pd.Series(np.arange(10, dtype=float) * 50.0, index=years)
        availability = pd.Series([3] * 6 + [2] * 4, index=years)
        calibrator = fit_amplitude_calibrator(
            observed,
            predicted,
            AmplitudeCalibrationConfig(
                method="variance",
                variance_reference="max_proxy_nest",
                minimum_overlap=4,
                slope_bounds=(0.01, 10.0),
            ),
            availability=availability,
        )
        adjusted = calibrator.apply(predicted)
        dense_reference = predicted.loc[availability >= 3]
        self.assertEqual(calibrator.reference, "max_proxy_nest")
        self.assertEqual(calibrator.reference_overlap, 6)
        self.assertEqual(calibrator.reference_min_proxy_count, 3)
        self.assertAlmostEqual(adjusted.mean(), dense_reference.mean())
        self.assertAlmostEqual(adjusted.std(ddof=1), dense_reference.std(ddof=1))

    def test_variance_calibration_observation_reference_is_explicit(self):
        years = np.arange(1900, 1910)
        predicted = pd.Series(np.arange(10, dtype=float), index=years)
        observed = 2.0 + predicted * 3.0
        calibrator = fit_amplitude_calibrator(
            observed,
            predicted,
            AmplitudeCalibrationConfig(
                method="variance",
                variance_reference="observation",
                minimum_overlap=4,
            ),
        )
        self.assertEqual(calibrator.reference, "observation")
        self.assertAlmostEqual(calibrator.apply(predicted).std(), observed.std())

    def test_dynamic_variance_uses_effective_network_size_without_interpolation(self):
        rng = np.random.default_rng(19)
        years = np.arange(1900, 1980)
        predicted = pd.Series(rng.normal(size=len(years)), index=years)
        effective = pd.Series([1.0] * 40 + [4.0] * 40, index=years)
        observed = pd.Series(
            np.where(effective.to_numpy() == 1.0, 2.0, 1.0)
            * predicted.to_numpy(),
            index=years,
        )
        calibrator = fit_amplitude_calibrator(
            observed,
            predicted,
            AmplitudeCalibrationConfig(
                method="dynamic_variance",
                minimum_overlap=20,
                dynamic_max_bins=2,
                dynamic_minimum_bin_years=20,
                dynamic_shrinkage_years=0.0,
                slope_bounds=(0.1, 4.0),
            ),
            effective_availability=effective,
        )
        adjusted = calibrator.apply(predicted, effective)
        self.assertEqual(calibrator.method, "dynamic_variance")
        self.assertEqual(len(calibrator.slope_knots), 2)
        self.assertGreater(calibrator.slope_knots[0], calibrator.slope_knots[1])
        self.assertAlmostEqual(adjusted.iloc[:40].std(), observed.iloc[:40].std())
        self.assertAlmostEqual(adjusted.iloc[40:].std(), observed.iloc[40:].std())

    def test_automatic_amplitude_choice_is_inner_fold_candidate(self):
        rng = np.random.default_rng(21)
        years = np.arange(1900, 1990)
        target = pd.Series(np.sin(np.arange(len(years)) / 7.0), index=years)
        matrix = pd.DataFrame(
            {
                "p1": target.to_numpy() + rng.normal(0, 0.15, len(years)),
                "p2": target.to_numpy() + rng.normal(0, 0.20, len(years)),
            },
            index=years,
        )
        model = fit_native_pcr(
            matrix,
            target,
            years,
            PCAConfig(selection="fixed", n_components=1, max_components=1),
            ReconstructionConfig(
                regression="ridge",
                ridge_alphas=(10.0,),
                validation_block_years=20,
                n_bootstrap=0,
                amplitude=AmplitudeCalibrationConfig(
                    method="auto",
                    minimum_overlap=15,
                    dynamic_minimum_bin_years=10,
                ),
            ),
            forced_n_components=1,
            forced_alpha=10.0,
            forced_regression="ridge",
        )
        self.assertNotEqual(model.amplitude_calibrator.method, "auto")
        self.assertEqual(
            set(model.selection_table["amplitude_candidate"]),
            {
                "none",
                "variance_observation",
                "variance_max_proxy_nest",
                "dynamic_variance",
            },
        )

    def test_kaiser_and_kaiser_cv_component_rules(self):
        eigenvalues = np.array([3.0, 1.2, 0.9, 0.4])
        self.assertEqual(
            _component_candidates(
                eigenvalues,
                PCAConfig(selection="kaiser", max_components=4),
            ),
            [2],
        )
        self.assertEqual(
            _component_candidates(
                eigenvalues,
                PCAConfig(selection="kaiser_cv", max_components=4),
            ),
            [1, 2],
        )
        self.assertEqual(
            _component_candidates(
                eigenvalues,
                PCAConfig(
                    selection="kaiser", kaiser_threshold=0.8, max_components=4
                ),
            ),
            [3],
        )

    def test_amplitude_auto_cannot_change_structural_model(self):
        rng = np.random.default_rng(29)
        years = np.arange(1900, 1990)
        slow = np.sin(np.arange(len(years)) / 8.0)
        fast = 0.45 * np.cos(np.arange(len(years)) / 2.5)
        target = pd.Series(slow + fast, index=years)
        matrix = pd.DataFrame(
            {
                "p1": slow + rng.normal(0, 0.15, len(years)),
                "p2": fast + rng.normal(0, 0.15, len(years)),
                "p3": target.to_numpy() + rng.normal(0, 0.30, len(years)),
            },
            index=years,
        )
        common = dict(
            regression="ridge",
            ridge_alphas=(0.0, 10.0, 100.0),
            validation_block_years=20,
            n_bootstrap=0,
        )
        pca = PCAConfig(selection="blocked_cv", max_components=3)
        baseline = fit_native_pcr(
            matrix,
            target,
            years,
            pca,
            ReconstructionConfig(
                **common,
                amplitude=AmplitudeCalibrationConfig(method="none"),
            ),
        )
        automatic = fit_native_pcr(
            matrix,
            target,
            years,
            pca,
            ReconstructionConfig(
                **common,
                amplitude=AmplitudeCalibrationConfig(
                    method="auto", dynamic_minimum_bin_years=10
                ),
            ),
        )
        self.assertEqual(automatic.n_components, baseline.n_components)
        self.assertEqual(automatic.regression_name, baseline.regression_name)
        self.assertEqual(automatic.alpha, baseline.alpha)
        structure = automatic.selection_table.query("selection_stage == 'structure'")
        self.assertTrue((structure["amplitude_method"] == "none").all())
        self.assertEqual(int(structure["selected"].sum()), 1)
        amplitude = automatic.selection_table.query("selection_stage == 'amplitude'")
        self.assertEqual(int(amplitude["selected"].sum()), 1)
        np.testing.assert_array_equal(
            automatic.selection_table["passes_skill_floor"].to_numpy(bool),
            (
                (automatic.selection_table["median_ce"] >= 0.0)
                & (automatic.selection_table["median_re"] >= 0.0)
            ).to_numpy(bool),
        )

    def test_proxy_weights_downweight_correlated_neighbors_and_report_error(self):
        rng = np.random.default_rng(23)
        years = np.arange(1900, 1950)
        shared = rng.normal(size=len(years))
        matrix = pd.DataFrame(
            {
                "p1": shared,
                "p2": shared + rng.normal(0, 0.01, len(years)),
                "p3": rng.normal(size=len(years)),
            },
            index=years,
        )
        matrix.attrs["proxy_metadata"] = {
            "p1": {"lat": 10.0, "lon": 100.0, "archive": "Wood", "proxy": "d18O", "site": "A", "source": "S", "metadata": {}},
            "p2": {"lat": 10.2, "lon": 100.1, "archive": "Wood", "proxy": "d18O", "site": "B", "source": "T", "metadata": {}},
            "p3": {"lat": 40.0, "lon": 150.0, "archive": "Wood", "proxy": "d18O", "site": "C", "source": "U", "metadata": {}},
        }
        table = compute_proxy_weights(
            matrix,
            years,
            ProxyWeightConfig(
                enabled=True,
                explicit_error_sd=(("p3", 2.0),),
                redundancy_radius_km=250.0,
                redundancy_correlation_threshold=0.9,
                redundancy_min_overlap=20,
            ),
        ).set_index("pid")
        self.assertEqual(int(table.loc["p1", "redundancy_group_size"]), 2)
        self.assertEqual(int(table.loc["p2", "redundancy_group_size"]), 2)
        self.assertAlmostEqual(float(table.loc["p1", "redundancy_weight"]), 0.5)
        self.assertLess(float(table.loc["p3", "error_weight"]), 1.0)
        self.assertLess(float(table.loc["p3", "weight"]), 1.0)

    def test_all_network_sensitivity_modes_return_compact_results(self):
        rng = np.random.default_rng(7)
        years = np.arange(1900, 1960)
        target = pd.Series(np.sin(np.arange(len(years)) / 6.0), index=years)
        records = ProxyCollection(
            [
                ProxyRecord(
                    f"p{number}", years,
                    target.to_numpy() + rng.normal(0, 0.1 + number * 0.01, len(years)),
                )
                for number in range(4)
            ]
        )
        screening_config = ScreeningConfig(min_overlap=15, p_threshold=1.0, r_threshold=0.2)
        screening = screen_proxies(records, target, screening_config)
        results = run_network_sensitivities(
            records,
            screening,
            target,
            screening_config,
            PCAConfig(selection="fixed", n_components=1, max_components=1),
            ReconstructionConfig(
                regression="ridge", ridge_alphas=(0.1,), validation_block_years=15,
                n_bootstrap=0, auto_tune=False,
            ),
            SensitivityConfig(
                single_proxy=True,
                leave_one_proxy_out=True,
                split_periods=(((1900, 1929), (1930, 1959)),),
                random_delete_fraction=0.25,
                random_delete_repeats=2,
            ),
        )
        self.assertEqual(len(results["single_proxy"]), 4)
        self.assertEqual(len(results["leave_one_proxy_out"]), 4)
        self.assertEqual(len(results["random_proxy_deletion"]), 2)
        self.assertEqual(len(results["split_period"]), 1)
        self.assertFalse(results["random_proxy_deletion_summary"].empty)

    def test_low_frequency_layer_is_in_outer_validation(self):
        years = np.arange(1900, 1980)
        target = pd.Series(np.sin(np.arange(len(years)) / 10.0), index=years)
        matrix = pd.DataFrame(
            {"annual": target.to_numpy() + 0.2 * np.cos(np.arange(len(years)))},
            index=pd.Index(years, name="Year"),
        )
        low = ProxyRecord(
            "low", np.arange(1902.5, 1980, 5.0),
            np.interp(np.arange(1902.5, 1980, 5.0), years, target),
        )
        folds = blocked_validation(
            matrix,
            target,
            PCAConfig(selection="fixed", n_components=1, max_components=1, min_proxies_per_year=1),
            ReconstructionConfig(
                regression="ridge", ridge_alphas=(0.1,), validation_block_years=20,
                n_bootstrap=0, auto_tune=False,
                multiresolution=MultiresolutionConfig(
                    enabled=True,
                    minimum_calibration_overlap=5,
                ),
            ),
            low_resolution_records=[low],
        )
        self.assertFalse(folds.empty)
        self.assertIn("core_ce", folds)
        self.assertIn("raw_ce", folds)
        self.assertIn("pca_selection", folds)
        self.assertIn("kaiser_component_count", folds)
        self.assertTrue((folds["low_frequency_proxy_count"] >= 0).all())
        self.assertEqual(folds.iloc[0]["holdout_position"], "early")
        self.assertEqual(folds.iloc[-1]["holdout_position"], "late")

    def test_outer_edge_matches_explicit_split_over_full_reconstruction_domain(self):
        years = np.arange(1900, 1960)
        signal = np.sin((years - 1900) / 8.0)
        target = pd.Series(signal, index=years)
        low_years = np.arange(1880, 1980, 5)
        records = ProxyCollection(
            [
                ProxyRecord("annual", years, signal + 0.05 * np.cos(years)),
                ProxyRecord("low", low_years, np.sin((low_years - 1900) / 8.0)),
            ]
        )
        screening_config = ScreeningConfig(
            min_overlap=10,
            low_resolution_min_overlap=8,
            low_resolution_cutoff_years=5.0,
            p_threshold=1.0,
            r_threshold=0.0,
        )
        pca_config = PCAConfig(
            selection="fixed", n_components=1, max_components=1,
            min_proxies_per_year=1,
        )
        reconstruction_config = ReconstructionConfig(
            method="native_missing",
            calibration_period=(1900, 1959),
            reconstruction_period=(1880, 1979),
            regression="ridge",
            ridge_alphas=(0.1,),
            validation_block_years=20,
            n_bootstrap=0,
            auto_tune=False,
            multiresolution=MultiresolutionConfig(
                enabled=True,
                regression_max_resolution_years=4.0,
                minimum_calibration_overlap=5,
            ),
        )
        outer = blocked_pipeline_validation(
            records, target, screening_config, pca_config, reconstruction_config
        )
        screening = screen_proxies(records, target, screening_config)
        explicit = run_network_sensitivities(
            records,
            screening,
            target,
            screening_config,
            pca_config,
            reconstruction_config,
            SensitivityConfig(split_periods=(((1920, 1959), (1900, 1919)),)),
        )["split_period"].iloc[0]
        early = outer.loc[outer["holdout_position"] == "early"].iloc[0]
        for metric in ("r", "rmse", "re", "ce"):
            self.assertAlmostEqual(float(early[metric]), float(explicit[metric]), places=10)


if __name__ == "__main__":
    unittest.main()

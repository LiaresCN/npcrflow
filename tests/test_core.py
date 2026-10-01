from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from npcrflow.config import (
    AmplitudeCalibrationConfig,
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
from npcrflow.model import _component_candidates, fit_native_pcr
from npcrflow.pipeline import filter_records_with_report, run_pipeline
from npcrflow.records import ProxyCollection, ProxyRecord
from npcrflow.low_frequency import _native_support_widths, variational_low_frequency_adjustment
from npcrflow.reconstruction import blocked_pipeline_validation, blocked_validation
from npcrflow.reconstruction import (
    build_proxy_matrix,
    split_resolution_roles,
    tune_multiresolution_config,
)
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

    def test_subdecadal_proxy_enters_regression_without_interpolation(self):
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
        self.assertEqual(core, ["annual", "three_year"])
        self.assertEqual(low, ["eleven_year"])
        matrix = build_proxy_matrix(
            records,
            core,
            ScreeningConfig(),
            interpolation="none",
        )
        self.assertEqual(int(matrix["three_year"].notna().sum()), len(years[::3]))
        self.assertTrue(np.isnan(matrix.loc[1901, "three_year"]))

    def test_any_interpolation_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "interpolation is disabled"):
            ReconstructionConfig(interpolation="linear")  # type: ignore[arg-type]

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

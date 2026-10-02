from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from npcrflow.amplitude import fit_amplitude_calibrator
from npcrflow.config import AmplitudeCalibrationConfig, PCAConfig, ReconstructionConfig, ScreeningConfig
from npcrflow.model import fit_native_pcr
from npcrflow.nesting import pool_nest_ensembles
from npcrflow.random_ensemble import _adjust_predictions, random_holdout_indices
from npcrflow.records import ProxyRecord
from npcrflow.resolution_nest import (ResolutionNestModel, build_resolution_matrix,
                                      fuse_resolution_ensemble, fuse_resolution_subnests)
from npcrflow.screening import (_native_screening_pair, correlation_with_effective_dof,
                                screen_proxies)
from npcrflow.validation import reconstruction_metrics


class NativeScreeningTests(unittest.TestCase):
    def test_pair_actual_native_values_with_complete_target_window_means(self):
        years = np.arange(1900, 1949)
        target = pd.Series(np.arange(len(years), dtype=float), index=years)
        record = ProxyRecord("stone", [1901.2, 1904.2, 1910.2, 1948.2], [4, 5, 9, 100])
        config = ScreeningConfig(period=(1900, 1948), native_window_bins=(3, 5, 10))
        # This irregular record's median spacing is 6 years -> 10-year windows.
        proxy, observed, width = _native_screening_pair(record, target, config)
        self.assertEqual(width, 10)
        self.assertAlmostEqual(proxy.loc[1900], 4.5)
        self.assertAlmostEqual(observed.loc[1900], target.loc[1900:1909].mean())
        self.assertTrue(np.isnan(proxy.loc[1920]))
        self.assertTrue(np.isnan(proxy.loc[1940]))  # terminal partial window
        self.assertEqual(proxy.notna().sum(), 2)
        np.testing.assert_array_equal(record.value, [4, 5, 9, 100])

    def test_three_year_window_alignment_matches_resolution_matrix(self):
        years = np.arange(1900, 1960)
        target = pd.Series(np.sin(np.arange(60) / 5), index=years)
        times = np.arange(1901., 1960, 3) + .2
        record = ProxyRecord("stone", times, np.arange(len(times)))
        proxy, observed, width = _native_screening_pair(record, target, ScreeningConfig(period=(1900, 1959)))
        matrix, window_target, _ = build_resolution_matrix(
            target.to_frame("tree"), {"stone": record, "tree": ProxyRecord("tree", years, target)},
            ["stone"], target, 1900, 1959, 3)
        self.assertEqual(width, 3)
        np.testing.assert_allclose(proxy, matrix.stone, equal_nan=True)
        np.testing.assert_allclose(observed, window_target, equal_nan=True)

    def test_holdout_or_missing_target_year_invalidates_whole_window(self):
        years = np.arange(1900, 1960)
        target = pd.Series(np.arange(60), index=years).drop([1904, 1905])
        record = ProxyRecord("stone", np.arange(1901, 1960, 3), np.arange(20))
        proxy, observed, _ = _native_screening_pair(record, target, ScreeningConfig(period=(1900, 1959)))
        self.assertTrue(np.isnan(proxy.loc[1903]))
        self.assertTrue(np.isnan(observed.loc[1903]))
        changed = replace(record, value=record.value + np.where(record.time == 1904, 1e9, 0))
        proxy2, _, _ = _native_screening_pair(changed, target, ScreeningConfig(period=(1900, 1959)))
        pd.testing.assert_series_equal(proxy, proxy2)

    def test_effective_dof_uses_native_step_and_does_not_bridge_hiatus(self):
        index = np.arange(1900, 2020, 3)
        x = pd.Series(np.sin(np.arange(40) / 6), index=index)
        y = pd.Series(np.sin(np.arange(40) / 6) + .2 * np.cos(np.arange(40)), index=index)
        r1, p1, n1, _ = correlation_with_effective_dof(x, y)
        r3, p3, n3, n = correlation_with_effective_dof(x, y, year_step=3)
        self.assertEqual(r1, r3)
        self.assertEqual(n1, n)
        self.assertLess(n3, n)
        self.assertGreater(p3, p1)
        gaps = x.iloc[::2]
        _, _, effective, paired = correlation_with_effective_dof(gaps, y, year_step=3)
        self.assertEqual(effective, paired)

    def test_annual_screening_is_unchanged_and_low_resolution_is_auditable(self):
        years = np.arange(1900, 1960)
        target = pd.Series(np.sin(np.arange(60) / 6), index=years)
        annual = ProxyRecord("tree", years, target.to_numpy())
        stone = ProxyRecord("stone", years[1::3], target.iloc[1::3].to_numpy())
        config = ScreeningConfig(period=(1900, 1959), low_resolution_min_overlap=8,
                                 low_resolution_cutoff_years=1.5, p_threshold=1,
                                 proxy_multiple_testing="none")
        frame = screen_proxies({"tree": annual, "stone": stone}, target, config).set_index("pid")
        old = screen_proxies({"tree": annual}, target, replace(config, low_resolution_pairing="annual")).iloc[0]
        self.assertEqual(frame.loc["tree", "r"], old.r)
        self.assertEqual(frame.loc["tree", "p_effective_raw"], old.p_effective_raw)
        self.assertEqual(frame.loc["stone", "screening_window_years"], 3)
        self.assertEqual(frame.loc["stone", "n_overlap"], 20)
        self.assertEqual(frame.loc["stone", "lag1_pair_count"], 19)


class RandomEnsembleTests(unittest.TestCase):
    def test_default_count_and_method_and_validation(self):
        config = ReconstructionConfig()
        self.assertEqual(config.n_bootstrap, 500)
        self.assertEqual(config.bootstrap_method, "random_holdout")
        self.assertEqual(config.bootstrap_validation_fraction, 1 / 3)
        for kwargs in ({"bootstrap_method": "invalid"}, {"bootstrap_validation_fraction": .6},
                       {"bootstrap_block_years": 0}):
            with self.assertRaises(ValueError):
                ReconstructionConfig(**kwargs)

    def test_random_split_is_disjoint_and_without_replacement(self):
        train, validation = random_holdout_indices(101, 500, 1 / 3, 42)
        self.assertEqual(train.shape, (500, 67))
        self.assertEqual(validation.shape, (500, 34))
        for tr, va in zip(train, validation):
            self.assertFalse(set(tr) & set(va))
            self.assertEqual(len(set(tr) | set(va)), 101)
        tr2, va2 = random_holdout_indices(101, 500, 1 / 3, 42)
        np.testing.assert_array_equal(train, tr2)
        np.testing.assert_array_equal(validation, va2)

    def test_fast_affine_amplitude_matches_existing_calibrator(self):
        rng = np.random.default_rng(9)
        raw = rng.normal(size=(90, 8))
        observed = rng.normal(size=90)
        train, _ = random_holdout_indices(90, 8, 1 / 3, 2)
        availability = np.repeat([2, 3, 5], 30)
        for method, reference in (("none", "observation"), ("ols", "observation"),
                                  ("variance", "observation"), ("variance", "max_proxy_nest"),
                                  ("dynamic_variance", "observation")):
            config = AmplitudeCalibrationConfig(method=method, variance_reference=reference,
                                                 minimum_overlap=15)
            result, _ = _adjust_predictions(raw, observed, train, availability, availability, config)
            for run in range(8):
                ix = train[run]
                mapper = fit_amplitude_calibrator(pd.Series(observed[ix]), pd.Series(raw[ix, run]), config,
                                                 pd.Series(availability[ix]), pd.Series(availability[ix]))
                expected = mapper.apply(pd.Series(raw[:, run]), pd.Series(availability))
                np.testing.assert_allclose(result[:, run], expected, atol=1e-12)

    def test_500_runs_reuse_one_full_nest_pca_and_validate_regression(self):
        import npcrflow.model as modeling
        rng = np.random.default_rng(7)
        years = np.arange(1800, 2001)
        latent = np.sin(np.arange(len(years)) / 8)
        matrix = pd.DataFrame({"a": latent + rng.normal(0, .2, len(years)),
                               "b": latent + rng.normal(0, .2, len(years))}, index=years)
        target = pd.Series(latent + rng.normal(0, .1, len(years)), index=years)
        config = ReconstructionConfig(regression="ols", n_bootstrap=500,
                                      full_network_outer_validation=False, rescreen_outer_folds=False)
        with patch("npcrflow.model._fit_pairwise_basis", wraps=modeling._fit_pairwise_basis) as basis:
            model = fit_native_pcr(matrix, target, years[-101:],
                                   PCAConfig(selection="kaiser", max_components=8), config)
        self.assertEqual(basis.call_count, 1)
        np.testing.assert_array_equal(basis.call_args.args[1], years)
        self.assertEqual(model.ensemble_predictions.shape, (201, 500))
        self.assertTrue(model.selection_table.folds.eq(500).all())
        self.assertTrue(model.selection_table.calibration_samples_per_run.eq(67).all())
        self.assertTrue(model.selection_table.validation_samples_per_run.eq(34).all())
        repeated = fit_native_pcr(matrix, target, years[-101:],
                                  PCAConfig(selection="kaiser", max_components=8), config)
        pd.testing.assert_frame_equal(model.ensemble_predictions, repeated.ensemble_predictions)
        self.assertGreater(model.ensemble_predictions.loc[1900].std(), 0)

    def test_pool_is_all_runs_not_median_of_nest_medians(self):
        years = pd.Index([1900, 1901], name="Year")
        a = pd.DataFrame([[0, 1, 100], [1, 2, 3]], index=years)
        b = pd.DataFrame([[2, 3, 4], [4, 5, 6]], index=years)
        members = [SimpleNamespace(ensemble_predictions=v, core_pids=("a", "b"),
                                   spec=SimpleNamespace(start=1900, end=1901)) for v in [a, b]]
        result = pool_nest_ensembles(members, pd.DataFrame({"median": [2, 3]}, index=years),
                                     ReconstructionConfig(n_bootstrap=3))
        self.assertEqual(result.loc[1900, "median"], 2.5)
        self.assertNotEqual(result.loc[1900, "median"], np.median([1, 3]))
        self.assertTrue(result.ensemble_n.eq(6).all())
        for label, q in (("q05", .05), ("q25", .25), ("median", .5), ("q75", .75), ("q95", .95)):
            np.testing.assert_allclose(result[label], np.quantile(np.concatenate([a, b], axis=1), q, axis=1))

    def test_vectorized_fusion_matches_scalar_member_by_member(self):
        years = pd.Index(np.arange(1900, 1990))
        rng = np.random.default_rng(3)
        core = pd.DataFrame(rng.normal(size=(90, 7)), index=years)
        target = pd.Series(np.sin(np.arange(90) / 5), index=years)
        record = ProxyRecord("stone", years[1::3], target.iloc[1::3].to_numpy())
        matrix, window_target, windows = build_resolution_matrix(target.to_frame("tree"),
            {"stone": record, "tree": ProxyRecord("tree", years, target)},
            ["stone"], target, 1900, 1989, 3)
        predictions = pd.DataFrame(rng.normal(size=(30, 7)), index=windows.index)
        layer = ResolutionNestModel(3, ("stone",), SimpleNamespace(ensemble_predictions=predictions),
                                    matrix, window_target, predictions[0], windows, None)
        config = ReconstructionConfig(n_bootstrap=7)
        batched = fuse_resolution_ensemble(core, [layer], config)
        for run in range(7):
            scalar, _ = fuse_resolution_subnests(core[run], [replace(layer, prediction=predictions[run])], config)
            np.testing.assert_allclose(batched[run], scalar, atol=2e-7)


if __name__ == "__main__":
    unittest.main()

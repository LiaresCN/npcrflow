# Configuration reference

The complete JSON template is `examples/pdo_config.json`. It declares every
current option; no notebook cell or hidden global variable changes the method.
The same fields are available through the typed Python configuration classes.

## Proxy input and screening

- `proxy_filter`: exact and storage-rounding-equivalent de-duplication,
  geographic bounds, archive/proxy/climate metadata filters, maximum plausible
  endpoint, and an optional maximum native resolution for controlled ablation.
- `target`: year/value columns, target season, missing-month requirement,
  detrending, and optional target filtering. Target filtering is off by default.
- `screening`: calibration period, fixed/automatic proxy season, candidate
  seasons, overlap requirements for annual and low-resolution records,
  effective-DOF p and absolute-r thresholds, within-record Holm correction,
  optional cross-record correction, and optional archive caps.

## PCR and regression

- `pca.method`: `pairwise` retains native missingness; `complete` is the strict
  complete-matrix comparison.
- `pca.selection`: fixed, Kaiser, explained variance, or blocked validation.
- `pca.score_ridge`: stabilizes PC scores when only part of the network exists
  in a year; it does not fill proxy values.
- `reconstruction.interpolation`: only `none` is accepted. Any interpolation
  request is a hard error; native missing years remain `NaN`.
- `reconstruction.regression`: `auto`, `ols`, `ridge`, `pls`, `elasticnet`, or
  `random_forest`. With `auto`, `regression_candidates` defaults to Ridge, PLS,
  and ElasticNet, and the family is selected only with inner contiguous blocks
  of the calibration interval. Random forest remains available explicitly but
  is excluded from the default automatic set because of its PDO overfitting.
- `pls_components` and the `random_forest_*` fields expose the corresponding
  model assumptions in the first configuration object.
- `regression_candidates`, `ridge_alphas`, `validation_block_years`, and
  `auto_tune` control inner blocked selection. Outer fold scores and spectral
  diagnostics are never reused for this choice. The selected concrete family
  is reported in each validation row and remains fixed during bootstrap and
  low-frequency-weight tuning.
- `n_bootstrap`, `bootstrap_block_years`, and
  `minimum_bootstrap_success_fraction` control moving-block uncertainty. A run
  fails loudly if too few members fit instead of silently reporting a
  one-member "ensemble".
- `evaluation_lowpass_periods` requests diagnostic comparisons such as 10- and
  20-year low-pass correlation and amplitude. These filters are applied only
  after prediction for evaluation and never change target or proxy inputs.
- `evaluation_period_bands` requests direct spectral checks. For each band the
  output reports observed and reconstructed peak period, peak-period error,
  band amplitude ratio, and phase-aware spectral similarity. Only the longest
  genuinely consecutive common segment is used; a segment shorter than 1.5
  times the slow edge of the band is reported as unavailable.

## Amplitude

`reconstruction.amplitude.method` is one of:

- `none`: retain the regression scale;
- `ols`: training-only affine calibration from raw predictions to target;
- `variance`: training-only matching of mean and standard deviation.
- `dynamic_variance`: smoothly varies the training-derived scale with the
  error/redundancy-weighted number of available proxies;
- `auto`: compares the declared amplitude candidates only in inner contiguous
  folds, penalizing both poor CE/RE and a standard-deviation ratio far from one.

For `variance`, `variance_reference` is either `observation` or
`max_proxy_nest`. The latter uses the densest proxy-availability tier with at
least `minimum_overlap` calibration years and does not construct or save NEST
files. `minimum_overlap` and positive `slope_bounds` are explicit safeguards.
The mapping is refitted inside every fold. Results retain raw and calibrated
predictions, coefficients, reference type and support, standard-deviation
ratio, and variance ratio.

## Proxy errors and redundancy

`reconstruction.proxy_weights` controls weighted PCA and missing-data score
estimation. Measurement-error standard deviations can be supplied by metadata
field names or explicit `(pid, error_sd)` pairs. They are converted to a
signal-reliability weight using the proxy's training-period variance. Nearby,
same-site, or same-source records are placed in one redundancy group only when
their training-period correlation also exceeds the configured threshold.
Their combined group influence is divided across members. All weights are
recomputed without the held-out climate target and written to
`proxy_weights.csv`; no proxy value is altered or filled.

## Multiple resolutions and low frequency

`reconstruction.multiresolution` contains every assumption of this layer:

- `enabled`: reserve low-resolution records for native-window assimilation;
- `state_timestep_years`: currently fixed explicitly to one year;
- `regression_max_resolution_years`: records at or below this resolution enter
  pairwise PCR at their original observed years; slower records enter the
  native-window low-frequency layer. The default `10.0` therefore allows a
  three-year stalagmite record to participate in linear regression without
  filling either of its two intervening years;
- `lowpass_period_years`: frequency boundary of the component allowed to move;
- `smoothness_multiplier`: strength of the second-difference penalty;
- `core_anchor_weight` and `proxy_constraint_weight`: relative data terms. The
  core anchor is normalized by the same fold-safe reference scale used for the
  increment cap, so a low-variance core does not become infinitely rigid;
- `auto_tune`, `proxy_constraint_weight_candidates`, and
  `lowpass_period_candidates`: select low-frequency influence using only inner
  contiguous training folds. Weight zero is always permitted, so enabling the
  module does not force sparse low-resolution evidence into the final state;
- `selection_lowpass_period_years` and `minimum_tuning_folds`: declare the
  frequency-scale objective and minimum evidence needed for automatic tuning;
- `minimum_calibration_overlap`: minimum low-resolution proxy-system overlap;
- `maximum_support_multiplier`: caps an irregular sample's inferred support
  relative to that record's typical resolution, so a hiatus is not treated as
  one centuries-long observation window;
- `maximum_increment_ratio`: optional bound relative to the larger of core
  low-frequency SD and calibration-only target low-frequency SD. Referencing
  only the core would suppress the correction precisely when the core lacks
  low-frequency variance;
- `preserve_calibration_mean`: prevents the low-frequency layer from silently
  shifting the calibration-period baseline.

The output state is annual, but the information is not uniformly annual.
`proxy_availability.csv` labels each year `annual_core`, `low_frequency_only`,
or `unconstrained`. A low-resolution-only annual value is a smooth latent-state
estimate, not recovered year-to-year variability.
`low_frequency_observations.csv` lists the exact native samples and annual
support windows used, replacing opaque saved NEST workbooks with a compact
observation-operator audit.

## Validation, sensitivities, and output

- `rescreen_outer_folds=True` repeats target-based screening inside each
  contiguous outer fold.
- `min_ce`, `min_re`, and `strong_skill_threshold` classify results; they do
  not tune against the outer folds.
- `sensitivity` keeps optional single-proxy, leave-one-proxy-out, directed
  calibration/validation periods, and repeated random network deletion. All
  can remain disabled for a normal full-network reconstruction.
- `output` controls only compact tables and the two summary figures. Per-NEST
  files and bootstrap-member files are not saved.

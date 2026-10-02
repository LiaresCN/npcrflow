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
- `screening.detrend` optionally removes a linear trend from both the proxy and
  target only for their screening correlation. It does not alter the values
  later passed into PCA/PCR.

## PCR and regression

- `pca.method`: `pairwise` retains native missingness; `complete` is the strict
  complete-matrix comparison.
- `pca.selection`: `fixed`, exact legacy `kaiser`, `kaiser_cv`, explained
  `variance`, or `blocked_cv`. Exact Kaiser retains every eigenvalue above
  `kaiser_threshold` (normally 1). `kaiser_cv` uses that count only as the
  upper bound for contiguous validation; it does not assume every component
  above one predicts the target. `max_components` is an explicit safety cap.
- In full validation work, use `blocked_cv` for the predeclared primary run and
  repeat the same screened network with `kaiser_cv`, exact `kaiser`, and
  `variance` (normally `variance_fraction=0.90`) as PCA references. A `fixed`
  1-to-`max_components` sweep is an optional dimension diagnostic. Alternative
  PCA runs remain sensitivities and do not replace the primary run after their
  results are seen.
- `pca.score_ridge`: stabilizes PC scores when only part of the network exists
  in a year; it does not fill proxy values.
- The manifest stores the complete fitted eigenvalue spectrum and full-model
  Kaiser count. Each outer validation row stores its training-only Kaiser count
  beside the actually selected component count.
- `reconstruction.interpolation`: `archive_linear` or `none`.
  `archive_linear` fills only complete interior gaps in the archives named by
  `interpolation_archives` (Wood and Coral by default), only when native
  resolution is no coarser than `interpolation_max_resolution_years`, and only
  through `interpolation_max_gap_years` consecutive missing years. It never
  extrapolates beyond a record's endpoints. Screening always uses observed
  values. `none` retains every native gap as a strict sensitivity.
- `retain_longest_annual_segment=True` trims the saved main reconstruction and
  availability table to the longest consecutive interval with finite annual
  estimates. This controls the final product, not the temporal support of the
  input proxies.
- `reconstruction.detrend_proxies` controls linear detrending before NEST PCA;
  it is independent of screening detrending. No temporal proxy filter is
  applied implicitly.
- Every NEST estimates its PCA mean and standard deviation again from its own
  training years. `reconstruction.standardization_period=(1950, 2000)` adds the
  former WNPSM common-reference z-score once, in memory, before NEST fitting;
  `None` skips this otherwise redundant preprocessing transform. In every
  external validation fit the common reference is intersected with that fit's
  calibration years. Native-resolution proxies use their actually observed
  values in the same reference period and are not interpolated.
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
  low-frequency-weight tuning. A candidate passes the configured internal
  thresholds when its median inner-fold CE and RE meet `min_ce` and `min_re`;
  the complete selection table reports candidates that did not pass. Each
  explicit NEST applies this decision independently.
- `internal_ce_re_comparison="ge"` preserves the frozen inclusive CE/RE
  thresholds. Set `"gt"` to require both internal medians strictly greater
  than their respective thresholds; zero is then rejected at a zero threshold.
  This comparison never changes external sensitivities into acceptance gates.
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
  folds after the PCR structure is fixed. Robust CE/RE is primary, correlation
  is the second criterion, and standard-deviation fidelity is only a later
  tie-breaker. Amplitude can therefore no longer change the selected PC count,
  regression family, or regularization.

For `variance`, `variance_reference` is either `observation` or
`max_proxy_nest`. The latter uses the densest proxy-availability tier with at
least `minimum_overlap` calibration years and does not write an additional
NEST file. `minimum_overlap` and positive `slope_bounds` are explicit safeguards.
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

## Explicit NESTs, multiple resolutions, and low frequency

`reconstruction.method="explicit_nest"` is the primary workflow. It builds
coverage-defined NESTs in memory, fits PCA/PCR separately in every NEST,
applies internal CE/RE there, and combines accepted NEST predictions.
`method="native_missing"` is the former unified-matrix sensitivity.

`reconstruction.nest` controls the primary structure:

- `minimum_span_years`, `minimum_total_proxies`, and
  `minimum_calibration_years` define eligible coverage NESTs;
- `combination` selects the median of overlapping accepted NESTs or the
  densest available NEST;
- `require_internal_ce_re` applies `min_ce` and `min_re` during each NEST's
  model construction, not during supplementary external validation;
- `direct_annual_resolution_years` defines the annual PCA/PCR layer;
- `multiresolution_subnests=True` allows coarser records to participate in
  their own native-window PCA, calibration, and regression;
- `subnest_resolution_bins=(2, 3, 5, 10)` and
  `subnest_max_resolution_years=10` declare the window layers. Annual proxies
  and observations are aggregated onto the same windows as the native proxy;
  the native proxy is never interpolated;
- `minimum_subnest_calibration_windows` prevents a sparse resolution layer
  from being fitted without enough observed target windows;
- `subnest_constraint_weight` and `subnest_smoothness_multiplier` control how
  accepted window predictions constrain the annual state. They do not turn a
  three-year observation into annual data.

`reconstruction.multiresolution` controls the optional fallback for records
coarser than the resolution-sub-NEST limit:

- `enabled`: allow very low-resolution native-window assimilation;
- `state_timestep_years`: currently fixed explicitly to one year;
- `direct_pcr_max_resolution_years` and the backward-compatible
  `regression_max_resolution_years` route records in the unified
  `native_missing` sensitivity; explicit NEST routing is controlled by the
  `nest` fields above;
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
`nest_summary.csv` lists coverage NESTs and their resolution sub-NESTs,
including native-window and predicted-window counts. Very-low-resolution
constraint summaries are retained when that optional layer is active. Detailed
per-window matrices and predictions stay in memory and are not written once
per overlapping NEST. These compact audits replace the opaque saved Excel
workbooks.

## Validation, sensitivities, and output

- `min_ce` and `min_re` apply to median CE/RE only during internal NPCR model
  construction. Values of 0 or 0.05 are typical declared thresholds.
- `external_validation_fraction` defaults to one third. The external segment
  sensitivity has only two directions: later 2/3 -> early 1/3 and early 2/3 ->
  late 1/3; it has no middle holdout and does not classify the main result.
- `full_network_outer_validation=True` runs the directed sensitivity with the
  complete proxy grid selected from the full screening period held fixed.
  `rescreen_outer_folds=True` additionally runs the stricter variant that
  repeats target-based screening inside each direction's calibration period.
  Both use correlation as the declared assessment metric.
- `strong_skill_threshold` remains readable only for v0.4.1 configuration
  compatibility and no longer creates an external `strong` label.
- `sensitivity` keeps optional single-proxy, leave-one-proxy-out, directed
  calibration/validation periods, and repeated random network deletion. All
  can remain disabled for a normal full-network reconstruction.
- `output` controls compact tables and summary figures.
  `show_external_sensitivities=False` keeps directed holdout correlations out
  of the main observation figure while retaining their CSV audit; turn it on
  only when that supplementary panel is wanted. Per-NEST files and
  bootstrap-member files are not saved.

## Runtime without changing the NEST method

The implementation prepares annual proxy data once, computes finite-overlap
correlations in batches, groups native observations into windows, and caches
PC scores by internal fold and PC count. Each NEST still has its own fitted
PCA and CE/RE assessment; no cache is shared across different fitted bases.
These are engineering optimizations, not changes to scientific thresholds.

For a first main-result check, `n_bootstrap=0`,
`full_network_outer_validation=False`, `rescreen_outer_folds=False`, and
disabled proxy-network sensitivities avoid optional repeated reconstructions.
This still runs the primary NEST PCA/PCR and internal CE/RE when
`auto_tune=True`. A zero-bootstrap check does not provide bootstrap uncertainty
intervals; do not interpret its repeated point-value quantile columns as such.
Enable the desired supplementary experiments and bootstrap separately when
they are needed, rather than rerunning them with every configuration change.

A declared smaller PC search or one Ridge penalty can reduce runtime further,
but it changes the search scope and is not guaranteed to preserve the selected
model. To test fixed choices while retaining internal CE/RE, use
`PCAConfig(selection="fixed", n_components=...)`, a single `ridge_alphas`
value, and keep `auto_tune=True`. In contrast, `auto_tune=False` skips internal
candidate validation; it is not the equivalent of a fully CE/RE-checked main
NPCR run.

Use one numerical-library thread per allocated CPU for these small linear
systems. The MEL examples use one CPU and one BLAS thread; requesting many
threads is not a substitute for avoiding repeated score calculations. The
score-cache benchmark and its scope are documented in
`docs/validation_results.md`.

# npcrflow

Version **1.0.0** (2026-10-03) restores the user's
original per-NEST PCA followed by 500 random 2/3-calibration, 1/3-validation
regressions and yearwise pooling of all accepted NEST x run predictions.
It also adds matching-window low-resolution screening. This is the adopted
standard method; earlier release tags and wheels remain available for reproduction.

`npcrflow` turns the former multi-notebook NPCR workflow into one importable,
auditable pipeline.  It reads the raw 21-column Dod2k pickle directly, screens
proxies with effective degrees of freedom, constructs the original
coverage-defined NESTs in memory, fits each NEST independently, reconstructs
without mandatory proxy interpolation, validates on contiguous time blocks,
and writes only compact results. It never writes per-NEST workbooks, matrices,
or bootstrap members.

The original WNPSM, PDO, and Dod2k files are read-only inputs.  All derived
files live under this project.

## Installation

```bash
python -m pip install git+https://github.com/LiaresCN/npcrflow.git@v1.0.0
```

For an offline MEL installation, use the release wheel without modifying the
source environment:

```bash
python -m pip install /path/to/npcrflow-1.0.0-py3-none-any.whl
```

## Minimal use

```python
from pathlib import Path
from npcrflow import (
    AmplitudeCalibrationConfig, ExplicitNestConfig, MultiresolutionConfig,
    OutputConfig, PCAConfig, PipelineConfig, ProxyFilterConfig,
    ProxyWeightConfig, ReconstructionConfig, ScreeningConfig,
    TargetConfig, run_pipeline,
)

cfg = PipelineConfig(
    proxy_filter=ProxyFilterConfig(
        latitude=(0, 60),
        longitude=(60, 180),
        climate_variables=("moisture", "temperature", "temperature+moisture"),
    ),
    target=TargetConfig(value_column="wnpsm", months=(6, 7, 8, 9)),
    screening=ScreeningConfig(
        period=(1940, 2010),
        season_mode="auto",
        p_threshold=0.10,
        r_threshold=0.20,
    ),
    pca=PCAConfig(method="pairwise", selection="kaiser", max_components=8),
    reconstruction=ReconstructionConfig(
        method="explicit_nest",
        calibration_period=(1940, 2010),
        interpolation="archive_linear",
        interpolation_archives=("Wood", "Coral"),
        interpolation_max_gap_years=2,
        interpolation_max_resolution_years=2.0,
        retain_longest_annual_segment=True,
        detrend_proxies=False,
        standardization_period=None,  # or, e.g., (1950, 2000)
        regression="ols",
        n_bootstrap=500,
        bootstrap_method="random_holdout",
        bootstrap_validation_fraction=1/3,
        min_ce=0.1, min_re=0.1, internal_ce_re_comparison="gt",
        full_network_outer_validation=False, rescreen_outer_folds=False,
        nest=ExplicitNestConfig(
            minimum_span_years=50,
            combination="median",
            require_internal_ce_re=True,
            direct_annual_resolution_years=1.5,
            multiresolution_subnests=True,
            subnest_resolution_bins=(2, 3, 5, 10),
            subnest_max_resolution_years=10.0,
        ),
        amplitude=AmplitudeCalibrationConfig(
            method="auto",
            auto_candidates=("variance_observation", "variance_max_proxy_nest"),
            variance_reference="observation",  # observation | max_proxy_nest
            minimum_overlap=20,
            slope_bounds=(0.25, 4.0),
        ),
        proxy_weights=ProxyWeightConfig(
            enabled=False,  # enable as an audited independence sensitivity
            redundancy_radius_km=250,
            redundancy_correlation_threshold=0.85,
        ),
        multiresolution=MultiresolutionConfig(
            enabled=False,
            state_timestep_years=1,      # annual latent-state output
            regression_max_resolution_years=10.0,
            lowpass_period_years=10.0,
            smoothness_multiplier=1.0,
            core_anchor_weight=1.0,
            proxy_constraint_weight=1.0,
            auto_tune=True,              # training-only; zero is a candidate
            proxy_constraint_weight_candidates=(
                0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0
            ),
            lowpass_period_candidates=(10.0, 20.0),
            selection_lowpass_period_years=10.0,
            minimum_tuning_folds=2,
            minimum_calibration_overlap=8,
            maximum_support_multiplier=3.0,
            maximum_increment_ratio=1.0,
            preserve_calibration_mean=True,
        ),
    ),
    output=OutputConfig(directory=Path("results/wnpsm"), save_nests=False),
)

result = run_pipeline(
    "/share/home/lrs/Date collection/proxy数据库/"
    "Dod2k_V20260513_filteredproxy_lrs_use.pkl",
    "/path/to/annual_or_monthly_observations.xlsx",
    cfg,
)
```

All choices are in this one configuration cell.  If no reconstruction end is
specified, the output period follows the selected proxy coverage and can extend
beyond 2000 whenever the proxy network permits it.
Either bound may be left automatic; for example,
`reconstruction_period=(0, None)` fixes a 0 CE start while retaining the
data-driven endpoint.

Raw Dod2k input is conservatively de-duplicated in memory by exact canonical
time/value identity and by a second storage-rounding check. The latter requires
the complete timestamp vector, archive, proxy variable, and rounded location to
match, with value differences limited to numeric storage precision; correlation
alone is never enough to delete a record. The source file is never modified,
and dropped rows plus group membership are written as compact audit CSVs. Set
`deduplicate_near=False` to retain rounding-equivalent copies in a controlled
comparison, or `deduplicate_exact=False` to disable de-duplication entirely.

Observation filtering is explicit and off by default. Use
`TargetConfig(lowpass_years=None)` for the original observed index (the primary
PDO workflow), or set a period such as `lowpass_years=10` only for a labelled
low-frequency sensitivity experiment. The manifest records this choice.
Proxy screening detrending (`ScreeningConfig.detrend`) and reconstruction-stage
proxy detrending (`ReconstructionConfig.detrend_proxies`) are separate switches
and are both off in the PDO baseline. Proxy values are never silently filtered.
In the standard random-holdout method, PCA standardization and loadings are
fitted once over that full NEST's proxy data before random regression splits.
The optional moving-block and zero-ensemble blocked checks use calibration
PCA instead. Set `standardization_period=(1950, 2000)`, as in the WNPSM example, to add
the former common-reference proxy z-score once in memory *before* NEST
construction. In an outer validation run, this first reference uses only the
intersection with that run's calibration years; no intermediate proxy or NEST
workbook is written.

## Command line

No package changes to the user-managed `mybase` environment are required:

```bash
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
/share/home/lrs/.conda/envs/mybase/bin/python -m npcrflow.cli run \
  PROXIES.pkl OBSERVATIONS.xlsx --config examples/pdo_config.json
```

Create the audited Dod2k copy with:

```bash
/share/home/lrs/.conda/envs/mybase/bin/python -m npcrflow.cli deduplicate \
  SOURCE.pkl OUTPUT.pkl --report dropped.csv --groups groups.csv
```

## What changed from the notebooks

- The original NPCR structure is retained explicitly: proxy coverage defines
  unique NESTs, each NEST has its own PCA/PCR and internal CE/RE assessment,
  and accepted NEST reconstructions are combined. The optimization removes
  only the Excel hand-off and saved NEST matrices; `nest_summary.csv` keeps a
  compact audit.
- Mixed resolution is handled inside each coverage NEST. A three-year
  stalagmite joins a three-year resolution sub-NEST together with three-year
  window means of the annual proxies and target, and therefore participates in
  PCA, calibration, and regression. Its accepted window prediction constrains
  the annual state while the annual layer supplies within-window variability.
  No stalagmite observations are interpolated into artificial annual values.

- Proxy screening always uses observed values. During reconstruction,
  `archive_linear` may fill only complete, bounded short gaps in explicitly
  allowed near-annual archives. The default allows Wood and Coral, limits gaps
  to two years, and never extrapolates either record endpoint. Ice,
  speleothem, and other archives retain their native timestamps.
  `interpolation="none"` remains available for a strict native-gap sensitivity.
  Pairwise covariance estimates the PCA basis from available overlaps, and a
  least-squares score is computed from the proxy values available in each year.
- Screening p values use lag-1 effective sample size. Automatic seasonal
  selection applies a Holm correction across seasons tested for each
  subannual record; already annual records are not retested under fake seasons.
  An optional second Benjamini-Hochberg FDR correction can be applied across
  all candidate proxy records. Archive-specific caps are also configurable;
  candidates within a capped archive are ranked by adjusted p value, absolute
  correlation, effective sample size, and stable proxy ID.
- The final reconstruction always uses the complete proxy grid that passes the
  declared effective-DOF significance and absolute-correlation thresholds.
  Continuous holdouts are supplementary robustness sensitivities, not gates
  that accept or reject this main result.
- Two directed segment sensitivities are provided: later two thirds calibrate
  the withheld early third, and early two thirds calibrate the withheld late
  third. There is no compulsory middle holdout. Each direction is calculated
  both with the complete screened proxy grid fixed (`full_proxy_network`) and,
  when enabled, with target-based proxy screening repeated inside the training
  period (`rescreened_network`). Their declared assessment metric is
  correlation; CE/RE remain available in the audit table but do not classify
  the main reconstruction.
- Standard paleoclimate definitions are used: RE is referenced to the
  calibration mean and CE to the validation mean.  CE is never randomized.
- Ordinary multiple linear regression (`regression="ols"`) is the v1.0
  standard. Ridge, PLS, ElasticNet and random forest remain declared options;
  they are not presumed to outperform OLS. `regression="auto"` compares
  explicitly configured families in the declared internal validation protocol.
  Random forest's bounded extrapolation is a limitation for reconstruction.
- CE/RE are used only during internal NPCR model construction, analogous to
  the regression screening within the former WNPSM NEST loop. PC count,
  regression, regularization, and amplitude candidates are judged by their
  inner-fold median CE and RE against configurable `min_ce`/`min_re` values
  (v1.0 default: both strictly >0.1; explicitly selectable >0.05 or >0).
  The standard computes these medians across500 random 2/3-to-1/3 runs.
  External segment sensitivities never change those
  choices or label the full reconstruction as failed.
- Amplitude calibration is explicit. The v1.0 standard compares only the
  observation-variance and max-proxy-tier variance references; `none` remains
  available. Per-NEST calibration does not guarantee final combined SD=1.
  `ols` estimates a
  training-only affine map; `variance` matches the training-period mean and
  standard deviation either to the observation or to the densest eligible
  proxy network (`variance_reference="max_proxy_nest"`). The latter is the
  auditable, no-file equivalent of the legacy maximum-proxy NEST: it uses the
  highest proxy-count threshold that still supplies `minimum_overlap` training
  years. The map is fitted again inside every inner and outer
  fold. Raw predictions, calibrated predictions, slope, and intercept remain
  auditable, so amplitude cannot be changed after looking at withheld scores.
- In the unified-matrix sensitivity, native records can still participate in
  pairwise missing-data PCR. In the primary explicit-NEST method, records up
  to ten-year resolution instead participate in their corresponding
  resolution sub-NEST. Only eligible Wood/Coral short gaps may be filled under
  the declared interpolation policy; other unobserved years remain `NaN`.
  Records too sparse for a resolution sub-NEST can use native-window
  low-frequency assimilation. The latent result remains annual, but a decadal observation
  constrains only the mean low-pass state over its native support interval. It
  does not become ten annual observations. The PCR core is explicitly split
  into low- and high-frequency components; only a smooth, bounded increment to
  the low-pass component is solved, and the defined high-frequency component
  is added back unchanged. In low-resolution-only intervals, annual values are
  a smooth latent-state estimate and must not be interpreted as observed annual
  variability. This is a scalar variational observation operator, not full
  dynamical 4D-Var.

The former whole-trajectory native-window adjustment failed the PDO edge tests
and was replaced in version 0.2 by the component-separated design above.
Version 0.3 adds resolution ablation and explicit 10/20-year validation; the
single unique >10-year PDO record still degrades the late holdout, so this layer
remains experimental and off. Native 1.5–5-year records improve some
low-frequency correlation, amplitude, and phase measures when they enter PCR
at observed years only, but the gain is direction-dependent and does not
robustly recover the observed PDO spectral peak. They are therefore retained
as validation-gated candidates rather than assumed to be beneficial.

Version 0.3.4 corrects a low-frequency amplitude scaling defect. Core anchoring
and the optional increment cap now use the larger of the core low-frequency SD
and the calibration-only observed low-frequency SD. Previously, a core that
lacked low-frequency variance was paradoxically anchored most strongly and
allowed almost no correction. In a 360-member paired synthetic audit, three or
more independent 12-year records recover a withheld 25-year cycle and near-unit
amplitude under low-to-moderate noise without proxy interpolation. One record
does not recover amplitude reliably. The real PDO network still selects zero
weight for its single independent >10-year d18O record, so this controlled
capability is not presented as a real PDO improvement.

Version 0.3.5 adds the explicit observation versus maximum-proxy-network
variance reference. The legacy notebook calculated maximum-NEST statistics but
its active assignment scaled non-reference NESTs to observations while leaving
the reference NEST unscaled. The new options are mutually explicit, fold-safe,
and report the chosen reference support without writing NEST workbooks.

Version 0.4.0 is the frozen method release. It adds training-fold-only
automatic amplitude selection, a smooth variance model indexed by effective
proxy-network size, and error/redundancy-weighted PCA. The PDO example enables
automatic amplitude selection but leaves redundancy weighting off by default:
the independence sensitivity correctly downweights replicated Palmyra, Fiji,
and Vanuatu coral information, but lowers PDO external CE/RE. The capability
is retained for honest pseudo-replication tests rather than forced into every
reconstruction.

Version 0.4.1 separates structural PCR selection from amplitude calibration.
PC count, regression, and regularization are selected from raw predictions;
only then can an amplitude option be chosen. It also restores exact Kaiser
selection as an auditable compatibility mode and adds `kaiser_cv`, while
retaining blocked validation as the PDO default. Direct Kaiser keeps 19 PCs
and overfits the current PDO network; the revised default keeps four PCs,
restores apparent `r=0.778`, and gives outer-block correlations
`0.561/0.559/0.697` with positive CE in every block.

For a full v1.0 method evaluation, compare the predeclared capped `kaiser` primary run
with `kaiser_cv`, `blocked_cv`, and90% cumulative-variance selection on the
same screened proxy grid. A fixed-PC sweep can be added as a dimension
diagnostic. These PCA alternatives are structural sensitivity references, not
a post hoc contest for replacing the primary reconstruction.

Version 0.4.2 restores the interpretation used by the WNPSM/PDO workflow. Main
result evidence is the full-network reconstruction's correlation with the
observed target plus CE/RE from internal NPCR model construction. Continuous
segment tests are supplementary only, use the two directed 2/3-to-1/3 edge
splits, and report both fixed-full-grid and fold-rescreened networks. They no
longer produce a pass/fail or `strong` classification for the main result.

Version 0.5.0 makes the saved main product the longest continuous interval of
finite annual estimates. Near-annual Wood and Coral records may fill complete
interior gaps up to a declared length (two years by default); no endpoint is
extrapolated, and all other archives remain native. Every filled proxy-year is
reported separately and `interpolation="none"` remains available.

Version 0.6.0 restores explicit coverage-defined NESTs as the primary method
without restoring Excel intermediates. Every NEST is independently fitted and
screened by internal CE/RE. Resolution sub-NESTs let 2–10-year records
participate in window-scale PCA/PCR with annual proxies aggregated to matching
windows; their predictions are fused into the annual state without proxy
interpolation.

Version1.0.0 adopts the original full-NEST PCA ->500 repeated random
2/3-calibration/1/3-validation regressions ->median CE/RE NEST acceptance
->pool all accepted NEST x run predictions by year sequence. Matching-window
screening and native-step DOF make low-resolution screening consistent in
scale with the reconstruction layers. See [release notes](docs/release_v1.0.0.md).

## Compact outputs

Depending on the enabled options, a run writes `source_qc.csv`, `proxy_screening.csv`,
`reconstruction.csv`, `observation_fit.csv`, `primary_reconstruction_summary.csv`,
`validation_folds.csv`, `validation_summary.csv`,
`model_selection.csv`, `nest_summary.csv`, `proxy_availability.csv`,
`proxy_interpolation_audit.csv`, sensitivity summaries, one
proxy map, one observation diagnostic figure, and `manifest.json`. The manifest contains input hashes and the full
configuration. `save_nests` remains a compatibility field and no
implementation path writes NEST matrices, Excel workbooks, or member
ensembles.

`primary_reconstruction_summary.csv` puts the declared main-result evidence in
one row: the full reconstruction's correlation with observations and the
selected internal NPCR candidate's median CE/RE. `validation_folds.csv` and
the mode-specific `validation_full_proxy_network.csv` and
`validation_rescreened_network.csv` are supplementary robustness results.
Published reconstructions are not used as the validation reference.

Both apparent and outer-fold tables report configured low-pass diagnostics
(10 and 20 years in the PDO template). These compare filtered observations with
filtered predictions after reconstruction; they do not filter the PDO target
used for screening or fitting and do not interpolate proxy records. A fold
shorter than two requested periods is reported as unavailable rather than used
to make a low-frequency claim.

Configured period bands add a distinct frequency-domain audit: observed and
reconstructed dominant period, band amplitude ratio, and phase-aware spectral
similarity. This operates only on consecutive final observation/prediction
segments, never fills a proxy gap, and refuses bands that the validation length
cannot resolve adequately.

`source_qc.csv` gives one row for every loaded record, including all metadata
filter reasons.  Records with implausible end years are excluded and reported,
never silently edited in the source database.

In explicit-NEST mode, `proxy_availability.csv` reports the number of
contributing NESTs and the largest annual, resolution-sub-NEST, and optional
low-resolution proxy counts available each year. The `native_missing`
sensitivity retains the `annual_core`, `low_frequency_only`, and
`unconstrained` labels. `proxy_interpolation_audit.csv` records archive,
native resolution, eligibility, and number of filled years for each proxy.
When amplitude calibration is enabled,
`reconstruction.csv` retains `point_raw` and `median_raw` beside the calibrated
series. A multiresolution run additionally retains the core low/high-frequency
components, adjusted low-frequency component, and applied increment.
`nest_summary.csv` reports each resolution sub-NEST's native observed-window
count and accepted prediction-window count. Detailed window matrices and
predictions remain in memory because writing them once for every overlapping
coverage NEST would recreate the large intermediate-file problem. A
three-year stalagmite remains a sequence of native three-year observations and
is never expanded into three synthetic annual observations.

Each enabled network sensitivity writes both its repeat-level table and a
compact `*_summary.csv` containing the minimum, 5/25/50/75/95th percentiles,
and maximum of the applicable correlation, RMSE, RE, and CE metrics.
`validation_summary.csv` reports the two directed external correlation
sensitivities by network mode. CE/RE and error fields remain available only as
context; there is no external gate or main-result class. Whether to display a
sensitivity in a paper is a reporting choice, while its saved audit result is
kept reproducible. `show_external_sensitivities=False` leaves these tests out
of the main observation figure by default. Low-frequency runs also report
`core_*` diagnostics before adjustment.

## Current verification

Run the local unit suite on a compute node:

```bash
PYTHONPATH=src /share/home/lrs/.conda/envs/mybase/bin/python \
  -m unittest discover -s tests -v
```

Version1.0.0 includes tests for end-to-end mixed-
resolution case in which a native three-year speleothem and annual tree-ring
means jointly enter a three-year PCA/PCR layer. The development default uses
500 random 2/3-calibration, 1/3-validation regressions per NEST, with the
NEST's PCA built once, and pools accepted NEST x run predictions by year for
the median and quantiles. Moving-block Bootstrap remains an explicit option.

The validated500-run PDO benchmark takes3:57 on one c061 CPU (3:51.74 pipeline),
with65 selected proxies,374 accepted coverage NESTs,444 accepted resolution
layers, annual552–2011 and apparent observation r0.79258069. SD ratio0.69439977
remains damped; no independent external validation was run in this benchmark.
See [the complete benchmark record](docs/random_nest_ensemble_2026-10-03.md).

The primary PDO regression driver is `examples/run_pdo_from_raw_dod2k.py`; the
associated Slurm launcher is `scripts/run_pdo_raw_dod2k.slurm`. The older
`run_pdo_validation.py` name is retained only as a compatibility entry point to
the same raw-input workflow.

The current PDO engineering test uses the unfiltered observed PDO and only
`d18O` from the four requested archives. Main evidence comes from the complete
screened-network correlation and internal NPCR CE/RE. Directed segment tests,
single-proxy reconstructions, random proxy deletion, and alternate network
treatments are reported separately as robustness sensitivities. See
[`docs/validation_results.md`](docs/validation_results.md) before interpreting
or publishing a reconstruction.

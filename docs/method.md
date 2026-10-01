# Method summary

## Screening

The observation target is unfiltered by default. Optional target low-pass
filtering is a declared preprocessing sensitivity and is never silently applied.
The primary PDO example uses the original observed PDO index.

For each proxy and target overlap, the lag-1 effective sample size is

`N_eff = N (1 - r1_proxy r1_target) / (1 + r1_proxy r1_target)`.

Lag-1 correlations are computed only across consecutive annual pairs.  The
Pearson t statistic then uses `N_eff - 2` degrees of freedom.  For subannual
records, configured seasons are aggregated without filling missing months;
automatic season selection reports both raw and Holm-adjusted p values.
The season-adjusted values can then be corrected across the complete candidate
proxy pool with Benjamini-Hochberg FDR. The current engineering-focused,
unfiltered PDO configuration instead uses per-record `p <= 0.1`, `|r| >= 0.2`,
and `d18O` records from Wood, Coral, GlacierIce, and Speleothem archives. The
optional archive-cap output retains pre-cap selection, within-archive rank, and
cap-exclusion columns when such a temporary engineering limit is requested.
An optional separate minimum-overlap threshold admits genuinely low-resolution
records to the low-frequency layer without relaxing the annual-network rule.

## Native-resolution PCR

The annual matrix contains observed annual/seasonal bins and `NaN` elsewhere.
The default reconstruction policy may linearly fill only complete, bounded
short gaps in explicitly allowed near-annual archives (Wood and Coral by
default). Gaps longer than the configured limit, record endpoints, and all
other archives remain untouched. Screening still uses observed values only.
Pairwise proxy correlations are estimated from finite overlaps, symmetrized,
and decomposed spectrally. For each year, PC scores are solved from the
loadings of the proxy values available that year. A small configured ridge
penalty stabilizes underdetermined years; years with fewer than the required
values remain missing.

The saved main result is trimmed to the longest contiguous interval having a
finite annual reconstruction. This produces the longest defensible annual
series rather than downsampling the result to the coarsest proxy or retaining
disconnected annual fragments. Native and interpolated proxy counts remain
separate in the availability audit, and every filled proxy-year is listed in
`proxy_interpolation_audit.csv`.

When proxy weighting is enabled, standardized proxy columns are multiplied by
the square root of their training-derived reliability/redundancy weight before
the eigendecomposition and missing-data score solve. Supplied measurement
errors become signal-reliability weights. Records are grouped as redundant
only when they are geographically close, share a site/source, or both, and
also exceed the configured training-period correlation threshold. A group has
approximately the influence of one independent record rather than the number
of database copies or nearby series it contains. These weights never fill or
alter an observation.

PC count can be fixed, selected by cumulative explained variance, reproduced
with the legacy Kaiser rule (`eigenvalue > kaiser_threshold`), bounded by that
rule and then selected with contiguous validation (`kaiser_cv`), or selected
directly by blocked validation. Kaiser is applied to the standardized
training-period correlation structure and capped explicitly; it is not assumed
to guarantee predictive skill. Regularization strength and—when
`regression="auto"` is requested—the regression family are chosen by the same
inner contiguous-block design. The default automatic candidate set is Ridge,
PLS, and ElasticNet. Random forest is intentionally excluded from that set
because its PDO apparent fit did not generalize under contiguous external
validation.

For a full NPCR method evaluation, keep `blocked_cv` as the predeclared primary
PCA rule and run `kaiser_cv`, exact `kaiser`, and cumulative `variance`
selection on the same screened proxy grid as structural references. A fixed-PC
sweep may additionally diagnose sensitivity to dimension. Hold the target,
calibration period, regression candidates, amplitude procedure, and random seed
constant. Compare selected PC count, main observation correlation, internal
median CE/RE, SD ratio, and directed external correlations. These alternative
PCA runs are supplementary and are not used to retrospectively choose the main
result.
Outer contiguous folds refit target-based proxy screening, proxy means/scales,
the PCA basis, component choice, and regression. The held-out target therefore
cannot influence even the network membership. The full reconstruction is fit only after validation.
Moving-block calibration resampling provides compact pointwise quantiles; the
individual bootstrap members are not written. A configured minimum success
fraction prevents failed members from being silently reduced to a nominal
ensemble.

## Amplitude calibration

Amplitude handling is a declared part of the fitted model. `none` preserves the
raw regression scale. `ols` fits `target = intercept + slope * prediction` on
the current training years. `variance` uses the ratio of training-period target
and prediction standard deviations and an intercept that preserves the
training-period mean. Configured positive slope bounds prevent unstable
extrapolation. Every inner and outer fold estimates its own mapping without
using the withheld target. Validation tables retain both raw and calibrated
scores plus the fitted slope and intercept. Exact variance matching is not the
default because it can amplify unpredictable variance and worsen CE/RMSE.

`dynamic_variance` estimates variance ratios in quantile bins of the effective
weighted proxy count, shrinks each local estimate toward the global
training-period ratio, and linearly interpolates the bounded slopes between bin
centres. It is therefore continuous rather than a hard NEST boundary.
`method="auto"` treats no adjustment, both global variance references, and the
dynamic model as inner-fold candidates. Selection is explicitly two-stage.
First the PC count, regression family, and regularization are chosen from raw
predictions. The skill-floor gate applies to the minimum inner-fold CE/RE, not
only their median. With that structure locked, amplitude candidates are ranked by
robust CE/RE, then correlation, then standard-deviation fidelity. The selected
option is locked before the outer holdout is predicted. This prevents a better
amplitude ratio from purchasing a worse PCR structure.

## Low-frequency constraint

The latent state has an explicit one-year time step, so the default final
reconstruction is annual even when proxy resolutions differ. For a
low-resolution proxy observation at time `t` with native support `d`, an
observation row averages the annual low-pass state over
`[t-d/2, t+d/2)`. Each sample's support is derived from the midpoints to its
neighboring native timestamps, so an irregular series does not acquire a
single artificial interval. Inferred width is capped by a configured multiple
of the record's median native resolution, so a depositional hiatus remains a
gap rather than making one sample represent decades or centuries. A
calibration-period linear proxy system model maps the proxy
to the low-pass target component. The PCR reconstruction is decomposed into a
low-pass component and its complementary high-frequency component. A weighted
variational solve estimates only a smooth low-frequency increment from core
anchoring and proxy-window mismatch. The increment may be explicitly bounded
relative to the larger of the core low-frequency scale and the low-frequency
scale of the current calibration-only target, then mean-centered over
calibration. This prevents a low-variance core from imposing a near-zero cap
on the component that low-resolution evidence is intended to restore, without
using any held-out observation. The core-anchor normalization uses the same
reference scale; otherwise a deficient near-zero core would paradoxically
receive an arbitrarily strong anchor.
The original high-frequency component is then added back unchanged. This
retains native timestamps and avoids treating a decadal or irregular value as
ten annual observations.

When automatic multiresolution tuning is enabled, proxy-constraint weight and
low-pass period are chosen using contiguous folds contained entirely within the
current training interval. Weight zero is an explicit candidate. Candidate
ranking prioritizes the smaller of annual and configured low-pass CE, with a
small low-pass-correlation tie-breaker. If there are too few inner folds or a
low-resolution record cannot meet its calibration-overlap requirement inside
them, the selected weight is zero. Thus the variational layer can use useful
native-window evidence but cannot be forced into an outer prediction merely
because the option was enabled.

Annual values in intervals supported only by low-resolution proxies represent
a smooth latent low-frequency estimate. They do not contain independently
resolved annual information and must be labelled accordingly.

When this layer is enabled, each directed segment sensitivity refits the model
mapping using its calibration segment, then applies native proxy constraints to
the withheld third. The fixed-full-network mode retains the complete screened
grid; the rescreened mode also repeats proxy selection. `core_*` columns retain
the corresponding unadjusted-PCR diagnostics. These are supplementary results,
not main-result CE/RE gates.

## Skill

- `RE = 1 - SSE_validation / sum((y_validation - mean(y_calibration))^2)`
- `CE = 1 - SSE_validation / sum((y_validation - mean(y_validation))^2)`

The smaller of inner-fold median RE and CE drives automatic model selection
inside NPCR construction, corresponding to the statistical check formerly
performed for each WNPSM NEST regression. Configurable `min_ce` and `min_re`
are normally 0 or 0.05. They do not act on external segment tests.

Main reconstruction evidence consists of the complete screened-network
correlation with observations plus these internal NPCR CE/RE results. External
segment experiments supplement robustness only. They use two directions—later
two thirds to reconstruct the early third, and early two thirds to reconstruct
the late third—with no compulsory middle holdout. One variant fixes the full
screened proxy grid; a second refits screening inside each calibration segment.
Correlation is the declared external metric. No published reconstruction is
used as a validation target, and no external CE/RE creates a failure label.

## Regression alternatives

OLS, Ridge, PLS, ElasticNet, and an optional random forest share the same
leakage-safe validation path and native-missing PC scores. A fixed family can
be declared, or `auto` can select among an explicit candidate tuple entirely
inside the current calibration interval. The selected family is then locked
for low-frequency-weight tuning and moving-block bootstrap refits.
Ridge remains the conservative single-model default because sparse-record PC
scores are noisy, effective rank is uncertain, and prediction is required far
outside the instrumental interval. Tree ensembles and neural
networks are not enabled merely from their in-period fit: their bounded or
poorly identified extrapolation must first beat the linear baselines under the
same contiguous-block and network-drop tests. Configured 10- and 20-year
low-pass diagnostics are calculated only after prediction to test the intended
frequency-scale contribution; they never preprocess proxies or the fitted
target.

For a more direct PDO-cycle check, configured period bands are evaluated on
the longest consecutive common annual segment after prediction. Observation
and reconstruction are linearly detrended and tapered identically. The audit
reports dominant band period, amplitude ratio, and normalized complex
cross-spectrum: its real part is a phase-aware similarity and its magnitude is
a phase-free spectral-pattern similarity. At least 1.5 cycles of the slow band
edge are required. This makes a 10–30-year diagnosis unavailable in a 34-year
outer third but available as a labelled sensitivity in a 50/51-year split.

A generic Kalman smoother is also not silently applied. Without an independently
justified state transition and process-noise model it would manufacture temporal
persistence and make validation look smoother by construction. The explicit
native-window variational layer above supplies the requested low-frequency
constraint while keeping its assumptions observable and fold-testable.

# NPCRFlow1.0.0 — 2026-10-03

The user's original NPCR statistical sequence is the adopted standard:
coverage NEST ->once-per-full-NEST PCA ->500 random 2/3-calibration,
1/3-validation regressions ->median CE/RE acceptance ->pool all accepted
NEST x run reconstructions by year for the median and5/25/75/95 percentiles.

## Standard configuration

- Pairwise missing-data PCA; Kaiser eigenvalue>1; explicit PC cap8.
- Ordinary multiple linear regression (OLS), no automatic Ridge penalty search.
- Configurable `n_bootstrap=500`, `bootstrap_method="random_holdout"`,
  `bootstrap_validation_fraction=1/3`; stable declared random seed.
- Internal NEST median CE>0.1 **and** RE>0.1. Users can explicitly choose
  >0.05 or >0; no automatic relaxation or external holdout veto.
- Standard CE uses validation mean; RE uses calibration mean. The original
  Notebook's variance/median-denominator formulas are not copied.
- Two declared training-only amplitude candidates: observation variance and
  densest proxy-availability tier variance. Final pooled-median SD is not
  guaranteed to equal observation SD.
- Optional two-direction early/late1/3 external sensitivities; off by default,
  assess correlation and never accept/reject the full main reconstruction.
- Ridge, alternate PC rules, moving-block resampling, network weighting and
  other sensitivities remain explicit options, not silent replacements.

## Integrated raw-input workflow

Read raw proxy/observation files, conservatively deduplicate in memory, screen
at declared significance/absolute-r thresholds, reconstruct, and write compact
audits from one configuration. No saved NEST workbooks, matrices or member files.
Low-resolution screening averages the observation onto matching native windows,
counts one actual paired window once, and estimates effective DOF at the native
window step without bridging missing windows. These records participate in
2/3/5/10-year PCA/PCR layers; fuse each layer member into its annual member
before final pooling. Stone/ice observations are never interpolated. Only
declared near-annual Wood/Coral bounded interior gaps may be filled (two years);
strict `interpolation="none"` remains available. Endpoints follow data rather
than2000; main output is the longest contiguous finite annual interval.

## Verified benchmark and limits

Unfiltered observed PDO/d18O,1900–2000 screening/calibration, p<=.1 and |r|>=.2:
65 selected records;374/412 accepted coverage NESTs,500 successful regressions
each;444 accepted native layers with9 actual native PCA proxy members.
Annual552–2011, apparent observation r0.79258069, SD ratio0.69439977;
internal accepted-layer median CE0.399163/RE0.467924. Full one-CPU c061 run
takes3:57 scheduler wall (231.739773s pipeline) and writes5.63MB compact outputs.
All saved years have ordered, nonzero-width quantiles. External tests were
not run in that benchmark. Details: [benchmark record](random_nest_ensemble_2026-10-03.md).

The final distribution describes random-regression-split/network spread,
conditional on screened proxies and model choices, not all measurement,
age-model or selection uncertainty. PCA is shared over the full NEST before
random regression splits, not independently fitted on each training split.
The older no-ensemble benchmark used different proxy screening, PCA fit scope
and estimator; do not advertise an identical-output or isolated-Bootstrap speedup.

Earlier tags/wheels remain unchanged. The GitHub v1.0.0 release provides an
offline wheel and source distribution; MEL can instead use `PYTHONPATH=src`
with existing mybase without installing anything into that environment.

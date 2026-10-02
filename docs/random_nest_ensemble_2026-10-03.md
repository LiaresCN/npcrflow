# Restored NEST ensemble and native-window screening

Experimental branch `codex/ols-kaiser-runtime`, scientific implementation
`e184298f15495ec096f9d70a5105bb7ded1fe36c`. Frozen v0.6.0 and its wheel remain
unchanged; no new release is claimed.

## User-approved workflow

1. Screen actual proxy observations against observed PDO. Annual proxies use
   annual pairing; native records slower than 1.5 years use matching-window
   PDO means and window-step effective degrees of freedom, with no proxy fill.
2. Construct original unique coverage NESTs in memory.
3. Fit PCA once over each full NEST's proxy data, as in WNPSM Notebook cells
   8–10. Use declared Kaiser eigenvalue >1 with cap8; skip zero qualifying PCs.
4. Randomly partition finite calibration overlap into 2/3 training and 1/3
   validation, without replacement, 500 times. Refit ordinary regression in
   each run, predict the full NEST, and evaluate that run's validation.
5. Apply median internal CE **>0.1** and RE **>0.1** to the NEST, not to every
   individual member. These same runs supply the reconstruction distribution.
6. Native 2/3/5/10-year layers follow the same PCA/random-regression sequence
   on complete windows. Fuse each native layer's run into its corresponding
   annual run without interpolating the proxy observations.
7. Pool **all finite runs of all accepted NESTs by year** for the median and
   q05/q25/q75/q95, matching Notebook cell14. Never replace this with median
   of NEST medians or first combining NESTs within a replicate.

This restores the statistical sequence, not bitwise legacy Notebook output:
pairwise missing-data PCA, declared PC cap, standard RE/CE definitions,
native-resolution fusion, seed, and declared per-training amplitude choices
are retained from the modular method. The Notebook's nonstandard RE variance
and CE median denominators are not copied. Proxy-only PCA uses the full NEST
and is shared across random partitions; this is internal regression checking,
not fully independent PCA validation. External directed contiguous holdouts
remain optional supplementary sensitivities, not main-result acceptance gates.

`n_bootstrap=500` is configurable, including explicit `0` for an engineering
point-fit check. `bootstrap_method="random_holdout"` is the development
default. `bootstrap_validation_fraction=1/3` is configurable. The previous
`moving_block` method remains an explicit alternative, not the restored
default. `ensemble_n` counts NEST x run values and can exceed500. These are
empirical split/network-spread quantiles, not an all-source confidence interval.
Age, measurement and selection uncertainty are not sampled.

## Completed full PDO run

- Launch: `sbatch examples/run_pdo_ols_kaiser_runtime.sbatch`.
- Slurm30127, c061, 1CPU, PythonPID3846785, China2026-10-03 01:02:24.
- Completed exit0, scheduler wall **3:57**, CPU3:49.625, maximum RSS1839736KiB.
- Entire raw-input pipeline including compact output writes: **231.739773s**
  (3:51.74); scheduler includes Python startup/teardown.
- Log: `logs/pdo_ols_kaiser_runtime-30127.out`.
- Outputs: `results/pdo_ols_kaiser8_native_screening_bootstrap500/`.
- Unfiltered PDO, d18O only, screening/calibration1900–2000, p<=.1 and |r|>=.2.
  Same read-only raw Dod2k/PDO files and SHA256 as prior benchmarks.
- 65 proxies: Wood9, Coral30, GlacierIce14, Speleothem12.
- 412 candidate coverage NESTs, 374 accepted, 30 internal CE/RE exclusions,
  8 insufficient annual-core exclusions. Every accepted coverage NEST has
  exactly500 successful regressions: **187000 total NEST x run members**.
- 444 accepted resolution layers; nine native proxies actually retained in
  window PCA, all Speleothem in this screened network:
  `iso2k_1864`, `iso2k_396`, `iso2k_873`, `sisal_21.0_9`, `sisal_289.0_189`,
  `sisal_471.0_314`, `sisal_506.0_325`, `sisal_547.0_345`, `sisal_766.0_420`.
- Annual longest contiguous output **552–2011**. All saved years have finite
  median and ordered nonzero-width quantiles. Pooled yearwise members1000–185500.
- Across accepted annual/window layers, median internal CE0.399163 and
  RE0.467924; smallest layer-median CE0.101893/RE0.161527, strictly passing.
- Full-network apparent observation fit: **r0.79258069**, SD ratio0.69439977.
  Diagnostic10-year low-pass r0.89005174, SD ratio0.77264554. Observations and
  proxies are not filtered for fitting. No external sensitivity was run.
- 5–95% interval width: median1.327178 PDO units, min0.238553, max7.741393.
  Main median range[-2.364556,2.719255]; q05 minimum-8.083670, q95 maximum6.475228.
  Wider early intervals reflect the empirical reconstruction-member spread.
- One Coral year interpolated under the existing bounded policy; no native
  stone/ice interpolation or endpoint extrapolation. Compact outputs5630159B;
  no NEST matrices, workbooks, plots or individual-member files saved.
- 76 unit tests pass; completed-output audit and extra count/quantile/gate/
  PCA-scope checks pass.

## Comparison and interpretation

| Configuration | Screened | Annual output | Apparent observation r | SD ratio | Scheduler wall |
| --- | ---: | --- | ---: | ---: | --- |
| Prior OLS/Kaiser, no ensemble, old low-resolution screen | 64 | 927–2011 | .799321 | .752405 | 9:21 |
| Restored500-run NEST ensemble and native-window screen | 65 | 552–2011 | .792581 | .694400 | 3:57 |

Correlation differs by -0.006740, not a collapse; amplitude remains damped.
Per-NEST training variance matching before fusion and pooled median does not
guarantee final SD=observed SD. Report this instead of claiming final variance
was forced to unity. Network, PCA fitting scope, internal validation protocol
and final estimator changed, so this is **not** an identical-result speedup
or an isolated Bootstrap-effect comparison.

The efficient implementation reuses NEST PCA scores, solves small centered
least-squares regressions, vectorizes skill/amplitude operations, reuses native
window operators with sparse solves, and computes exact pooled quantiles in
year chunks. Numerical tests compare affine maps with the existing calibrator
and native ensemble fusion with member-by-member scalar fusion. Raw inputs
and the user-managed mybase environment are unchanged.

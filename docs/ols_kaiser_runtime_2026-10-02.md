# User-selected OLS/Kaiser PDO runtime benchmark

This is an experimental parameter baseline, not a replacement of the frozen
v0.6.0 release and not an exact-result engineering-speedup benchmark.

## Configuration and timing

Full raw-input PDO main reconstruction on one CPU and one numerical-library
thread on c061, using the existing read-only raw Dod2k database and unfiltered
observed PDO. Screening remains d18O only in Wood, Coral, GlacierIce and
Speleothem archives, 1900–2000, adjusted p <= 0.1 and absolute r >= 0.2.

User changes:

- Kaiser eigenvalue > 1 retention, with a hard maximum of 8 PCs and existing
  native-window sample-size safeguards;
- ordinary multiple linear regression (`regression="ols"`), no Ridge alpha
  search;
- the existing contiguous internal folds, approximately 2/3 calibration and
  1/3 validation;
- both internal median CE and RE strictly > 0.1, using
  `internal_ce_re_comparison="gt"`; no automatic relaxation;
- only two internal amplitude candidates, `variance_observation` and
  `variance_max_proxy_nest`, after the regression structure is fixed.

Coverage NESTs and native 2–10-year resolution-sub-NESTs are retained. Bootstrap,
external/network sensitivities, figures and the separate >10-year variational
module are disabled, as in the previous timed main run. The two amplitude
references are internally selected per model, not two separate final products.
Existing per-NEST calibration and densest-availability-tier semantics are
unchanged; there is no final-output variance-forcing operation.

Slurm job 30037 completed with exit 0 on 2026-10-02, 21:45:11–21:54:32 China
time. Python PID was 3823937, source revision
`916b16ea3184802452dceeaa14b06d77e1fee508` on `codex/ols-kaiser-runtime`.
Total job wall time is **9:21**. The full `run_pipeline` interval, including
reading, screening, reconstruction and compact output saves, is
**553.184320 seconds (9:13.18)**. Total CPU time is 9:10.730; maximum resident
memory is 302,988 KiB. The historical frozen full-search job took 60:40:
84.59% less job wall time, or approximately 6.49 times faster, for the new
scientific configuration.

## Results and required qualifications

| Quantity | Frozen full search | New OLS/Kaiser baseline |
| --- | ---: | ---: |
| Job wall time | 60:40 | 9:21 |
| Screened d18O proxies | 64 | 64 |
| Candidate coverage NESTs | 397 | 397 |
| Accepted coverage NESTs | 370 | 292 |
| Attempted resolution sub-NESTs | 1,029 | 810 |
| Accepted resolution sub-NESTs | 934 | 639 |
| Saved candidate summaries for accepted models | 43,796 | 2,793 |
| Apparent observation correlation, 1900–2000 | 0.7759349 | 0.7993206 |
| Final SD / observed SD, 1900–2000 | 0.6439354 | 0.7524049 |
| Longest contiguous annual output | 552–2011 | 927–2011 |

New overall accepted-layer internal median CE is 0.3670655 and RE is
0.4862737. Minimum accepted-layer median CE is 0.1024607 and RE is 0.1054751:
all satisfy the user's strict >0.1 thresholds. These summaries combine 292
coverage models and 639 resolution models; they do not require every individual
fold to exceed 0.1 or impose an external-validation veto.

The exact same 397 coverage specifications and source-input hashes are
verified. Nevertheless, PC rule/cap, regression, thresholds and amplitude
candidates changed, so model acceptance and predictions changed. Fewer coverage
models proceed to resolution fitting. Do not attribute all time savings solely
to removal of alpha tuning, or call this a result-preserving acceleration.

The new output has 1,085 consecutive annual estimates. Five speleothem proxies
and one ice proxy remain in window-scale PCA: iso2k_1626, iso2k_1864,
sisal_226.0_149, sisal_573.0_357, sisal_766.0_420 and sisal_800.0_458.
Only one Coral year is filled under the previously declared bounded policy;
none of these native-resolution records is interpolated. The remaining SD
ratio below one is consistent with per-NEST calibration before fusion and
combination, not a claim of final-output variance matching.

The reported observation correlation is a calibration fit, not independent
validation. No bootstrap uncertainty or external sensitivities were estimated
in this timed run. Compact outputs occupy approximately 6.0 MB, without NEST
matrices or Excel workbooks. All 65 tests and the strict completed-output audit
pass. No package installation, dataset modification or figure export occurred.

## Reproducibility

- Runner: `examples/run_pdo_ols_kaiser_runtime.py`.
- Submission: `sbatch examples/run_pdo_ols_kaiser_runtime.sbatch`.
- Log: `logs/pdo_ols_kaiser_runtime-30037.out`.
- Output: `results/pdo_ols_kaiser8_strict010_two_variances_timing/`.
- Complete configuration, PID, revision, status and pipeline timing:
  `runtime.json` inside that output directory.
- Audit:
  `PYTHONPATH=src python examples/audit_explicit_nest_outputs.py results/pdo_ols_kaiser8_strict010_two_variances_timing`.

The runner refuses to overwrite a nonempty output directory. Supply a new
`--output-directory` when repeating this measurement. The frozen release is
unchanged; no new release was published from this experiment.

# Changelog

## 0.6.0 — explicit in-memory and multiresolution NESTs

- Restored the original coverage-defined NPCR sequence as the primary method:
  build unique longest NESTs, fit PCA/PCR independently in each NEST, apply
  internal median CE/RE thresholds, and combine accepted reconstructions.
- Removed the former Excel hand-off and per-NEST matrix files. Added the
  compact `nest_summary.csv` audit instead.
- Prepared the screened proxy matrix once and sliced it in memory for each
  NEST, eliminating repeated annualization and interpolation-policy work while
  keeping every NEST's PCA, regression, and CE/RE fit independent.
- Limited candidate fitting to PC scores from training years, avoiding repeated
  out-of-calibration score solves. A paired full-matrix/optimized test also
  covers duplicated bootstrap years and verifies identical full predictions.
- Added 2/3/5/10-year resolution sub-NESTs. Coarser proxy observations join
  PCA and regression together with annual proxies and targets aggregated onto
  identical native windows; proxy gaps are never filled to annual values.
- Fused accepted sub-NEST window predictions into the annual state with an
  explicit window observation operator while retaining annual within-window
  variability.
- Applied the explicit NEST structure to the directed external holdouts and
  optional proxy-network sensitivity experiments. Kept the v0.5 unified
  native-missing model as `method="native_missing"`.
- Restored the optional legacy common-reference proxy z-score as an in-memory
  pre-NEST transform (1950–2000 in the WNPSM example), while retaining each
  NEST's independent training-period PCA standardization. Outer holdouts limit
  the common reference to their calibration years.

## 0.5.0 — longest continuous annual reconstruction

- Made the final product the longest contiguous interval with a finite annual
  reconstruction instead of retaining disconnected annual fragments.
- Added bounded interior linear interpolation for explicitly allowed
  near-annual archives. The default permits Wood and Coral only, limits gaps
  to two years, and never extrapolates record endpoints.
- Kept screening on observed proxy values and retained native timing for ice,
  speleothem, and other archives.
- Added `proxy_interpolation_audit.csv` plus native/interpolated availability
  counts so every filled proxy-year remains identifiable.
- Retained `interpolation="none"` as an explicit reproducibility and
  sensitivity option.

## 0.4.2 — main-result evidence and supplementary robustness

- Restored the WNPSM/PDO interpretation: main-result evidence is the complete
  screened-network correlation with observations plus median CE/RE from
  internal NPCR construction.
- Moved contiguous segment tests to supplementary sensitivity status. They no
  longer create pass/fail or strong/weak labels for the main reconstruction.
- Replaced compulsory early/middle/late holdouts with two directed edge tests:
  later 2/3 to early 1/3 and early 2/3 to late 1/3.
- Added both fixed-full-proxy-grid and fold-rescreened variants, with
  correlation as the declared external assessment metric.
- Added `primary_reconstruction_summary.csv` and separate validation-mode CSVs.
- Changed internal CE/RE thresholding from the worst inner fold to the
  WNPSM-like median inner-fold CE and RE; thresholds remain configurable.

## 0.4.1 — two-stage PCA/amplitude selection

- Restored an exact configurable Kaiser (`eigenvalue > threshold`) mode and
  added `kaiser_cv`, which uses Kaiser only as a contiguous-validation bound.
- Split model selection into structural and amplitude stages so variance
  fidelity cannot change PC count, regression family, or regularization.
- Added PCA eigenvalues, Kaiser count, selection-stage audit fields, and
  fold means/minima/maxima; CE/RE gates now require every outer fold to pass.

## 0.4.0 — frozen method release

- Added inner-fold automatic selection among no adjustment, observation
  variance, maximum-proxy-network variance, and dynamic variance calibration.
- Added smooth network-dependent amplitude scaling based on the effective
  weighted proxy count, with shrinkage and bounded slopes.
- Added proxy measurement-error reliability weights and correlation-gated
  spatial/site/source redundancy groups in weighted PCA and PC scoring.
- Added `proxy_weights.csv`, effective-network availability, selected
  amplitude metadata, and dynamic slope knots to compact audit outputs.
- Retained native missingness, no proxy interpolation, selection-aware outer
  validation, and no saved NEST workbooks.

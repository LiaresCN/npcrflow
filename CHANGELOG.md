# Changelog

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

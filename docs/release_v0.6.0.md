# NPCRFlow v0.6.0

Restores the original NPCR structure: unique longest coverage NESTs, independent
PCA/PCR and internal CE/RE in each NEST, followed by combination. NEST matrices
remain in memory; no Excel workbooks or bootstrap members are saved.

Native 2/3/5/10-year records participate in window-scale PCA and regression
together with annual proxies and observations aggregated onto matching windows.
Accepted window predictions constrain the annual result without interpolating
ice, speleothem or sediment records into annual observations.

The unfiltered-PDO/d18O benchmark selects 64 proxies, fits 397 coverage NESTs
(370 accepted) and 1,029 resolution sub-NESTs (934 accepted), and produces
1,460 consecutive annual values over 552–2011. Observation-period correlation
is 0.7759 and SD ratio 0.6439. The matched no-fusion ablation gives 0.7608 and
0.6247. These are apparent calibration fits, not independent validation.
Amplitude remains damped. Kaiser-CV, Kaiser and variance-retention results are
provided as same-network references, not replacements selected by apparent r.

Fold/PC score caching reduces the complete one-CPU PDO runtime from 73:39 to
60:40 (17.6%). Final annual predictions have zero difference; model selection,
CE/RE and compact audits match. Sixty-two tests pass; Python 3.10–3.12 are
covered by GitHub CI. The wheel's 18 source modules match the verified checkout.

This engineering benchmark disables bootstrap and external sensitivities;
their independently configurable functionality remains available. A zero-
bootstrap run does not supply bootstrap uncertainty intervals. Screening,
calibration periods, seasons, detrending, optional common-reference
standardization, PCA rules, regression, amplitude calibration, CE/RE thresholds,
resolution layers and sensitivity settings remain front-loaded configuration.

Raw datasets, large generated results, caches and credentials are not included.
See `README.md`, `docs/configuration.md` and `docs/validation_results.md` for use
and reproducible evidence.

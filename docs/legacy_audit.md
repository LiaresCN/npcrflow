# Audit of the WNPSM and PDO notebook workflow

Source code reviewed:

- `/share/home/lrs/WNPSM/筛选proxy_for rec.ipynb`
- `/share/home/lrs/WNPSM/npcr-wnpsm-era5.ipynb`
- `/share/home/lrs/codex/projects/wnpsm_npcr/npcr_pipeline.py`
- `/share/home/lrs/jgra_recPDO_code/jgra_code_public/V1_sub/recPDO_code_V1.ipynb`
- `/share/home/lrs/jgra_recPDO_code/jgra_code_public/V1_sub/NPCR重建过程文件/NPCR重建PDO-bybothproxy-0608+coral-.ipynb`
- `/share/home/lrs/jgra_recPDO_code/返修/NESTpcr_v1.ipynb`
- `/share/home/lrs/jgra_recPDO_code/返修/proxy前处理-corr-石笋.ipynb`
- `/share/home/lrs/jgra_recPDO_code/返修/返修二/NPCR_new_rec0.2-1.ipynb`

## Correctness issues

1. Proxy series were floored to integer years, linearly interpolated, and in
   several notebooks extrapolated before screening.  This can create apparent
   observations and alter low-frequency variance.
2. Random individual years were used for train/validation splits despite strong
   autocorrelation.  PCA and standardization were usually fit before the split,
   leaking validation-period proxy structure into the fitted representation.
3. The notebooks labelled `1 - var(error)/var(validation)` as RE.  Standard RE
   uses squared error relative to the calibration mean.  CE used either the
   validation mean or median in different versions.  One early PDO notebook
   multiplied CE by a random perturbation.
4. Screening significance in early PDO code used nominal Pearson p values; the
   WNPSM screening later added lag-1 effective sample size but duplicated the
   same code in multiple cells.
5. PCA/regression output was round-tripped through many multi-sheet Excel files.
   The same nests and large bootstrap members were repeatedly saved, increasing
   the chance of stale-file mixing.
6. The reconstruction endpoint was repeatedly hard-coded at 2000, even where
   observations or proxies continued later.
7. Detrending each reconstructed nest before combination removed the very
   low-frequency behavior that low-resolution archives should help constrain.
8. RF/XGBoost/MLP trials reused the same random-year design and fixed
   hyperparameters.  Their in-range scores therefore do not demonstrate safer
   pre-instrumental extrapolation.
9. The smaller legacy PDO tree network was not produced from all tree-ring
   measurements. `proxy前处理-corr.ipynb` first kept only coral, tree,
   speleothem, and glacier-ice rows whose `paleoData_variableName` was `d18O`.
   It then required at least 50 observations, rejected coarse early spacing,
   removed duplicate processed sheets, and used an effective-DOF significance
   test. The intermediate `filtered_proxydb.pkl` has only 86 records (41 tree,
   40 coral, 5 speleothem), all labelled `d18O`, whereas the 2026 raw Dod2k file
   contains 1,782 ring-width records. Allowing all proxy variables therefore
   created hundreds of tree-ring candidates that were never in the old PDO
   candidate pool.

## Design response

`npcrflow` rejects the legacy linear-interpolation option; any non-`none`
request is a hard error. The path uses native missingness, calibration-contained
preprocessing, nested contiguous validation, standard RE/CE, deterministic
seeds, in-memory results, automatic end years, regularized linear alternatives,
and compact sensitivity tables.

# Validation status

> **Interpretation update (v0.4.2):** this file preserves engineering results
> produced through v0.4.1. Continuous holdouts are now supplementary
> robustness sensitivities, not pass/fail criteria for the main
> reconstruction. Main-result evidence is the complete screened-network
> correlation with observations plus CE/RE from internal NPCR construction.
> The current standard uses only the two directed 2/3-to-1/3 edge tests and no
> compulsory middle holdout. Older references below to “primary outer
> validation,” negative CE failure, or three-fold gates are historical.

All scores below compare the reconstruction directly with the instrumental or
reanalysis observation target. Published reconstructions are not validation
references. The results distinguish two different questions:

1. **Conditional validation:** does the regression work after a proxy network has
   already been selected using the full calibration interval?
2. **Selection-aware validation:** does the entire procedure generalize when
   proxy screening is repeated using only each outer fold's training target?

Through v0.4.1 the second question was treated as the primary engineering
validation. From v0.4.2 both questions are supplementary robustness tests; the
main evidence is defined in the update above.

## PDO

### Version 0.6.0 explicit-NEST verification

The annual-only ablation uses the same unfiltered PDO, d18O scope, screening
and calibration period as the primary mixed-resolution run, with native
resolution restricted to at most 1.5 years. Its 56 proxies form 268 unique
coverage NESTs, all accepted by the declared internal median CE/RE thresholds.
In 1900–2000 the apparent observation correlation is `r=0.753323`, SD ratio
`0.621563`, and RMSE `0.621631`. Across accepted NESTs, internal median CE is
`0.396914` and RE is `0.444483`. The saved annual interval is 552–2011,
containing 1,460 consecutive finite values. Total compact outputs are about
5.4 MB; no predictor matrices, NEST workbooks, or bootstrap members are saved.
This run disables bootstrap and external sensitivities to isolate the NEST
and resolution comparison.

A stricter fusion ablation retains all 63 reconstruction-eligible screened
records and the same 397 coverage NESTs as the mixed-resolution run, but sets
the resolution constraint weight to zero. It accepts 370 coverage NESTs and
gives apparent `r=0.760785`, SD ratio `0.624688`, and RMSE `0.614677` against
1900–2000 observations. The verified mixed-resolution implementation
accepts the same 370 coverage NESTs plus 934 resolution sub-NESTs and gives:

| Same 397-NEST coverage grid | Observation r | SD ratio | 10-year low-pass r | Low-pass SD ratio |
| --- | ---: | ---: | ---: | ---: |
| Resolution fusion disabled | 0.7608 | 0.6247 | 0.8556 | 0.6619 |
| Native-window PCA/PCR fusion | 0.7759 | 0.6439 | 0.8770 | 0.6893 |

This is a modest apparent calibration improvement, not an independent
validation or universal low-resolution benefit. Both annual products are
552–2011. Six native records (five speleothems and one ice record) are retained
in accepted window-scale PCA fits; native proxy values are not interpolated.
The accepted annual NESTs have median internal CE `0.393042` and RE `0.446296`;
accepted resolution sub-NESTs have median CE `0.487802` and RE `0.581601`.
The compact outputs total about 24.3 MB, without NEST matrices or workbooks.
Only one Coral proxy-year is interpolated, with no endpoint extrapolation;
native resolution-layer records remain unfilled. The completed single-CPU
pre-score-cache run takes 1:13:39 and uses a maximum of about 384 MB RAM.
Its paired score-cache run completes in 1:00:40 with about 393 MB maximum RAM.
Both use one CPU on c061, identical raw-input hashes and all settings except
the output directory. The complete wall time falls by 779 seconds (17.6%).
The full reconstruction, NEST summary, model-selection table, proxy weights,
screening, interpolation audit and observation-statistics tables match within
`1e-10`; maximum annual prediction difference is exactly zero. This is one
controlled complete PDO benchmark, not a universal speed guarantee. The
annual-only ablation changes the coverage grid and cannot alone isolate fusion.

The earlier development run (`pdo_v060_explicit_nests_smoke`, four allocated
CPUs) gave `r=0.775248` and 885 accepted resolution sub-NESTs before the final
finite-overlap/window engineering corrections and audit changes. It is a
historical diagnostic, not the final numerically paired cache benchmark.

All supplementary PCA references have completed on the identical screened
network and raw-input hashes:

| PCA selection | Accepted coverage NESTs | Observation r | SD ratio | Annual interval |
| --- | ---: | ---: | ---: | --- |
| Kaiser + internal CV | 367 | 0.7669 | 0.6331 | 851–2011 |
| Kaiser | 360 | 0.7884 | 0.6246 | 851–2011 |
| 90%-variance retention | 362 | 0.7930 | 0.6274 | 552–2011 |

All respect the declared five-PC safety cap; Kaiser is therefore not an
unlimited reproduction of the former notebook. Internal CE/RE gates pass for
all retained layers in these references. They do not replace the predeclared
blocked-CV primary method merely because apparent correlation is larger.

The score-cache engineering benchmark
(`examples/benchmark_score_cache.py`, reference `2615aa1`) uses 40 synthetic
missing-data proxies, five candidate PC counts, six Ridge penalties and
101 calibration years. On one c060 CPU, three repeats reduced median tuning
time from 3.12 to 2.30 seconds, about 26%, and score calculations from 171 to
26. Model-selection tables, internal CE/RE and full-period predictions match
within `1e-12`; maximum prediction difference was zero. This timing measures
one tuned PCR, not screening, uncertainty, external sensitivities or the
complete 397-NEST PDO workflow. Its complete matched PDO result is reported
above: 17.6% less wall time with the same final reconstruction and model choices.

`examples/audit_explicit_nest_outputs.py` reads completed outputs without
changing them. It checks unique coverage networks, internal CE/RE gates,
actual retained native PCA members, annual continuity, interpolation eligibility,
and absence of NEST workbooks. Its optional paired comparison checks final
reconstructions, NEST audits, model-selection tables and proxy weights.

The independent controlled check in
`examples/run_synthetic_resolution_nests.py` pairs ten annual tree proxies
with one 3-, 5-, or 10-year native stalagmite. Trees contain the annual
component but no injected slow component. Only the first 100 target years
are supplied to calibration or tuning; the last 100 years are held out.
The proxy network is predeclared, so this isolates NEST fitting and fusion
after screening rather than testing the screening procedure. Three seeds
give the following held-out medians:

| Network | Annual r | SD ratio | 10-year low-pass r | Low-pass SD ratio |
| --- | ---: | ---: | ---: | ---: |
| Ten annual trees | 0.6596 | 0.6895 | 0.0790 | 0.1251 |
| + one 3-year stalagmite | 0.7340 | 0.6930 | 0.6703 | 0.1654 |
| + one 5-year stalagmite | 0.7037 | 0.6764 | 0.4786 | 0.1362 |
| + one 10-year stalagmite | 0.6590 | 0.6742 | 0.0752 | 0.1254 |

This demonstrates that a native-resolution sub-NEST can improve held-out slow
covariation while its proxy remains sparse. Amplitude remains damped, and the
10-year case does not improve with only about ten calibration samples. It is
method-capability evidence under a known chronology and proxy system, not a
claim of equivalent skill for a real stalagmite or PDO network.

### Version 0.5.0 longest continuous annual output

With bounded Wood/Coral interpolation enabled (maximum two-year interior gap,
no endpoint extrapolation), the saved PDO result is the longest continuous
annual interval: 538–2012, containing 1,475 consecutive finite annual values.
Only two proxy-year values are interpolated, both from two Coral records; no
Wood, GlacierIce, or Speleothem value is filled in this run. The first and last
saved years each retain at least two native model proxies.

The scientific fit is effectively unchanged: main `r=0.7782`, SD ratio
`0.7154`, internal median `CE=0.5085`, and internal median `RE=0.5167`.
Version 0.4.2 retained the full 0–2013 index even though its finite estimates
were split into 102 separate fragments; its longest continuous finite fragment
was already 538–2012. Version 0.5.0 therefore makes the final product explicit
and compact rather than claiming a disconnected maximum span.

### Version 0.4.2 main evidence and robustness sensitivities

The complete screened-network reconstruction is unchanged from the recommended
v0.4.1 4-PC result: observed-period `r=0.7781`, effective-DOF
`p=1.61e-12`, and SD ratio `0.7153`. During internal NPCR construction, the
selected candidate has median `CE=0.5086` and `RE=0.5168`, above the declared
zero thresholds. These quantities form the main-result evidence.

The two directed fixed-full-network correlation sensitivities are
`r=0.7698` for the early third and `r=0.7167` for the late third. Repeating
proxy screening inside each calibration segment gives `r=0.5605` and
`r=0.6989`. They are saved for reproducibility and interpreted only as
supplementary robustness; their CE/RE values do not accept or reject the main
reconstruction.

The current engineering test starts directly from the global raw Dod2k
database, uses the **unfiltered** ERSSTv5 PDO target, and restricts the candidate
pool to `d18O` from wood, coral, glacier ice, and speleothems. The full
1900–2000 screen selects 66 records (31 coral, 17 glacier ice, 9 speleothem,
and 9 wood); all 66 enter the primary core PCR layer. The reconstruction has
finite values from 0–2013, rather than ending at 2000.

The apparent 1900–2000 fit is `r = 0.782`, `RE = CE = 0.602`; it is not
independent because those observations fitted the final model. Three contiguous
outer folds each withhold approximately one third of the observation interval
(34/34/33 years), repeat screening and all model fitting using the remaining
two thirds, and give median `r = 0.560`, `RE = 0.245`, and `CE = 0.232`. All
three fold CE and RE values are positive. The early 1900–1933 holdout gives
`r = 0.523`, `RE = 0.245`, `CE = 0.232`; the late 1968–2000 holdout gives
`r = 0.691`, `RE = 0.464`, `CE = 0.459`. The explicit bidirectional edge tests
match these outer folds exactly.

Randomly deleting 20% of the selected network is also stable across ten
repeats: the median-of-fold medians is `r = 0.753`, `RE = 0.466`, and
`CE = 0.459`. The final reconstruction retains about 46.5% of observed
1900–2000 variance, so amplitude damping remains a limitation even though the
contiguous validation is positive.

The version 0.1 whole-trajectory native-window adjustment failed the same
two-thirds/one-third test: it reduced the
outer medians to `r = 0.005`, `RE = 0.017`, `CE = -0.013` and made the late
edge negative. The unadjusted core scores were positive, so this degradation is
attributable to that superseded adjustment rather than core NPCR.

## Version 0.2.1 method checks

These checks test the new options rather than select a PDO result. Optional
single-proxy, network-deletion, and directed split sensitivities remain
available but were disabled so the complete selected network is the focus.

All reported ensemble results now require at least 80% successful moving-block
members; the current runs contain all 50/50 members. A duplicate-year alignment
bug that had silently reduced early v0.2 runs to one member has been corrected.

Training-only OLS amplitude calibration gives outer-fold median `r = 0.523`,
`RE = 0.254`, and `CE = 0.241`, versus raw-prediction `RE = 0.245` and
`CE = 0.232` from the same fold models. Median held-out SD ratio rises from
`0.347` raw to `0.374`; the change is deliberately modest. Training-period
variance matching raises median held-out SD ratio to `0.499` and median CE to
`0.265`, but its fold 5th-percentile CE is only `0.014`. In the apparent final
50-member median it reaches SD ratio `0.988` while reducing CE from
approximately `0.543` raw to `0.480`, confirming that forced equal variance can amplify
unpredictable variability. Neither amplitude method is silently preferred;
`none`, `ols`, and `variance` remain explicit choices with raw/calibrated
outputs.

The corrected routing lets records at or below 10-year resolution participate
directly in pairwise PCR at their observed years and leaves all intervening
years missing. Only slower records enter the low-pass layer. The full PDO
selection therefore splits into 64 native-PCR and 2 low-pass-window records.
With amplitude calibration disabled, all-native PCR has apparent `r=0.782`, SD
ratio `0.682`, and outer median `r=0.560`, SD ratio `0.376`. Before applying the
low-pass increment, the hybrid core has apparent `r=0.785`, SD ratio `0.678`.
After the >10-year low-pass adjustment these become `r=0.774`, SD ratio `0.645`;
outer medians remain `r=0.560`, SD ratio `0.376`, but late-fold `r` falls from
`0.691` to `0.583`. The low-pass increment therefore remains experimental and
off by default, while native sub-decadal PCR routing is retained.

## Version 0.3.0 low-frequency ablation

The duplicate audit now recognizes database copies that differ only by numeric
storage rounding while requiring identical timestamps, archive/proxy identity,
and location. This removes 175 rows in 144 groups from the 2,815-row raw table,
including 11 rounding-equivalent cross-database copies. In particular,
`iso2k_821` and `pages2k_1293` are one Antarctic ice-core series, not two
independent >10-year constraints. The corrected d18O selection has 64 unique
records: 56 at or below 1.5-year resolution, seven between 1.5 and 5 years,
none between 5 and 10 years, and one near 11.6 years.

Adding the seven 1.5–5-year records directly to native-missing PCR improves the
outer-fold median annual correlation only slightly (`0.561 -> 0.563`), but the
10-year low-pass correlation rises from `0.783` to `0.797` and its SD ratio
rises from `0.231` to `0.282`. The strongest change is in the middle holdout:
annual `r` rises from `0.317` to `0.560` and 10-year `r` from `-0.034` to
`0.465`. No interpolation is used; the added values occur only at their native
annualized timestamps.

The single unique >10-year record adds no outer-fold benefit when placed
directly in PCR. Routing it through the experimental native-window variational
layer degrades the late holdout from annual `r=0.697` to `0.570` and from
10-year `r=0.869` to `0.589`. This layer therefore remains off for PDO.

Because a 34-year one-third holdout is too short to establish a 20-year cycle,
a separate bidirectional 50/51-year sensitivity was run. When 1900–1949 is
used for calibration, adding 1.5–5-year records improves validation over
1950–2000 from annual `r=0.681` to `0.729`, 10-year `r=0.740` to `0.829`, and
20-year `r=0.784` to `0.909`; 10-year CE changes from `-0.068` to `0.239` and
20-year CE from `-0.289` to `0.125`. In the reverse direction, annual
correlation is essentially unchanged (`0.651 -> 0.648`), while 10-year
correlation rises `0.883 -> 0.890` and 20-year correlation `0.923 -> 0.947`.
Amplitude does not improve consistently in the reverse direction, so the
evidence supports better low-frequency phase/covariability, not a solved
amplitude or mean-bias problem.

Under the same unique all-native network, PLS gives the best current outer
annual median (`r=0.574`, SD ratio `0.394`, CE `0.268`) compared with Ridge
(`0.563`, `0.376`, `0.257`) and ElasticNet (`0.563`, `0.386`, `0.265`). These
are modest improvements. Random forest raises the apparent correlation to
`0.843` but reduces outer annual `r` to `0.427` and outer 10-year `r` to
`0.491`, a clear overfit; it remains an optional negative-control model rather
than a recommended reconstruction method.

## Version 0.3.1 multiresolution safeguard

The native-window variational layer now selects its constraint weight and
low-pass period using contiguous folds contained entirely inside each current
training interval. Weight zero is an explicit candidate. The selection
objective prioritizes the smaller of annual and 10-year CE, with a small
10-year-correlation term; no outer observation enters this choice.

For the one unique >10-year PDO record, the full-calibration tuner selects
weight `0.0`. Its baseline candidate has median inner annual/10-year CE of
`0.510/0.559`. Every nonzero candidate is worse; for example, weight `1.0` at
10 years gives `0.453/0.501`. The final apparent fit therefore stays at
`r=0.787`, 10-year `r=0.910`, and 20-year `r=0.950`, exactly the native-PCR
core rather than the degraded fixed-weight adjustment.

All three selection-aware outer folds also choose zero influence: the early
and middle folds contain no eligible >10-year record after training-only
screening, and the late fold rejects the available record during inner tuning.
Final and core predictions are consequently identical. The late-fold annual
and 10-year correlations remain `0.697` and `0.869`, instead of falling to
`0.570` and `0.589` under the former forced weight. Synthetic unit checks
separately verify that the same tuner chooses a nonzero weight for an
informative low-resolution record and zero for a noisy one, without filling
any proxy value.

## Version 0.3.2 PDO cycle audit

The 10- and 20-year low-pass correlations above describe slow covariation but
do not prove that a PDO spectral peak is reconstructed. A separate 10–30-year
frequency-domain audit now reports peak period, band amplitude, and phase-aware
spectral similarity. It requires at least 45 consecutive years, so the normal
34-year outer thirds are correctly unavailable and the bidirectional 50/51-year
split is used as the independent sensitivity.

Over the full 1900–2000 fit, the observed and Ridge reconstructions both peak
at `25.25` years. Adding 1.5–5-year records raises phase similarity from
`0.730` to `0.832`, but lowers band amplitude ratio from `0.540` to `0.511`.
This result is apparent rather than independent.

With 1900–1949 calibration and 1950–2000 validation, the observed peak is
`25.5` years but both networks peak at `12.75` years. Adding 1.5–5-year records
still improves band amplitude ratio `0.599 -> 0.688` and phase similarity
`0.671 -> 0.802`, but it does not recover the dominant period. Reversing the
split, the <=1.5-year network reproduces the observed `25`-year peak, whereas
the <=5-year network shifts it to `16.67` years and reduces phase similarity
`0.748 -> 0.700`. The low-resolution contribution therefore improves some
low-frequency covariance but does not robustly recover the PDO cycle in both
directions.

PLS partly improves the reverse validation relative to Ridge: annual
correlation rises `0.648 -> 0.666`, 10–30-year amplitude ratio `0.209 ->
0.265`, and phase similarity `0.700 -> 0.750`. It still retains the incorrect
`16.67`-year peak; moreover, its apparent full-period peak is `12.625` years
instead of the observed `25.25`. ElasticNet retains the apparent `25.25`-year
peak but is weaker in the independent forward split. Hence no advanced
regressor is uniformly superior for PDO-cycle recovery. A spectral tuning
objective is not added because the leakage-safe inner folds are too short to
resolve 10–30-year periods independently.

## Version 0.3.3 automatic regression-family audit

`regression="auto"` now compares Ridge, PLS, and ElasticNet using only
contiguous inner blocks of the current calibration interval. The selected
concrete family is reported and then held fixed during low-frequency tuning
and bootstrap refitting. Random forest is excluded from the default candidate
set because version 0.3.0 showed clear external overfitting. Proxy values are
still retained only at their native observation years.

On the unfiltered PDO d18O test, automatic family selection does not improve
the three one-third outer-fold medians. At the 1.5-year resolution limit all
three outer folds select Ridge, so results are identical. At the 5-year limit
only the middle fold selects ElasticNet; its annual `r/CE` changes only from
`0.559/0.192` to `0.560/0.194`, and the early and late folds remain Ridge.

The bidirectional 50/51-year sensitivity is more cautionary. With 1900–1949
calibration and the <=5-year network, inner validation selects ElasticNet, but
external 1950–2000 annual `r/CE` falls from Ridge's `0.729/0.329` to
`0.712/0.049`; 10–30-year phase similarity falls from `0.802` to `0.728`.
With 1950–2000 calibration, automatic selection chooses Ridge and exactly
reproduces the fixed-Ridge early validation. At <=1.5 years, the forward
ElasticNet choice raises annual correlation only `0.681 -> 0.685` while CE
falls `0.172 -> 0.148`; the reverse split again chooses Ridge.

Automatic model-family selection is therefore retained as an auditable
sensitivity option, not made the primary PDO estimator. The present evidence
supports regularized linear methods over random forest, but does not establish
PLS or ElasticNet as uniformly superior to Ridge. The limited, directional
low-frequency benefit of adding <=5-year records is a proxy-information effect,
not evidence that a more complex regressor recovered the PDO cycle.

## Version 0.3.4 native-window identifiability and amplitude scale

A controlled experiment now separates method capability from the limited real
PDO proxy network. The target contains a known 25-year component, while the
annual PCR predictors contain none of it. The first 100 years are available for
calibration and training-only parameter selection; the following 101 target
years are withheld. Only sparse 12- or 20-year proxy samples extend through the
validation interval. No sparse value is interpolated. Eighteen combinations of
resolution, 1/3/5 records, and three noise levels were run for 20 paired repeats
each. Every combination within a repeat uses the identical annual core.

This audit exposed two structural amplitude suppressors. Both the increment cap
and the core-anchor normalization had used only the core low-frequency standard
deviation. When the core lacked the target cycle, that scale approached zero:
the permitted increment collapsed while the core anchor became artificially
strong. Both terms now use the larger of the core scale and the low-frequency
scale estimated from the current calibration-only target. The latter is refit
inside every fold, so no withheld observation enters the solve. Weight
candidates now extend through 30.

Before assimilation, the paired core has median validation `r=0.275`, phase
similarity `-0.097`, band-amplitude ratio `0.075`, and identifies the correct
peak in only 20% of repeats. With three 12-year records, low/medium/high noise
gives median validation `r=0.940/0.863/0.718`, correct-peak fractions
`1.00/1.00/0.80`, phase similarity `0.983/0.894/0.812`, and amplitude ratios
`1.091/0.922/0.599`. With five 12-year records, the corresponding values are
`r=0.949/0.918/0.807`, correct-peak fractions `1.00/1.00/0.95`, phase
`0.980/0.944/0.856`, and amplitude ratios `1.053/0.839/0.550`.

One record is not enough for reliable amplitude: its 12-year amplitude ratios
remain `0.171/0.158/0.146`, even though peak identification improves modestly.
Multiple staggered 20-year records can still constrain a 25-year cycle in this
idealized setting: three low-noise records give `r=0.874`, phase `0.960`, and
amplitude ratio `0.832`; five give `0.906`, `0.975`, and `0.972`. These are
identifiability results under a known proxy-system relationship and chronology,
not proof that real records have equivalent information.

The real unfiltered PDO d18O audit remains conservative. It contains only one
independent record slower than ten years. Full-calibration tuning still selects
weight zero: the zero-weight objective is `0.602`, while the best nonzero
candidate reaches only `0.568`; larger weights degrade further. All three
one-third folds and both 50/51-year directions consequently retain the native
PCR result unchanged. The implementation can therefore restore a known cycle
and its amplitude when several independent low-resolution records contain that
signal, while rejecting the currently inadequate real PDO constraint.

## Version 0.3.5 amplitude-reference audit

The variance adjustment can now reference either the calibration observation
or the densest eligible proxy network. The latter is defined inside each
training fold as the highest proxy-count threshold that retains at least the
configured minimum number of years; no NEST file is created and no withheld
observation is used. The active legacy notebook code was internally
inconsistent: it calculated maximum-NEST statistics but scaled non-reference
NESTs to observations while leaving the reference NEST unscaled.

On the same unfiltered PDO d18O network with three contiguous one-third
holdouts, the results are:

| Variance setting | Median r | Median SD ratio | Median RMSE | Median RE | Median CE |
|---|---:|---:|---:|---:|---:|
| None | 0.561 | 0.379 | 0.754 | 0.264 | 0.252 |
| Observation | 0.561 | 0.523 | 0.737 | 0.299 | 0.287 |
| Maximum-proxy network | 0.572 | 0.352 | 0.831 | 0.209 | 0.185 |

The observation reference is preferable for this PDO case: it improves the
withheld amplitude ratio and median error/skill, though the withheld SD ratio
remains below one because the scaling coefficient is estimated only from the
training interval. The maximum-proxy reference is available as an internal
network-scale sensitivity, but it is not the recommended PDO default. A
positive affine adjustment does not directly change correlation for a fixed
model; small correlation differences here arise indirectly because amplitude
calibration participates in inner CE/RE-based model selection.

## Version 0.4.0 automatic, dynamic, and weighted method audit

Version 0.4.0 adds three fold-safe capabilities: automatic amplitude-model
selection, smooth effective-network-dependent variance calibration, and
proxy-error/spatial-redundancy weighting. All choices are fitted inside the
current training interval. Proxy values remain at their native timestamps.

On the unfiltered PDO d18O network, the automatic unweighted method retains
positive external skill and selects different amplitude treatments by outer
training interval. Its three-fold medians are `r=0.561`, `SD ratio=0.401`,
`RE=0.264`, and `CE=0.252`; the final full-calibration model selects dynamic
variance. This is essentially neutral relative to no amplitude adjustment and
less favorable than fixing observation-variance scaling in the earlier
development audit. Automatic selection is therefore a leakage-safe facility,
not a guarantee of improved performance on every target.

The redundancy audit identifies 12 downweighted entries in three principal
same-region/correlated groups: Palmyra, Fiji, and Vanuatu coral records that
appear in multiple source databases or closely related series. Their 64 raw
records have a combined weighted count of 56. Full redundancy weighting lowers
the PDO dynamic-variance medians to `r=0.501`, `RE=0.130`, and `CE=0.116`;
automatic amplitude selection cannot recover that loss (`r=0.499`, `RE=0.035`,
`CE=0.027`). This result is scientifically informative: part of the apparent
PDO skill depends on replicated regional coral information. Weighting remains
available and fully audited, but is disabled in the primary PDO template and
must be reported as an independence sensitivity rather than assumed to help.

## Version 0.4.1 two-stage PCA and amplitude selection

The legacy PDO/IPO and WNPSM notebooks used the Kaiser rule and retained every
standardized PCA eigenvalue greater than one. Applying that rule once to the
current 64-record native-missing PDO network retains 19 PCs. Its apparent
full-period correlation rises to `0.828`, but the three selection-aware outer
correlations are `0.406/0.071/0.619`; median `RE/CE` falls to `0.080/0.064`.
Likewise, a 90% cumulative-variance rule retains 20 PCs and gives outer median
`r=0.421`. Directly restoring the old threshold therefore overfits this global
network and is retained only as an auditable compatibility option.

`kaiser_cv` uses the 19-PC Kaiser count as an upper bound and chooses dimension
with inner contiguous folds. It selects seven PCs for the full model, giving
apparent `r=0.803`, while outer correlations are `0.561/0.448/0.697`. This is
safer than direct Kaiser but weaker in the middle block than the capped direct
blocked search.

The revised two-stage blocked method first selects the raw PCR structure and
only then compares amplitude mappings. It selects four PCs and no amplitude
adjustment. Apparent `r` recovers from the frozen automatic method's `0.741` to
`0.778`; the three outer correlations are `0.561/0.559/0.697`, with all fold
CE values positive (`0.252/0.192/0.469`). Relative to the frozen joint search,
the middle holdout improves from `r=0.412`, `CE=-0.013` to `r=0.559`,
`CE=0.192`. The new default therefore remains blocked validation, not direct
Kaiser. Kaiser eigenvalues/counts are reported as diagnostics, and amplitude
fidelity can no longer alter PC count, regression family, or regularization.

## WNPSM

Using the legacy notebook's effective 60–180°E raw-longitude subset selects 35
proxies and reconstructs 1080–2011. Its apparent fit to the ERA5-derived WNPSM
observations over 1940–2000 is `r = 0.653`, `RE = CE = 0.420`. Selection-aware
outer-fold medians are `r = 0.286`, `RE = -0.016`, and `CE = -0.043`; the
minimum non-negative skill gate fails.

## Interpretation

Random 20% proxy deletion measures robustness conditional on a chosen network;
it does not reproduce the uncertainty of selecting the network from the target.
For the primary unfiltered d18O PDO test it remains positive after deletion.
More generally, gaps between apparent, deletion, split, and
selection-aware results diagnose network and period sensitivity; they are not
a reason to tune CE/RE thresholds until all tests pass.

The positive engineering tests should not by themselves be presented as a
definitive new PDO reconstruction. Their purpose is to verify transparent,
fold-safe method behavior across amplitude and resolution choices.

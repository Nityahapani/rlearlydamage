# Phase 12 post-hoc audit

This audit changes no registered decision. Only completed tests are read by default.
Panel-wide actor Holm is additional, not the registered within-setting three-effect correction. Joint-scale TOST accounts for uncertainty in the estimated clean AUC.

| test | n | actor raw p | within-setting Holm | panel-wide Holm (post-hoc) | registered ±10% margin | critic fixed / joint TOST | Adam fixed / joint TOST |
|---|---|---|---|---|---|---|---|
| G1 cartpole-rflip | 8 | 0.0036499 | 0.01095 | 0.0072999 | 47.855 | 0.033716 / 0.034175 | 0.0077001 / 0.0081018 |
| G2 acrobot-rflip | 8 | 0.092881 | 0.27864 | 0.092881 | 10.354 | 0.92756 / 0.93302 | 0.51979 / 0.52499 |
| G3 si-actnoise | 8 | 3.3993e-05 | 0.00010198 | 0.00013173 | 7.989 | 4.2667e-06 / 1.5373e-05 | 5.4524e-07 / 1.466e-06 |
| G4 freeway-rflip | 4 | 0.00021687 | 0.0006506 | partial: not in tally | 3.804 | 0.96226 / 0.96373 | 0.035844 / 0.04187 |
| G5 si-obsnoise | 8 | 6.037e-06 | 1.8111e-05 | 3.0185e-05 | 7.989 | 1.0664e-05 / 1.985e-05 | 4.6891e-06 / 6.3926e-06 |
| G6 si-extinction | 8 | 3.2932e-05 | 9.8796e-05 | 0.00013173 | 7.989 | 1.1398e-06 / 2.5899e-06 | 1.0011e-06 / 2.006e-06 |

Gap-addendum hit count, all planned settings (G1 acute values had been seen): 1/1.
Untouched settings only (exclude G1): 0/0; no prediction was issued for intermediate relative gaps.
This prediction concerns a same-update gap, not persistence beyond lost time.

Exact 95% Clopper–Pearson interval for 1/44 non-recoveries: 0.0575%–12.021%. This assumes exchangeability under the tested protocol; one event does not establish a stable rare-event rate.

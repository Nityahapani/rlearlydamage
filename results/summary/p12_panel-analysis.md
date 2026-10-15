# Phase 12 — generalization panel: cross-setting summary (pre-registered)

PPO 2^3 actor x critic x Adam surgery at removal, fork reference, unmasked/undeficited CRN evaluation. Rules (LOG Phase 12 pre-registration): damage to localize iff total CI < 0; localization criterion met iff actor main-effect CI < 0 and actor has the largest Shapley share; critic/Adam TOST at ±10% of |CCC AUC| (earlier settings: ±3 return points). Paired 95% t-CIs; negative = harm.

| id | env | deficit | status | n | gap at removal (% of C) | total AUC | actor | critic | Adam | actor share | critic TOST p | Adam TOST p | delay-adj. AUC | delay-adj. end | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | SpaceInvaders | movement restriction | earlier (n = 12) | 12 | -37.0 [-45.1, -28.9]* (-63%) | -15.1 [-17.2, -13.0]* | -13.7 [-14.9, -12.5]* | -1.9 [-3.5, -0.4]* | +0.2 [-0.9, +1.3] | 90% (12/12) | 0.0814 | 0.0001 | – | – | REPLICATES |
| J3 | Asterix | movement restriction | earlier (n = 8) | 8 | -1.0 [-1.5, -0.5]* (-18%) | -0.1 [-0.5, +0.4] | -0.2 [-0.7, +0.3] | -0.0 [-0.2, +0.2] | +0.0 [-0.1, +0.2] | –% (–) | 0.0000 | 0.0000 | +0.5 [+0.0, +1.0] | +1.0 [+0.2, +1.7]* | NO DAMAGE TO LOCALIZE |
| J4 | SpaceInvaders | reward flip | earlier (n = 8) | 8 | -58.2 [-66.4, -50.1]* (-100%) | -32.3 [-40.7, -23.9]* | -32.8 [-36.9, -28.7]* | +1.2 [-2.5, +4.9] | -1.9 [-4.0, +0.2] | 100% (8/8) | 0.1487 | 0.1315 | -18.6 [-27.2, -9.9]* | -10.7 [-22.0, +0.6] | REPLICATES |
| J7 | SpaceInvaders (2x256) | movement restriction | earlier (n = 8) | 8 | -49.9 [-56.9, -43.0]* (-73%) | -19.7 [-23.7, -15.6]* | -15.7 [-17.6, -13.8]* | -2.5 [-4.5, -0.5]* | -0.4 [-2.6, +1.7] | 82% (8/8) | 0.3030 | 0.0126 | -5.1 [-8.3, -1.8]* | -2.4 [-7.8, +2.9] | REPLICATES |
| E2 | Asterix | reward flip | earlier (n = 8) | 8 | -3.7 [-4.6, -2.8]* (-68%) | -2.3 [-2.9, -1.7]* | -2.1 [-2.6, -1.7]* | -0.1 [-0.2, +0.0] | +0.0 [-0.2, +0.2] | 94% (8/8) | 0.0000 | 0.0000 | -1.7 [-2.1, -1.3]* | -1.2 [-1.7, -0.6]* | REPLICATES |
| G1 | CartPole-v1 | reward flip | complete | 8 | -448.2 [-502.1, -394.2]* (-98%) | -223.3 [-347.1, -99.5]* | -206.1 [-319.9, -92.3]* | -17.0 [-50.8, +16.8] | +18.7 [-3.0, +40.3] | 95% (8/8) | 0.0337 | 0.0077 | -196.6 [-322.8, -70.3]* | -91.8 [-262.4, +78.8] | REPLICATES |
| G2 | Acrobot-v1 | reward flip | complete | 8 | -134.5 [-217.1, -51.9]* (neg. scale) | -108.9 [-236.9, +19.0] | -42.5 [-94.3, +9.2] | -66.8 [-148.3, +14.6] | -10.7 [-27.1, +5.7] | –% (–) | 0.9276 | 0.5198 | -54.5 [-147.5, +38.5] | -98.6 [-248.3, +51.0] | NO DAMAGE TO LOCALIZE |
| G3 | SpaceInvaders | actuator noise p=0.5 | complete | 8 | -27.1 [-32.6, -21.6]* (-47%) | -9.5 [-12.7, -6.3]* | -11.0 [-13.8, -8.2]* | -0.5 [-2.1, +1.0] | +1.8 [+0.9, +2.7]* | 115% (8/8) | 0.0000 | 0.0000 | +4.2 [+0.9, +7.5]* | +8.3 [+2.7, +13.9]* | REPLICATES |
| G4 | Freeway-MinAtar | reward flip | partial (n = 4; pre-registered 8) | 4 | -31.0 [-32.2, -29.8]* (-100%) | -38.0 [-40.1, -35.9]* | -30.3 [-34.8, -25.8]* | -7.3 [-11.5, -3.1]* | -1.5 [-4.2, +1.3] | 79% (4/4) | 0.9623 | 0.0359 | -33.5 [-35.3, -31.8]* | -38.0 [-40.0, -36.0]* | – |
| G5 | SpaceInvaders | observation noise sigma=1 | complete | 8 | -24.2 [-32.9, -15.6]* (-42%) | -10.0 [-13.4, -6.5]* | -10.6 [-12.7, -8.5]* | -0.3 [-2.1, +1.6] | +1.0 [-0.5, +2.5] | 107% (8/8) | 0.0000 | 0.0000 | +3.7 [+0.5, +7.0]* | +7.9 [+2.9, +13.0]* | REPLICATES |
| G6 | SpaceInvaders | reward extinction | complete | 8 | -27.4 [-31.1, -23.8]* (-47%) | -9.6 [-13.4, -5.8]* | -10.3 [-12.9, -7.7]* | +0.2 [-1.1, +1.6] | -0.4 [-1.6, +0.9] | 104% (8/8) | 0.0000 | 0.0000 | +4.1 [+0.9, +7.4]* | +5.2 [+1.2, +9.1]* | REPLICATES |

## Pre-registered addendum: gap-damage prediction (relative gap < 25% => no damage; >= 60% => damage)

| id | setting | relative gap | prediction | observed (total CI < 0?) | hit |
|---|---|---|---|---|---|
| G1 | CartPole-v1, reward flip | 98% | damage | damage | yes |
| G2 | Acrobot-v1, reward flip | 44% | none | no damage | – |
| G3 | SpaceInvaders, actuator noise p=0.5 | 47% | none | damage | – |
| G4 | Freeway-MinAtar, reward flip | 100% | damage | damage | descriptive only (incomplete n) |
| G5 | SpaceInvaders, observation noise sigma=1 | 42% | none | damage | – |
| G6 | SpaceInvaders, reward extinction | 47% | none | damage | – |

Complete planned cohorts only: **1 of 1** hits where a prediction was issued (G1 acute values had been seen).
Untouched settings only (exclude G1): **0 of 0** hits. Intermediate gaps receive no prediction.
The addendum predicts a same-update gap, not persistence beyond lost time. See disclosure correction 79c9c85.
Descriptive: Spearman rho(relative gap, relative total damage) over 11 PPO settings = -0.84 (negative = larger gap, more damage). Relative total = total AUC damage / C(w1) (range for Acrobot).

## Pre-registered cross-setting summary

- Complete settings with a same-update gap to localize (total CI < 0): 8 — main, J4, J7, E2, G1, G3, G5, G6
- Actor is the largest carrier in 8 of 8: main 90%, J4 100%, J7 82%, E2 94%, G1 95%, G3 115%, G5 107%, G6 104%
- Actor Shapley share range: 82–115%

Per-test details: `p12_<name>-analysis.md`. Earlier settings: the files listed in the table.

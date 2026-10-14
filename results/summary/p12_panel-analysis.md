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
| G2 | Acrobot-v1 | reward flip | not run | – | – | – | – | – | – | – | – | – | – | – | – |
| G3 | SpaceInvaders | actuator noise p=0.5 | not run | – | – | – | – | – | – | – | – | – | – | – | – |
| G4 | Freeway-MinAtar | reward flip | not run | – | – | – | – | – | – | – | – | – | – | – | – |
| G5 | SpaceInvaders | observation noise sigma=1 | not run | – | – | – | – | – | – | – | – | – | – | – | – |
| G6 | SpaceInvaders | reward extinction | not run | – | – | – | – | – | – | – | – | – | – | – | – |

## Pre-registered addendum: gap-damage prediction (relative gap < 25% => no damage; >= 60% => damage)

| id | setting | relative gap | prediction | observed (total CI < 0?) | hit |
|---|---|---|---|---|---|
| G1 | CartPole-v1, reward flip | 98% | damage | damage | yes |

Hits: **1 of 1** settings with a prediction.
Descriptive: Spearman rho(relative gap, relative total damage) over 6 PPO settings = -0.94 (negative = larger gap, more damage). Relative total = total AUC damage / C(w1) (range for Acrobot).

## Pre-registered cross-setting summary

- Settings with damage to localize (total CI < 0): 5 — main, J4, J7, E2, G1
- Actor is the largest carrier in 5 of 5: main 90%, J4 100%, J7 82%, E2 94%, G1 95%
- Actor Shapley share range: 82–100%

Per-test details: `p12_<name>-analysis.md`. Earlier settings: the files listed in the table.

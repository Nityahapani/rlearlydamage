# Phase 8 PPO component surgery — p8_ppo_surgery_n12

Pre-registered Phase 8 J1+J2 (LOG 2026-10-13). Seeds 0-3 factorial from Phase 3 (365cf2b); J2 adds seeds 0-3 data arms. Note: the --reference reproduction check in the J1 summaries compares seeds 4-11 against the Phase 2 reference's seeds 0-3 and is therefore not meaningful for J1; it is exact for J2 (seeds 0-3).

Summaries: 20261011-p3-surgery-move-late-ppo-SpaceInvaders-MinAtar-365cf2b.json, 20261013-p3-p8-surgery-move-late-s4to7-ppo-SpaceInvaders-MinAtar-8c62fe0.json, 20261013-p3-p8-surgery-move-late-s8to11-ppo-SpaceInvaders-MinAtar-8a2de07.json, 20261013-p3-p8-dataarms-move-late-s0to3-ppo-SpaceInvaders-MinAtar-3f952c0.json
Seeds with all factorial arms (n = 12): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Removal at u150; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 80.6 | 58.7 | 62.2 | 74.8 | 97.6 |
| CCD | 81.6 | 58.7 | 57.0 | 75.5 | 101.0 |
| CDC | 80.5 | 58.7 | 58.5 | 71.6 | 99.5 |
| CDD | 78.8 | 58.7 | 51.2 | 71.2 | 100.2 |
| DCC | 67.2 | 21.7 | 41.6 | 59.4 | 88.6 |
| DCD | 68.5 | 21.7 | 39.9 | 56.6 | 90.8 |
| DDC | 65.4 | 21.7 | 39.4 | 56.8 | 88.0 |
| DDD | 65.5 | 21.7 | 33.9 | 52.8 | 89.6 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±3.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -15.1 [-17.2, -13.0]* | -37.0 [-45.1, -28.9]* | -28.4 [-33.4, -23.4]* | -22.0 [-28.8, -15.2]* | -7.9 [-15.7, -0.2]* | 0.0000 | | |
| actor main effect | -13.7 [-14.9, -12.5]* | -37.0 [-45.1, -28.9]* | -18.5 [-23.2, -13.9]* | -16.9 [-19.3, -14.4]* | -10.3 [-15.7, -5.0]* | 0.0000 | 0.0000 | 1.0000 |
| critic main effect | -1.9 [-3.5, -0.4]* | +0.0 [+0.0, +0.0] | -4.4 [-6.4, -2.5]* | -3.5 [-7.0, -0.0]* | -0.2 [-3.9, +3.6] | 0.0188 | 0.0376 | 0.0814 |
| opt main effect | +0.2 [-0.9, +1.3] | +0.0 [+0.0, +0.0] | -4.9 [-6.9, -3.0]* | -1.6 [-5.4, +2.1] | +2.0 [-1.4, +5.4] | 0.7423 | 0.7423 | 0.0001 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 3.6e-15)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -13.6 [-15.0, -12.2]* | 90% | 12/12 |
| critic | -1.8 [-3.2, -0.4]* | 12% | 10/12 |
| opt | +0.3 [-0.9, +1.5] | -2% | 7/12 |
| total | -15.1 [-17.2, -13.0]* | 100% | 12/12 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -13.3 [-16.2, -10.4]* | -37.0 [-45.1, -28.9]* |
| clean actor into damaged CDD - DDD | +13.3 [+11.0, +15.6]* | +37.0 [+28.9, +45.1]* |
| damaged critic only CDC - CCC | -0.0 [-1.9, +1.8] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | +3.0 [+0.8, +5.3]* | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | +1.1 [-0.8, +2.9] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | -0.1 [-2.0, +1.9] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | -1.8 [-4.5, +0.9] | +0.0 [+0.0, +0.0] |

## Replication on fresh seeds only (n = 8: [4, 5, 6, 7, 8, 9, 10, 11]), AUC

| contrast | AUC |
|---|---|
| total DDD - CCC | -15.5 [-18.7, -12.2]* |
| actor main effect | -14.2 [-16.0, -12.4]* |
| critic main effect | -1.8 [-3.9, +0.4] |
| opt main effect | +0.3 [-1.2, +1.8] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -16.2 | -12.8 | -3.5 | -0.7 | -12.5 |
| 1 | -12.7 | -12.1 | +0.1 | -0.8 | -12.0 |
| 2 | -15.9 | -13.6 | -0.5 | -1.4 | -13.8 |
| 3 | -12.6 | -12.3 | -5.2 | +2.4 | -11.5 |
| 4 | -14.9 | -14.4 | -1.9 | +1.0 | -14.3 |
| 5 | -23.7 | -18.1 | -3.8 | -1.6 | -18.1 |
| 6 | -15.6 | -14.2 | -6.0 | +4.0 | -14.0 |
| 7 | -12.1 | -13.1 | +3.0 | -0.4 | -13.6 |
| 8 | -13.1 | -12.1 | -0.7 | -1.4 | -11.8 |
| 9 | -17.9 | -16.1 | -1.3 | +0.8 | -16.6 |
| 10 | -11.6 | -11.3 | -2.0 | -0.9 | -10.4 |
| 11 | -15.0 | -14.3 | -1.6 | +0.9 | -14.3 |

## Data arms (behaviour swap for the first K updates)

| contrast | n | seeds | AUC | +10 | +50 | AUC p |
|---|---|---|---|---|---|---|
| ratio-neutral clean data into damaged: DDD+bCr - DDD | 12 | 0–11 | -5.4 [-7.0, -3.7]* | -18.0 [-22.4, -13.5]* | -1.5 [-7.8, +4.7] | 0.0000 (Holm 0.0000) |
| ratio-neutral damaged data into clean: CCC+bDr - CCC | 12 | 0–11 | -2.5 [-6.6, +1.7] | -11.7 [-25.0, +1.5] | -5.0 [-10.3, +0.3] | 0.2172 (Holm 0.2172) |
| IS clean data into damaged: DDD+bC - DDD | 12 | 0–11 | +8.3 [+6.1, +10.5]* | +11.8 [+7.0, +16.6]* | +13.2 [+4.4, +22.0]* | 0.0000 |
| IS damaged data into clean: CCC+bD - CCC | 12 | 0–11 | -9.4 [-14.7, -4.2]* | -26.8 [-34.4, -19.1]* | -11.5 [-18.4, -4.5]* | 0.0023 |
| IS - ratio-neutral (clean data, damaged agent): DDD+bC - DDD+bCr | 12 | 0–11 | +13.6 [+11.5, +15.7]* | +29.8 [+25.4, +34.2]* | +14.7 [+9.9, +19.5]* | 0.0000 |
| IS - ratio-neutral (damaged data, clean agent): CCC+bD - CCC+bDr | 12 | 0–11 | -7.0 [-11.1, -2.8]* | -15.1 [-27.8, -2.3]* | -6.5 [-12.2, -0.8]* | 0.0035 |

Swap diagnostics (seed-weighted mean over files; first K updates = swap, next K = after):

| arm | window | clip_frac | ratio_absdev | approx_kl | behaviour_kl |
|---|---|---|---|---|---|
| CCC | first K | 0.074 | 0.057 | 0.0067 | 0.000 |
| CCC | next K | 0.073 | 0.057 | 0.0069 | 0.000 |
| DDD | first K | 0.087 | 0.062 | 0.0083 | 0.000 |
| DDD | next K | 0.073 | 0.057 | 0.0063 | 0.000 |
| CCC+bD | first K | 0.336 | 0.257 | 0.1540 | 0.182 |
| CCC+bD | next K | 0.071 | 0.055 | 0.0065 | 0.000 |
| DDD+bC | first K | 0.335 | 0.230 | 0.0993 | 0.135 |
| DDD+bC | next K | 0.071 | 0.057 | 0.0062 | 0.000 |
| CCC+bDr | first K | 0.195 | 0.116 | 0.0497 | 1.070 |
| CCC+bDr | next K | 0.073 | 0.055 | 0.0070 | 0.000 |
| DDD+bCr | first K | 0.230 | 0.134 | 0.0379 | 0.757 |
| DDD+bCr | next K | 0.075 | 0.055 | 0.0077 | 0.000 |

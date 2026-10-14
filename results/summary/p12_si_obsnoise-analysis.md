# Phase 8 PPO component surgery — p12_si_obsnoise

Phase 12 G5 (pre-registered): SpaceInvaders, observation noise sigma=1. TOST margin = 10% of |CCC AUC| = ±7.989.

Summaries: 20261014-p3-p12-si-obsnoise-s0-ppo-SpaceInvaders-MinAtar-edd81e5.json, 20261014-p3-p12-si-obsnoise-s4-ppo-SpaceInvaders-MinAtar-2db54fb.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u150; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 79.9 | 58.2 | 62.7 | 75.0 | 97.0 |
| CCD | 80.3 | 58.2 | 59.6 | 74.7 | 97.8 |
| CDC | 78.9 | 58.2 | 57.4 | 69.0 | 95.5 |
| CDD | 78.4 | 58.2 | 51.8 | 68.7 | 99.8 |
| DCC | 66.9 | 34.0 | 47.1 | 57.1 | 83.5 |
| DCD | 69.7 | 34.0 | 46.5 | 60.0 | 88.8 |
| DDC | 68.6 | 34.0 | 43.9 | 58.0 | 86.4 |
| DDD | 69.9 | 34.0 | 44.3 | 58.7 | 93.7 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±8.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -10.0 [-13.4, -6.5]* | -24.2 [-32.9, -15.6]* | -18.5 [-23.5, -13.5]* | -16.3 [-25.9, -6.7]* | -3.3 [-15.1, +8.5] | 0.0003 | | |
| actor main effect | -10.6 [-12.7, -8.5]* | -24.2 [-32.9, -15.6]* | -12.5 [-16.2, -8.8]* | -13.4 [-18.1, -8.8]* | -9.4 [-15.9, -2.9]* | 0.0000 | 0.0000 | 0.9895 |
| critic main effect | -0.3 [-2.1, +1.6] | +0.0 [+0.0, +0.0] | -4.6 [-7.6, -1.6]* | -3.1 [-7.2, +1.0] | +2.1 [-2.5, +6.7] | 0.7507 | 0.7507 | 0.0000 |
| opt main effect | +1.0 [-0.5, +2.5] | +0.0 [+0.0, +0.0] | -2.3 [-5.0, +0.5] | +0.8 [-3.0, +4.5] | +4.4 [-0.1, +9.0] | 0.1468 | 0.2936 | 0.0000 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 1.8e-15)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -10.6 [-13.0, -8.3]* | 107% | 8/8 |
| critic | -0.3 [-2.2, +1.6] | 3% | 3/8 |
| opt | +1.0 [-0.6, +2.5] | -10% | 2/8 |
| total | -10.0 [-13.4, -6.5]* | 100% | 8/8 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -13.0 [-16.6, -9.3]* | -24.2 [-32.9, -15.6]* |
| clean actor into damaged CDD - DDD | +8.5 [+3.6, +13.4]* | +24.2 [+15.6, +32.9]* |
| damaged critic only CDC - CCC | -1.0 [-3.2, +1.2] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | -0.2 [-3.7, +3.4] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | +0.4 [-2.7, +3.5] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | -1.3 [-3.6, +0.9] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | -1.5 [-5.0, +2.0] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -8.7 | -8.4 | -1.0 | +0.1 | -8.2 |
| 1 | -10.1 | -11.6 | +1.1 | +1.3 | -11.9 |
| 2 | -13.7 | -12.3 | -2.7 | +3.0 | -12.8 |
| 3 | -7.3 | -7.4 | -4.2 | +3.7 | -7.2 |
| 4 | -14.4 | -11.9 | +0.9 | -1.5 | -12.5 |
| 5 | -14.5 | -14.3 | +1.2 | -0.6 | -14.6 |
| 6 | -8.6 | -11.1 | +0.4 | +0.4 | -10.5 |
| 7 | -2.5 | -7.7 | +2.1 | +1.7 | -7.2 |

## Fork readouts (n = 8)

- C at onset u100: 45.59; D at removal u150: 33.98; C at removal: 58.22
- unlearning during the window D(w1) - D(w0): -11.6 [-20.5, -2.7]*
- raw gap at removal DDD - CCC (+0): -24.2 [-32.9, -15.6]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..150: +3.7 [+0.5, +7.0]*
- delay-adjusted end (last 5 checkpoints): +7.9 [+2.9, +13.0]*

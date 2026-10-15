# Phase 8 PPO component surgery — p12_si_extinction

Phase 12 G6 (pre-registered): SpaceInvaders, reward extinction. TOST margin = 10% of |CCC AUC| = ±7.989.

Summaries: 20261014-p3-p12-si-extinction-s0-ppo-SpaceInvaders-MinAtar-2db54fb.json, 20261014-p3-p12-si-extinction-s4-ppo-SpaceInvaders-MinAtar-79700c2.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u150; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 79.9 | 58.2 | 62.7 | 75.0 | 97.0 |
| CCD | 79.8 | 58.2 | 63.4 | 77.2 | 99.6 |
| CDC | 79.5 | 58.2 | 57.1 | 73.1 | 94.8 |
| CDD | 80.0 | 58.2 | 55.8 | 71.5 | 94.8 |
| DCC | 70.6 | 30.8 | 43.9 | 63.4 | 91.0 |
| DCD | 67.8 | 30.8 | 42.2 | 60.9 | 84.1 |
| DDC | 69.3 | 30.8 | 42.5 | 62.4 | 87.2 |
| DDD | 70.3 | 30.8 | 44.0 | 60.2 | 87.4 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±8.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -9.6 [-13.4, -5.8]* | -27.4 [-31.1, -23.8]* | -18.7 [-24.7, -12.8]* | -14.8 [-21.6, -7.9]* | -9.5 [-18.6, -0.5]* | 0.0006 | | |
| actor main effect | -10.3 [-12.9, -7.7]* | -27.4 [-31.1, -23.8]* | -16.6 [-20.8, -12.4]* | -12.5 [-15.1, -9.8]* | -9.1 [-13.8, -4.4]* | 0.0000 | 0.0001 | 0.9623 |
| critic main effect | +0.2 [-1.1, +1.6] | +0.0 [+0.0, +0.0] | -3.2 [-5.2, -1.2]* | -2.3 [-5.3, +0.7] | -1.8 [-4.6, +1.0] | 0.6665 | 1.0000 | 0.0000 |
| opt main effect | -0.4 [-1.6, +0.9] | +0.0 [+0.0, +0.0] | -0.2 [-2.3, +1.9] | -1.0 [-3.3, +1.4] | -1.0 [-2.9, +0.9] | 0.5133 | 1.0000 | 0.0000 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 1.8e-15)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -10.0 [-12.6, -7.4]* | 104% | 8/8 |
| critic | +0.5 [-0.9, +1.9] | -5% | 3/8 |
| opt | -0.1 [-1.3, +1.1] | 1% | 5/8 |
| total | -9.6 [-13.4, -5.8]* | 100% | 7/8 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -9.2 [-12.9, -5.6]* | -27.4 [-31.1, -23.8]* |
| clean actor into damaged CDD - DDD | +9.7 [+7.3, +12.1]* | +27.4 [+23.8, +31.1]* |
| damaged critic only CDC - CCC | -0.4 [-2.1, +1.4] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | -2.5 [-5.2, +0.2] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | -0.1 [-2.8, +2.7] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | -1.0 [-3.0, +1.1] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | +0.1 [-2.4, +2.6] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -13.9 | -10.2 | -2.2 | -2.6 | -9.8 |
| 1 | -8.0 | -10.1 | +2.5 | -0.4 | -10.1 |
| 2 | -9.8 | -10.1 | +0.4 | -0.9 | -9.8 |
| 3 | -10.9 | -13.0 | -0.4 | +0.6 | -12.4 |
| 4 | -13.6 | -13.7 | -0.4 | +0.8 | -13.8 |
| 5 | -11.9 | -10.0 | -1.1 | -0.5 | -10.1 |
| 6 | -9.1 | -11.7 | +1.8 | -2.0 | -10.8 |
| 7 | +0.5 | -3.5 | +1.3 | +2.0 | -3.2 |

## Fork readouts (n = 8)

- C at onset u100: 45.59; D at removal u150: 30.79; C at removal: 58.22
- unlearning during the window D(w1) - D(w0): -14.8 [-24.5, -5.0]*
- raw gap at removal DDD - CCC (+0): -27.4 [-31.1, -23.8]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..150: +4.1 [+0.9, +7.4]*
- delay-adjusted end (last 5 checkpoints): +5.2 [+1.2, +9.1]*

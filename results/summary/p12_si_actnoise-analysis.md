# Phase 8 PPO component surgery — p12_si_actnoise

Phase 12 G3 (pre-registered): SpaceInvaders, actuator noise p=0.5. TOST margin = 10% of |CCC AUC| = ±7.989.

Summaries: 20261014-p3-p12-si-actnoise-s0-ppo-SpaceInvaders-MinAtar-21bb271.json, 20261014-p3-p12-si-actnoise-s4-ppo-SpaceInvaders-MinAtar-e5c6b78.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u150; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 79.9 | 58.2 | 62.7 | 75.0 | 97.0 |
| CCD | 81.6 | 58.2 | 62.0 | 75.8 | 96.7 |
| CDC | 79.8 | 58.2 | 60.9 | 71.6 | 99.5 |
| CDD | 81.5 | 58.2 | 57.5 | 74.4 | 98.2 |
| DCC | 69.4 | 31.1 | 41.2 | 61.8 | 92.1 |
| DCD | 70.9 | 31.1 | 42.2 | 61.3 | 96.0 |
| DDC | 68.0 | 31.1 | 43.8 | 58.3 | 87.3 |
| DDD | 70.4 | 31.1 | 43.1 | 61.3 | 89.5 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±8.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -9.5 [-12.7, -6.3]* | -27.1 [-32.6, -21.6]* | -19.6 [-23.3, -15.9]* | -13.7 [-22.8, -4.6]* | -7.5 [-16.5, +1.4] | 0.0002 | | |
| actor main effect | -11.0 [-13.8, -8.2]* | -27.1 [-32.6, -21.6]* | -18.2 [-21.0, -15.3]* | -13.6 [-17.1, -10.0]* | -6.6 [-12.4, -0.8]* | 0.0000 | 0.0001 | 0.9811 |
| critic main effect | -0.5 [-2.1, +1.0] | +0.0 [+0.0, +0.0] | -0.7 [-2.5, +1.0] | -2.1 [-8.0, +3.9] | -1.9 [-6.7, +3.0] | 0.4510 | 0.4510 | 0.0000 |
| opt main effect | +1.8 [+0.9, +2.7]* | +0.0 [+0.0, +0.0] | -0.9 [-3.2, +1.3] | +1.5 [+0.2, +2.8]* | +1.1 [-3.8, +6.1] | 0.0026 | 0.0052 | 0.0000 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 1.8e-15)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -10.9 [-13.7, -8.2]* | 115% | 8/8 |
| critic | -0.4 [-2.1, +1.2] | 5% | 4/8 |
| opt | +1.9 [+0.8, +3.0]* | -20% | 0/8 |
| total | -9.5 [-12.7, -6.3]* | 100% | 8/8 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -10.5 [-14.0, -7.0]* | -27.1 [-32.6, -21.6]* |
| clean actor into damaged CDD - DDD | +11.1 [+7.4, +14.8]* | +27.1 [+21.6, +32.6]* |
| damaged critic only CDC - CCC | -0.0 [-2.0, +1.9] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | +0.6 [-2.8, +3.9] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | +1.7 [-0.4, +3.8] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | -2.4 [-4.8, +0.1] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | +1.6 [-1.5, +4.7] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -11.9 | -13.8 | -0.1 | +0.5 | -13.3 |
| 1 | -7.0 | -14.5 | +1.2 | +2.6 | -13.3 |
| 2 | -13.8 | -14.1 | +1.7 | +2.0 | -15.2 |
| 3 | -5.9 | -5.3 | -1.8 | +1.3 | -5.4 |
| 4 | -11.2 | -11.3 | -2.5 | +2.2 | -11.1 |
| 5 | -14.3 | -12.4 | -1.7 | +0.2 | -12.5 |
| 6 | -8.2 | -7.8 | -2.6 | +1.9 | -7.7 |
| 7 | -3.8 | -9.0 | +1.6 | +3.7 | -9.0 |

## Fork readouts (n = 8)

- C at onset u100: 45.59; D at removal u150: 31.10; C at removal: 58.22
- unlearning during the window D(w1) - D(w0): -14.5 [-18.5, -10.5]*
- raw gap at removal DDD - CCC (+0): -27.1 [-32.6, -21.6]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..150: +4.2 [+0.9, +7.5]*
- delay-adjusted end (last 5 checkpoints): +8.3 [+2.7, +13.9]*

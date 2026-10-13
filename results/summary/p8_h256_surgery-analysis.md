# Phase 8 PPO component surgery — p8_h256_surgery

Pre-registered Phase 8 J7: architecture — PPO 2x256 actor and critic, movement restriction [100, 150), SpaceInvaders, recovery 150, seeds 0-7.

Summaries: 20261013-p3-p8-surgery-h256-s0to3-ppo-SpaceInvaders-MinAtar-d072d3c.json, 20261013-p3-p8-surgery-h256-s4to7-ppo-SpaceInvaders-MinAtar-8abfef6.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u150; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 94.2 | 68.2 | 78.3 | 87.5 | 108.7 |
| CCD | 92.8 | 68.2 | 67.4 | 81.4 | 101.8 |
| CDC | 89.6 | 68.2 | 64.5 | 86.9 | 107.1 |
| CDD | 90.3 | 68.2 | 56.2 | 83.1 | 103.0 |
| DCC | 76.5 | 18.3 | 46.1 | 65.8 | 97.5 |
| DCD | 77.1 | 18.3 | 44.7 | 67.3 | 95.0 |
| DDC | 75.9 | 18.3 | 41.9 | 66.3 | 97.0 |
| DDD | 74.6 | 18.3 | 33.6 | 67.3 | 100.2 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±3.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -19.7 [-23.7, -15.6]* | -49.9 [-56.9, -43.0]* | -44.7 [-52.5, -36.9]* | -20.2 [-33.1, -7.3]* | -8.5 [-15.2, -1.8]* | 0.0000 | | |
| actor main effect | -15.7 [-17.6, -13.8]* | -49.9 [-56.9, -43.0]* | -25.0 [-29.8, -20.3]* | -18.1 [-21.1, -15.0]* | -7.7 [-11.3, -4.1]* | 0.0000 | 0.0000 | 1.0000 |
| critic main effect | -2.5 [-4.5, -0.5]* | +0.0 [+0.0, +0.0] | -10.1 [-14.4, -5.7]* | +0.4 [-8.3, +9.2] | +1.0 [-3.5, +5.6] | 0.0199 | 0.0398 | 0.3030 |
| opt main effect | -0.4 [-2.6, +1.7] | +0.0 [+0.0, +0.0] | -7.2 [-11.5, -2.9]* | -1.9 [-7.8, +4.1] | -2.6 [-7.8, +2.6] | 0.6665 | 0.6665 | 0.0126 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 3.6e-15)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -16.1 [-18.0, -14.1]* | 82% | 8/8 |
| critic | -2.9 [-5.4, -0.4]* | 15% | 7/8 |
| opt | -0.7 [-3.0, +1.5] | 4% | 5/8 |
| total | -19.7 [-23.7, -15.6]* | 100% | 8/8 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -17.7 [-21.3, -14.2]* | -49.9 [-56.9, -43.0]* |
| clean actor into damaged CDD - DDD | +15.7 [+11.7, +19.7]* | +49.9 [+43.0, +56.9]* |
| damaged critic only CDC - CCC | -4.6 [-9.1, -0.1]* | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | +2.5 [-1.5, +6.5] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | -1.5 [-5.8, +2.9] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | +1.4 [-2.1, +4.8] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | -4.0 [-6.9, -1.0]* | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -16.2 | -17.0 | -0.5 | +3.4 | -17.7 |
| 1 | -23.4 | -18.4 | -2.0 | -3.2 | -18.3 |
| 2 | -18.1 | -17.2 | -4.5 | +2.3 | -16.8 |
| 3 | -24.1 | -17.2 | -3.4 | -0.3 | -18.2 |
| 4 | -20.7 | -11.1 | -3.1 | -4.4 | -11.8 |
| 5 | -11.3 | -14.8 | +1.7 | -1.2 | -13.8 |
| 6 | -17.5 | -14.9 | -2.3 | +0.5 | -15.2 |
| 7 | -26.2 | -15.2 | -6.2 | -0.4 | -16.6 |

## Fork readouts (n = 8)

- C at onset u100: 46.94; D at removal u150: 18.25; C at removal: 68.19
- unlearning during the window D(w1) - D(w0): -28.7 [-36.4, -21.0]*
- raw gap at removal DDD - CCC (+0): -49.9 [-56.9, -43.0]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..150: -5.1 [-8.3, -1.8]*
- delay-adjusted end (last 5 checkpoints): -2.4 [-7.8, +2.9]

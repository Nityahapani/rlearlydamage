# Phase 8 PPO component surgery — p8_rflip_surgery

Pre-registered Phase 8 J6: different deficit family — reward sign flip (prob 1.0) during [100, 150), PPO SpaceInvaders, recovery 150, seeds 0-7.

Summaries: 20261013-p3-p8-surgery-rflip-s0to3-ppo-SpaceInvaders-MinAtar-b634055.json, 20261013-p3-p8-surgery-rflip-s4to7-ppo-SpaceInvaders-MinAtar-d072d3c.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u150; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 79.9 | 58.2 | 62.7 | 75.0 | 97.0 |
| CCD | 80.3 | 58.2 | 62.3 | 69.5 | 95.7 |
| CDC | 78.7 | 58.2 | 56.4 | 75.4 | 94.1 |
| CDD | 79.5 | 58.2 | 53.8 | 73.2 | 96.0 |
| DCC | 48.5 | 0.0 | 0.9 | 39.5 | 77.6 |
| DCD | 41.5 | 0.0 | 1.3 | 26.0 | 81.6 |
| DDC | 49.5 | 0.0 | 0.9 | 43.5 | 70.9 |
| DDD | 47.6 | 0.0 | 0.2 | 40.5 | 75.6 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±3.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -32.3 [-40.7, -23.9]* | -58.2 [-66.4, -50.1]* | -62.5 [-70.3, -54.7]* | -34.5 [-44.8, -24.2]* | -21.4 [-36.5, -6.4]* | 0.0000 | | |
| actor main effect | -32.8 [-36.9, -28.7]* | -58.2 [-66.4, -50.1]* | -57.9 [-64.9, -51.0]* | -35.9 [-41.2, -30.7]* | -19.3 [-24.9, -13.7]* | 0.0000 | 0.0000 | 1.0000 |
| critic main effect | +1.2 [-2.5, +4.9] | +0.0 [+0.0, +0.0] | -4.0 [-6.0, -1.9]* | +5.6 [+0.1, +11.2]* | -3.8 [-8.9, +1.3] | 0.4583 | 0.4583 | 0.1487 |
| opt main effect | -1.9 [-4.0, +0.2] | +0.0 [+0.0, +0.0] | -0.8 [-2.3, +0.7] | -6.0 [-9.3, -2.7]* | +2.3 [-5.3, +10.0] | 0.0692 | 0.1385 | 0.1315 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 7.1e-15)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -32.4 [-36.5, -28.3]* | 100% | 8/8 |
| critic | +1.6 [-2.1, +5.4] | -5% | 2/8 |
| opt | -1.5 [-3.8, +0.8] | 5% | 6/8 |
| total | -32.3 [-40.7, -23.9]* | 100% | 8/8 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -31.4 [-34.8, -27.9]* | -58.2 [-66.4, -50.1]* |
| clean actor into damaged CDD - DDD | +31.9 [+24.2, +39.6]* | +58.2 [+50.1, +66.4]* |
| damaged critic only CDC - CCC | -1.2 [-4.9, +2.4] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | -6.0 [-13.0, +0.9] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | +0.4 [-2.4, +3.2] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | +1.9 [-1.9, +5.7] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | -0.4 [-3.4, +2.5] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -52.7 | -41.8 | -8.8 | -3.1 | -41.5 |
| 1 | -25.4 | -31.3 | +4.5 | -1.1 | -30.5 |
| 2 | -35.8 | -33.1 | -0.4 | -2.0 | -33.3 |
| 3 | -26.8 | -25.9 | +3.7 | -4.0 | -26.1 |
| 4 | -35.0 | -37.0 | +2.8 | -1.4 | -36.8 |
| 5 | -35.7 | -32.7 | +2.0 | -6.0 | -32.4 |
| 6 | -20.1 | -28.0 | +5.1 | +2.3 | -27.8 |
| 7 | -26.9 | -32.5 | +1.0 | -0.0 | -31.0 |

## Fork readouts (n = 8)

- C at onset u100: 45.59; D at removal u150: 0.00; C at removal: 58.22
- unlearning during the window D(w1) - D(w0): -45.6 [-50.3, -40.9]*
- raw gap at removal DDD - CCC (+0): -58.2 [-66.4, -50.1]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..150: -18.6 [-27.2, -9.9]*
- delay-adjusted end (last 5 checkpoints): -10.7 [-22.0, +0.6]

# Phase 8 PPO component surgery — p10_onsetref_rflip

Phase 10 K1: onset-reference (pause) surgery, SpaceInvaders reward flip; 'C' = onset (P) component

Summaries: 20261014-p3-p10-onsetref-rflip-s0to3-ppo-SpaceInvaders-MinAtar-f618d6c.json, 20261014-p3-p10-onsetref-rflip-s4to7-ppo-SpaceInvaders-MinAtar-36ced59.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u150; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 67.4 | 45.6 | 44.4 | 58.6 | 86.6 |
| CCD | 67.1 | 45.6 | 46.8 | 56.9 | 82.5 |
| CDC | 67.5 | 45.6 | 43.8 | 59.0 | 84.0 |
| CDD | 68.8 | 45.6 | 41.2 | 56.1 | 82.9 |
| DCC | 46.6 | 0.0 | 1.1 | 41.1 | 74.6 |
| DCD | 40.1 | 0.0 | 0.5 | 27.1 | 74.6 |
| DDC | 48.6 | 0.0 | 0.8 | 42.9 | 72.1 |
| DDD | 47.6 | 0.0 | 0.2 | 40.5 | 75.6 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±3.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -19.8 [-27.8, -11.8]* | -45.6 [-50.3, -40.9]* | -44.1 [-49.9, -38.4]* | -18.2 [-26.4, -9.9]* | -11.0 [-26.8, +4.7] | 0.0006 | | |
| actor main effect | -22.0 [-25.2, -18.8]* | -45.6 [-50.3, -40.9]* | -43.4 [-47.4, -39.4]* | -19.8 [-23.5, -16.0]* | -9.8 [-18.1, -1.5]* | 0.0000 | 0.0000 | 1.0000 |
| critic main effect | +2.8 [-2.8, +8.4] | +0.0 [+0.0, +0.0] | -1.7 [-3.1, -0.3]* | +3.7 [-4.1, +11.5] | -0.9 [-6.4, +4.6] | 0.2723 | 0.3767 | 0.4701 |
| opt main effect | -1.6 [-4.3, +1.0] | +0.0 [+0.0, +0.0] | -0.3 [-2.1, +1.5] | -5.3 [-8.3, -2.3]* | -0.5 [-6.1, +5.2] | 0.1883 | 0.3767 | 0.1296 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 7.1e-15)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -21.6 [-24.8, -18.4]* | 109% | 8/8 |
| critic | +3.1 [-2.6, +8.9] | -16% | 1/8 |
| opt | -1.3 [-3.7, +1.1] | 7% | 5/8 |
| total | -19.8 [-27.8, -11.8]* | 100% | 8/8 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -20.7 [-24.7, -16.8]* | -45.6 [-50.3, -40.9]* |
| clean actor into damaged CDD - DDD | +21.2 [+14.9, +27.5]* | +45.6 [+40.9, +50.3]* |
| damaged critic only CDC - CCC | +0.1 [-2.5, +2.7] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | -7.5 [-18.2, +3.1] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | -0.3 [-2.4, +1.8] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | +1.0 [-1.6, +3.5] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | +1.4 [-1.0, +3.8] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -40.8 | -28.6 | -11.9 | +0.1 | -28.7 |
| 1 | -15.5 | -19.3 | +6.4 | -4.7 | -18.6 |
| 2 | -21.1 | -26.3 | +8.2 | -6.0 | -25.3 |
| 3 | -15.2 | -20.1 | +7.7 | -1.9 | -20.4 |
| 4 | -20.0 | -23.2 | +2.1 | +0.9 | -23.1 |
| 5 | -22.2 | -21.4 | +1.4 | -4.8 | -20.5 |
| 6 | -8.0 | -17.3 | +7.6 | +1.2 | -17.2 |
| 7 | -15.4 | -19.5 | +1.1 | +2.1 | -19.2 |

# Phase 8 PPO component surgery — p10_onsetref_move

Phase 10 K2: onset-reference (pause) surgery, SpaceInvaders movement (n = 4, descriptive); 'C' = onset (P) component

Summaries: 20261014-p3-p10-onsetref-move-s0to3-ppo-SpaceInvaders-MinAtar-36ced59.json
Seeds with all factorial arms (n = 4): [0, 1, 2, 3]. Removal at u150; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 67.3 | 45.4 | 41.6 | 59.0 | 89.1 |
| CCD | 68.4 | 45.4 | 45.1 | 61.6 | 88.3 |
| CDC | 67.9 | 45.4 | 45.1 | 58.8 | 88.0 |
| CDD | 67.7 | 45.4 | 39.6 | 57.2 | 86.0 |
| DCC | 63.8 | 18.5 | 42.3 | 56.0 | 84.7 |
| DCD | 64.4 | 18.5 | 36.0 | 55.5 | 82.3 |
| DDC | 62.6 | 18.5 | 39.4 | 52.3 | 79.2 |
| DDD | 65.0 | 18.5 | 34.2 | 51.2 | 85.6 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±3.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -2.3 [-4.8, +0.2] | -26.9 [-45.6, -8.1]* | -7.4 [-17.6, +2.9] | -7.8 [-17.0, +1.3] | -3.5 [-19.5, +12.5] | 0.0603 | | |
| actor main effect | -3.9 [-5.7, -2.1]* | -26.9 [-45.6, -8.1]* | -4.8 [-12.5, +2.8] | -5.4 [-8.8, -2.0]* | -4.9 [-11.4, +1.6] | 0.0060 | 0.0181 | 0.8964 |
| critic main effect | -0.2 [-1.2, +0.9] | +0.0 [+0.0, +0.0] | -1.7 [-8.2, +4.9] | -3.2 [-9.8, +3.5] | -1.4 [-12.2, +9.3] | 0.6324 | 0.7060 | 0.0018 |
| opt main effect | +1.0 [-1.9, +3.9] | +0.0 [+0.0, +0.0] | -3.4 [-8.0, +1.3] | -0.1 [-4.4, +4.1] | +0.3 [-6.0, +6.5] | 0.3530 | 0.7060 | 0.0572 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 0.0e+00)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -3.6 [-5.5, -1.8]* | n/a (total CI includes 0) | 4/4 |
| critic | +0.1 [-2.1, +2.2] | n/a (total CI includes 0) | 1/4 |
| opt | +1.3 [-0.7, +3.3] | n/a (total CI includes 0) | 0/4 |
| total | -2.3 [-4.8, +0.2] | 100% | 4/4 |

Pre-registered replication criterion: **NO DAMAGE TO LOCALIZE** (total-damage CI includes 0) — not a failure to replicate

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -3.5 [-8.9, +1.9] | -26.9 [-45.6, -8.1]* |
| clean actor into damaged CDD - DDD | +2.7 [-0.3, +5.8] | +26.9 [+8.1, +45.6]* |
| damaged critic only CDC - CCC | +0.7 [-4.6, +5.9] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | -0.5 [-5.4, +4.4] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | +1.2 [-0.6, +2.9] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | -2.4 [-8.0, +3.3] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | +0.4 [-1.8, +2.7] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -4.3 | -2.8 | -1.1 | +2.2 | -3.6 |
| 1 | -2.8 | -4.8 | +0.1 | -0.2 | -4.1 |
| 2 | -1.2 | -4.9 | +0.5 | +2.9 | -4.8 |
| 3 | -1.0 | -3.1 | -0.2 | -0.9 | -2.0 |

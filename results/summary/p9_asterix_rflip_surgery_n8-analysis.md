# Phase 8 PPO component surgery — p9_asterix_rflip_surgery_n8

Phase 9 E2 completed (Phase 11): Asterix reward-flip surgery, n = 8 (pre-registered n; completed after unblinded n = 4 interim)

Summaries: 20261014-p3-p9-surgery-asterix-rflip-s0to3-ppo-Asterix-MinAtar-4a02dd2.json, 20261014-p3-p9-surgery-asterix-rflip-s4to7-ppo-Asterix-MinAtar-ac63459.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u440; recovery 200; eval every 10. AUC = mean eval over +10..+200. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +200 |
|---|---|---|---|---|---|
| CCC | 6.7 | 5.4 | 5.6 | 6.3 | 8.0 |
| CCD | 6.7 | 5.4 | 5.9 | 5.8 | 8.0 |
| CDC | 6.6 | 5.4 | 5.2 | 6.1 | 7.3 |
| CDD | 6.8 | 5.4 | 5.5 | 5.9 | 7.8 |
| DCC | 4.7 | 1.7 | 2.4 | 3.7 | 6.4 |
| DCD | 4.7 | 1.7 | 2.1 | 3.3 | 6.5 |
| DDC | 4.6 | 1.7 | 2.1 | 3.5 | 6.5 |
| DDD | 4.5 | 1.7 | 1.9 | 3.5 | 6.0 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +200 | AUC p | Holm p | TOST p (±3.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -2.3 [-2.9, -1.7]* | -3.7 [-4.6, -2.8]* | -3.7 [-4.9, -2.5]* | -2.9 [-3.6, -2.1]* | -2.1 [-2.6, -1.5]* | 0.0000 | | |
| actor main effect | -2.1 [-2.6, -1.7]* | -3.7 [-4.6, -2.8]* | -3.4 [-4.3, -2.5]* | -2.5 [-3.2, -1.8]* | -1.4 [-1.9, -1.0]* | 0.0000 | 0.0000 | 0.0012 |
| critic main effect | -0.1 [-0.2, +0.0] | +0.0 [+0.0, +0.0] | -0.3 [-0.6, +0.0] | -0.0 [-0.3, +0.3] | -0.4 [-0.9, +0.2] | 0.1180 | 0.2360 | 0.0000 |
| opt main effect | +0.0 [-0.2, +0.2] | +0.0 [+0.0, +0.0] | -0.0 [-0.3, +0.3] | -0.3 [-0.6, -0.1]* | +0.0 [-0.5, +0.5] | 0.9868 | 0.9868 | 0.0000 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 4.4e-16)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -2.1 [-2.6, -1.7]* | 94% | 8/8 |
| critic | -0.1 [-0.3, +0.0] | 5% | 6/8 |
| opt | -0.0 [-0.3, +0.2] | 1% | 4/8 |
| total | -2.3 [-2.9, -1.7]* | 100% | 8/8 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -2.1 [-2.6, -1.5]* | -3.7 [-4.6, -2.8]* |
| clean actor into damaged CDD - DDD | +2.3 [+2.1, +2.6]* | +3.7 [+2.8, +4.6]* |
| damaged critic only CDC - CCC | -0.1 [-0.4, +0.2] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | +0.2 [-0.0, +0.5] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | -0.0 [-0.4, +0.4] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | +0.1 [-0.1, +0.4] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | +0.0 [-0.4, +0.5] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -2.1 | -1.9 | +0.0 | -0.1 | -1.9 |
| 1 | -3.5 | -2.7 | -0.3 | -0.3 | -2.7 |
| 2 | -2.5 | -2.8 | +0.0 | +0.3 | -2.8 |
| 3 | -1.7 | -1.4 | -0.0 | +0.2 | -1.5 |
| 4 | -3.1 | -2.3 | -0.3 | -0.4 | -2.3 |
| 5 | -1.7 | -1.7 | +0.1 | -0.0 | -1.7 |
| 6 | -2.3 | -2.6 | -0.1 | +0.1 | -2.5 |
| 7 | -1.5 | -1.7 | -0.1 | +0.2 | -1.7 |

## Fork readouts (n = 8)

- C at onset u390: 5.36; D at removal u440: 1.70; C at removal: 5.41
- unlearning during the window D(w1) - D(w0): -3.7 [-4.6, -2.7]*
- raw gap at removal DDD - CCC (+0): -3.7 [-4.6, -2.8]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..200: -1.7 [-2.1, -1.3]*
- delay-adjusted end (last 5 checkpoints): -1.2 [-1.7, -0.6]*

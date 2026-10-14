# Phase 8 PPO component surgery — p9_asterix_rflip_surgery

Phase 9 E2: Asterix reward-flip surgery (n = 4, reduced from 8 — deviation)

Summaries: 20261014-p3-p9-surgery-asterix-rflip-s0to3-ppo-Asterix-MinAtar-4a02dd2.json
Seeds with all factorial arms (n = 4): [0, 1, 2, 3]. Removal at u440; recovery 200; eval every 10. AUC = mean eval over +10..+200. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +200 |
|---|---|---|---|---|---|
| CCC | 7.4 | 6.0 | 6.1 | 7.0 | 8.9 |
| CCD | 7.3 | 6.0 | 6.1 | 6.6 | 8.7 |
| CDC | 7.1 | 6.0 | 5.4 | 6.8 | 7.7 |
| CDD | 7.4 | 6.0 | 6.1 | 6.6 | 8.7 |
| DCC | 5.1 | 1.8 | 2.6 | 4.0 | 7.0 |
| DCD | 5.2 | 1.8 | 2.2 | 3.6 | 7.0 |
| DDC | 5.2 | 1.8 | 2.1 | 3.8 | 7.8 |
| DDD | 4.9 | 1.8 | 2.1 | 3.8 | 6.5 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +200 | AUC p | Holm p | TOST p (±3.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -2.4 [-3.6, -1.2]* | -4.2 [-6.0, -2.4]* | -4.0 [-6.9, -1.1]* | -3.2 [-5.1, -1.3]* | -2.4 [-3.0, -1.9]* | 0.0079 | | |
| actor main effect | -2.2 [-3.3, -1.1]* | -4.2 [-6.0, -2.4]* | -3.7 [-5.7, -1.7]* | -2.9 [-4.5, -1.4]* | -1.5 [-1.9, -1.0]* | 0.0077 | 0.0230 | 0.0491 |
| critic main effect | -0.1 [-0.3, +0.2] | +0.0 [+0.0, +0.0] | -0.3 [-1.1, +0.5] | -0.0 [-0.2, +0.2] | -0.2 [-1.3, +0.8] | 0.4606 | 0.9213 | 0.0000 |
| opt main effect | +0.0 [-0.4, +0.5] | +0.0 [+0.0, +0.0] | +0.1 [-0.7, +0.8] | -0.3 [-0.7, +0.1] | -0.1 [-1.4, +1.2] | 0.8491 | 0.9213 | 0.0001 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 0.0e+00)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -2.3 [-3.2, -1.3]* | 93% | 4/4 |
| critic | -0.1 [-0.4, +0.1] | 6% | 3/4 |
| opt | -0.0 [-0.5, +0.4] | 1% | 2/4 |
| total | -2.4 [-3.6, -1.2]* | 100% | 4/4 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -2.3 [-3.4, -1.2]* | -4.2 [-6.0, -2.4]* |
| clean actor into damaged CDD - DDD | +2.5 [+1.9, +3.0]* | +4.2 [+2.4, +6.0]* |
| damaged critic only CDC - CCC | -0.2 [-0.8, +0.4] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | +0.3 [-0.2, +0.7] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | -0.1 [-0.6, +0.5] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | +0.2 [-0.4, +0.9] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | +0.1 [-0.9, +1.0] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -2.1 | -1.9 | +0.0 | -0.1 | -1.9 |
| 1 | -3.5 | -2.7 | -0.3 | -0.3 | -2.7 |
| 2 | -2.5 | -2.8 | +0.0 | +0.3 | -2.8 |
| 3 | -1.7 | -1.4 | -0.0 | +0.2 | -1.5 |

## Fork readouts (n = 4)

- C at onset u390: 6.01; D at removal u440: 1.76; C at removal: 5.98
- unlearning during the window D(w1) - D(w0): -4.2 [-6.6, -1.9]*
- raw gap at removal DDD - CCC (+0): -4.2 [-6.0, -2.4]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..200: -1.8 [-2.6, -1.0]*
- delay-adjusted end (last 5 checkpoints): -1.2 [-2.2, -0.1]*

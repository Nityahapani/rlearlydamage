# Phase 8 PPO component surgery — p8_asterix_surgery

Pre-registered Phase 8 J3: PPO Asterix-MinAtar replication, allowed actions [0, 2, 4] (no left/right), meaning-matched onset u390 (long-control onset rule), window [390, 440), recovery 200, seeds 0-7.

Summaries: 20261013-p3-p8-surgery-asterix-s0to3-ppo-Asterix-MinAtar-bfca538.json, 20261013-p3-p8-surgery-asterix-s4to7-ppo-Asterix-MinAtar-a8e2e91.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u440; recovery 200; eval every 10. AUC = mean eval over +10..+200. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +200 |
|---|---|---|---|---|---|
| CCC | 6.7 | 5.4 | 5.6 | 6.3 | 8.0 |
| CCD | 6.9 | 5.4 | 5.8 | 6.0 | 8.0 |
| CDC | 6.9 | 5.4 | 5.9 | 6.3 | 7.7 |
| CDD | 6.6 | 5.4 | 5.8 | 5.8 | 7.3 |
| DCC | 6.5 | 4.4 | 4.8 | 5.4 | 8.0 |
| DCD | 6.6 | 4.4 | 4.8 | 6.1 | 8.5 |
| DDC | 6.5 | 4.4 | 4.7 | 5.9 | 7.9 |
| DDD | 6.7 | 4.4 | 5.0 | 5.9 | 8.3 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +200 | AUC p | Holm p | TOST p (±3.0) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -0.1 [-0.5, +0.4] | -1.0 [-1.5, -0.5]* | -0.7 [-1.4, +0.1] | -0.4 [-1.1, +0.2] | +0.3 [-1.1, +1.6] | 0.7373 | | |
| actor main effect | -0.2 [-0.7, +0.3] | -1.0 [-1.5, -0.5]* | -1.0 [-1.3, -0.6]* | -0.3 [-1.0, +0.4] | +0.4 [-0.6, +1.4] | 0.4254 | 1.0000 | 0.0000 |
| critic main effect | -0.0 [-0.2, +0.2] | +0.0 [+0.0, +0.0] | +0.1 [-0.1, +0.3] | +0.0 [-0.2, +0.3] | -0.3 [-0.9, +0.2] | 0.8399 | 1.0000 | 0.0000 |
| opt main effect | +0.0 [-0.1, +0.2] | +0.0 [+0.0, +0.0] | +0.1 [-0.2, +0.4] | -0.0 [-0.4, +0.4] | +0.1 [-0.4, +0.7] | 0.7328 | 1.0000 | 0.0000 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 1.1e-16)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -0.1 [-0.6, +0.3] | n/a (total CI includes 0) | 5/8 |
| critic | +0.0 [-0.2, +0.2] | n/a (total CI includes 0) | 4/8 |
| opt | +0.1 [-0.1, +0.2] | n/a (total CI includes 0) | 3/8 |
| total | -0.1 [-0.5, +0.4] | 100% | 4/8 |

Pre-registered replication criterion: **NO DAMAGE TO LOCALIZE** (total-damage CI includes 0) — not a failure to replicate

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -0.3 [-0.8, +0.3] | -1.0 [-1.5, -0.5]* |
| clean actor into damaged CDD - DDD | -0.1 [-0.7, +0.5] | +1.0 [+0.5, +1.5]* |
| damaged critic only CDC - CCC | +0.1 [-0.1, +0.3] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | -0.1 [-0.4, +0.2] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | +0.1 [-0.3, +0.5] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | -0.2 [-0.4, +0.1] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | -0.2 [-0.5, +0.1] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -0.3 | -0.6 | -0.0 | +0.1 | -0.5 |
| 1 | +0.3 | +0.8 | -0.0 | -0.3 | +0.8 |
| 2 | +0.3 | -0.2 | +0.5 | +0.0 | -0.2 |
| 3 | -0.3 | -0.1 | -0.2 | -0.2 | -0.0 |
| 4 | -1.0 | -1.0 | -0.2 | +0.3 | -1.0 |
| 5 | -0.5 | -0.7 | -0.0 | +0.1 | -0.7 |
| 6 | +0.6 | +0.2 | +0.1 | +0.1 | +0.3 |
| 7 | +0.3 | +0.2 | -0.3 | +0.1 | +0.3 |

## Fork readouts (n = 8)

- C at onset u390: 5.36; D at removal u440: 4.38; C at removal: 5.41
- unlearning during the window D(w1) - D(w0): -1.0 [-1.6, -0.3]*
- raw gap at removal DDD - CCC (+0): -1.0 [-1.5, -0.5]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..200: +0.5 [+0.0, +1.0]*
- delay-adjusted end (last 5 checkpoints): +1.0 [+0.2, +1.7]*

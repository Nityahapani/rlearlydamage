# Phase 8 PPO component surgery — p12_freeway_rflip

Phase 12 G4 (pre-registered): Freeway-MinAtar, reward flip. TOST margin = 10% of |CCC AUC| = ±3.804.

Summaries: 20261014-p3-p12-freeway-rflip-s0-ppo-Freeway-MinAtar-685fd17.json, 20261014-p3-p12-freeway-rflip-s2-ppo-Freeway-MinAtar-4254b3e.json
Seeds with all factorial arms (n = 4): [0, 1, 2, 3]. Removal at u70; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 38.0 | 31.0 | 31.4 | 35.6 | 44.4 |
| CCD | 37.2 | 31.0 | 31.1 | 33.7 | 44.0 |
| CDC | 37.9 | 31.0 | 31.1 | 34.4 | 44.8 |
| CDD | 37.3 | 31.0 | 31.1 | 35.2 | 43.9 |
| DCC | 16.9 | 0.0 | 0.0 | 8.5 | 30.9 |
| DCD | 12.4 | 0.0 | 0.0 | 4.6 | 25.6 |
| DDC | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DDD | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±3.8) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -38.0 [-40.1, -35.9]* | -31.0 [-32.2, -29.8]* | -31.4 [-32.8, -30.0]* | -35.6 [-38.0, -33.3]* | -44.3 [-46.5, -42.0]* | 0.0000 | | |
| actor main effect | -30.3 [-34.8, -25.8]* | -31.0 [-32.2, -29.8]* | -31.2 [-32.4, -29.9]* | -31.5 [-38.2, -24.8]* | -30.1 [-32.9, -27.3]* | 0.0002 | 0.0007 | 0.9998 |
| critic main effect | -7.3 [-11.5, -3.1]* | +0.0 [+0.0, +0.0] | -0.1 [-1.0, +0.7] | -3.2 [-8.8, +2.4] | -14.0 [-16.7, -11.3]* | 0.0115 | 0.0230 | 0.9623 |
| opt main effect | -1.5 [-4.2, +1.3] | +0.0 [+0.0, +0.0] | -0.1 [-0.9, +0.7] | -1.2 [-3.6, +1.2] | -1.6 [-4.4, +1.2] | 0.1876 | 0.1876 | 0.0359 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 7.1e-15)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -29.9 [-34.2, -25.7]* | 79% | 4/4 |
| critic | -7.0 [-11.9, -2.0]* | 18% | 4/4 |
| opt | -1.1 [-2.9, +0.7] | 3% | 4/4 |
| total | -38.0 [-40.1, -35.9]* | 100% | 4/4 |

**DESCRIPTIVE ONLY:** planned sample size was not reached. No confirmatory replication decision is applied.

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -21.2 [-29.1, -13.2]* | -31.0 [-32.2, -29.8]* |
| clean actor into damaged CDD - DDD | +37.3 [+35.3, +39.3]* | +31.0 [+29.8, +32.2]* |
| damaged critic only CDC - CCC | -0.2 [-1.1, +0.8] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | +12.4 [+0.1, +24.6]* | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | -0.8 [-1.9, +0.3] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | -0.0 [-0.1, +0.0] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | -0.7 [-1.4, -0.0]* | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -37.9 | -27.0 | -10.1 | -0.5 | -27.1 |
| 1 | -36.3 | -31.7 | -4.2 | -3.9 | -30.6 |
| 2 | -39.5 | -33.3 | -6.2 | -0.1 | -33.3 |
| 3 | -38.3 | -29.1 | -8.8 | -1.4 | -28.8 |

## Fork readouts (n = 4)

- C at onset u20: 24.84; D at removal u70: 0.00; C at removal: 31.02
- unlearning during the window D(w1) - D(w0): -24.8 [-26.4, -23.3]*
- raw gap at removal DDD - CCC (+0): -31.0 [-32.2, -29.8]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..150: -33.5 [-35.3, -31.8]*
- delay-adjusted end (last 5 checkpoints): -38.0 [-40.0, -36.0]*

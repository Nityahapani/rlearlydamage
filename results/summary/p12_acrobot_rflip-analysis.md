# Phase 8 PPO component surgery — p12_acrobot_rflip

Phase 12 G2 (pre-registered): Acrobot-v1, reward flip. TOST margin = 10% of |CCC AUC| = ±10.354.

Summaries: 20261014-p3-p12-acrobot-rflip-s0-ppo-Acrobot-v1-bf9a946.json, 20261014-p3-p12-acrobot-rflip-s4-ppo-Acrobot-v1-21bb271.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u110; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | -103.5 | -193.1 | -181.1 | -106.2 | -81.4 |
| CCD | -103.9 | -193.1 | -179.7 | -106.9 | -83.4 |
| CDC | -159.2 | -193.1 | -199.2 | -181.4 | -134.0 |
| CDD | -158.1 | -193.1 | -196.8 | -179.5 | -135.9 |
| DCC | -111.9 | -327.6 | -245.5 | -122.9 | -80.0 |
| DCD | -156.8 | -327.6 | -281.1 | -146.3 | -114.3 |
| DDC | -213.7 | -327.6 | -308.6 | -215.0 | -187.1 |
| DDD | -212.4 | -327.6 | -287.2 | -220.8 | -187.0 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±10.4) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -108.9 [-236.9, +19.0] | -134.5 [-217.1, -51.9]* | -106.1 [-200.4, -11.8]* | -114.6 [-243.4, +14.1] | -105.6 [-266.2, +54.9] | 0.0840 | | |
| actor main effect | -42.5 [-94.3, +9.2] | -134.5 [-217.1, -51.9]* | -91.4 [-170.2, -12.7]* | -32.8 [-76.2, +10.6] | -33.4 [-95.3, +28.5] | 0.0929 | 0.2786 | 0.9077 |
| critic main effect | -66.8 [-148.3, +14.6] | +0.0 [+0.0, +0.0] | -26.1 [-41.8, -10.4]* | -78.6 [-171.0, +13.8] | -71.2 [-184.7, +42.3] | 0.0933 | 0.2786 | 0.9276 |
| opt main effect | -10.7 [-27.1, +5.7] | +0.0 [+0.0, +0.0] | -2.6 [-6.4, +1.2] | -7.0 [-17.5, +3.5] | -9.5 [-27.6, +8.5] | 0.1663 | 0.2786 | 0.5198 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 5.7e-14)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -38.8 [-89.3, +11.7] | n/a (total CI includes 0) | 8/8 |
| critic | -63.1 [-140.6, +14.4] | n/a (total CI includes 0) | 8/8 |
| opt | -7.0 [-17.2, +3.3] | n/a (total CI includes 0) | 6/8 |
| total | -108.9 [-236.9, +19.0] | 100% | 8/8 |

Pre-registered replication criterion: **NO DAMAGE TO LOCALIZE** (total-damage CI includes 0) — not a failure to replicate

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -8.4 [-16.6, -0.2]* | -134.5 [-217.1, -51.9]* |
| clean actor into damaged CDD - DDD | +54.3 [-43.6, +152.2] | +134.5 [+51.9, +217.1]* |
| damaged critic only CDC - CCC | -55.6 [-140.1, +28.9] | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | +55.7 [-47.8, +159.1] | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | -0.4 [-7.1, +6.4] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | -1.3 [-5.8, +3.3] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | -54.6 [-138.6, +29.4] | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -6.2 | -3.9 | -2.5 | +0.0 | -3.9 |
| 1 | -8.5 | -5.2 | -3.2 | -1.4 | -4.8 |
| 2 | -11.4 | -7.3 | -2.9 | -1.7 | -7.1 |
| 3 | -294.5 | -79.8 | -223.2 | -57.2 | -57.9 |
| 4 | -21.8 | -11.7 | -10.8 | +1.0 | -11.8 |
| 5 | -398.7 | -182.1 | -218.1 | -1.3 | -181.2 |
| 6 | -20.6 | -13.7 | -6.4 | -9.8 | -10.6 |
| 7 | -109.5 | -36.5 | -67.7 | -15.5 | -33.2 |

## Fork readouts (n = 8)

- C at onset u60: -345.91; D at removal u110: -327.59; C at removal: -193.06
- unlearning during the window D(w1) - D(w0): +18.3 [-50.7, +87.4]
- raw gap at removal DDD - CCC (+0): -134.5 [-217.1, -51.9]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..150: -54.5 [-147.5, +38.5]
- delay-adjusted end (last 5 checkpoints): -98.6 [-248.3, +51.0]

# Phase 8 PPO component surgery — p12_cartpole_rflip

Phase 12 G1 (pre-registered): CartPole-v1, reward flip. TOST margin = 10% of |CCC AUC| = ±47.855.

Summaries: 20261014-p3-p12-cartpole-rflip-ppo-CartPole-v1-adeb6b0.json, 20261014-p3-p12-cartpole-rflip-s4-ppo-CartPole-v1-0e9e214.json
Seeds with all factorial arms (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Removal at u60; recovery 150; eval every 10. AUC = mean eval over +10..+150. Paired 95% t-CIs; * = CI excludes 0; negative = harm.

## Arm means

| arm | AUC | +0 | +10 | +50 | +150 |
|---|---|---|---|---|---|
| CCC | 478.5 | 457.4 | 455.6 | 483.7 | 472.2 |
| CCD | 482.8 | 457.4 | 463.4 | 492.5 | 486.3 |
| CDC | 373.4 | 457.4 | 202.5 | 391.8 | 464.4 |
| CDD | 409.8 | 457.4 | 208.7 | 450.8 | 485.3 |
| DCC | 183.1 | 9.2 | 9.3 | 65.3 | 311.3 |
| DCD | 221.9 | 9.2 | 9.2 | 62.2 | 376.7 |
| DDC | 260.0 | 9.2 | 9.7 | 146.8 | 432.3 |
| DDD | 255.2 | 9.2 | 9.2 | 167.3 | 428.3 |

## Total damage and main effects

| contrast | AUC | +0 | +10 | +50 | +150 | AUC p | Holm p | TOST p (±47.9) |
|---|---|---|---|---|---|---|---|---|
| total DDD - CCC | -223.3 [-347.1, -99.5]* | -448.2 [-502.1, -394.2]* | -446.4 [-502.8, -390.0]* | -316.5 [-404.5, -228.5]* | -43.9 [-206.3, +118.6] | 0.0037 | | |
| actor main effect | -206.1 [-319.9, -92.3]* | -448.2 [-502.1, -394.2]* | -323.2 [-386.2, -260.3]* | -344.3 [-398.4, -290.3]* | -89.9 [-227.2, +47.5] | 0.0036 | 0.0109 | 0.9933 |
| critic main effect | -17.0 [-50.8, +16.8] | +0.0 [+0.0, +0.0] | -126.9 [-171.7, -82.0]* | +13.2 [-24.6, +51.0] | +41.0 [-16.4, +98.3] | 0.2736 | 0.2736 | 0.0337 |
| opt main effect | +18.7 [-3.0, +40.3] | +0.0 [+0.0, +0.0] | +3.4 [-6.2, +12.9] | +21.3 [-7.2, +49.8] | +24.1 [-18.1, +66.3] | 0.0810 | 0.1619 | 0.0077 |

## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = 5.7e-14)

| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |
|---|---|---|---|
| actor | -212.4 [-329.3, -95.5]* | 95% | 8/8 |
| critic | -23.3 [-52.6, +6.1] | 10% | 5/8 |
| opt | +12.4 [-8.9, +33.7] | -6% | 2/8 |
| total | -223.3 [-347.1, -99.5]* | 100% | 8/8 |

Pre-registered replication criterion: **REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)

## Single transplants and rescues (AUC)

| contrast | AUC | +0 |
|---|---|---|
| damaged actor only DCC - CCC | -295.4 [-425.9, -164.9]* | -448.2 [-502.1, -394.2]* |
| clean actor into damaged CDD - DDD | +154.6 [+12.0, +297.3]* | +448.2 [+394.2, +502.1]* |
| damaged critic only CDC - CCC | -105.1 [-122.7, -87.5]* | +0.0 [+0.0, +0.0] |
| clean critic DCD - DDD | -33.3 [-63.8, -2.8]* | +0.0 [+0.0, +0.0] |
| damaged opt only CCD - CCC | +4.3 [-6.9, +15.6] | +0.0 [+0.0, +0.0] |
| clean opt DDC - DDD | +4.8 [-45.4, +54.9] | +0.0 [+0.0, +0.0] |
| clean-actor hybrid vs clean CDD - CCC | -68.7 [-107.0, -30.4]* | +0.0 [+0.0, +0.0] |

## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)

| seed | total | actor | critic | opt | phi actor |
|---|---|---|---|---|---|
| 0 | -478.5 | -436.8 | -40.6 | +13.0 | -441.5 |
| 1 | -140.7 | -117.9 | -43.5 | +14.7 | -115.9 |
| 2 | -108.1 | -156.6 | +20.7 | +45.5 | -162.5 |
| 3 | -136.1 | -198.7 | +42.5 | +65.5 | -213.8 |
| 4 | -181.5 | -146.4 | -36.9 | -3.2 | -144.7 |
| 5 | -167.2 | -117.5 | -40.5 | +18.6 | -126.8 |
| 6 | -132.9 | -74.9 | -64.9 | +11.5 | -76.5 |
| 7 | -441.5 | -400.2 | +27.4 | -16.2 | -417.7 |

## Fork readouts (n = 8)

- C at onset u10: 238.65; D at removal u60: 9.20; C at removal: 457.36
- unlearning during the window D(w1) - D(w0): -229.5 [-270.9, -188.0]*
- raw gap at removal DDD - CCC (+0): -448.2 [-502.1, -394.2]*
- delay-adjusted AUC DDD(w1+k) - C(w1+k-50), k = 10..150: -196.6 [-322.8, -70.3]*
- delay-adjusted end (last 5 checkpoints): -91.8 [-262.4, +78.8]

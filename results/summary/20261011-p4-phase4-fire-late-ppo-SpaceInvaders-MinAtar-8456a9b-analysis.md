# Phase 4 causal ablations — 20261011-p4-phase4-fire-late-ppo-SpaceInvaders-MinAtar-8456a9b.json

Seeds [0, 1, 2, 3]; mask window updates [100, 150) (movement removed, allowed [0, 1, 2]); recovery 150 updates; eval every 10; K = 10 behaviour updates; anchor beta = 1.0. All arms evaluated deficit-free with common random numbers. CIs: paired t over seeds.

## Arm scores (mean over seeds)

| arm | AUC | +0 | +10 | +20 | +50 | +150 |
|---|---|---|---|---|---|---|
| C_cont | 79.3 | 58.7 | 61.2 | 67.1 | 68.3 | 95.9 |
| D_online | 61.9 | 15.2 | 28.3 | 35.8 | 52.5 | 84.9 |
| D_actor_reset | 38.5 | 3.7 | 8.0 | 13.8 | 27.9 | 62.3 |
| D_actor_transplant | 74.8 | 58.7 | 55.8 | 60.8 | 67.9 | 93.3 |
| D_actor_opt_reset | 39.0 | 3.7 | 8.7 | 14.7 | 28.9 | 63.2 |
| D_beh_clean | 71.5 | 15.2 | 48.2 | 49.8 | 63.8 | 85.3 |
| D_beh_frozen_self | 59.1 | 15.2 | 19.7 | 30.9 | 51.5 | 79.7 |
| D_beh_xseed_clean | 66.6 | 15.2 | 34.8 | 41.6 | 60.0 | 91.6 |
| D_beh_tempered | 57.7 | 15.2 | 9.5 | 26.1 | 50.9 | 79.8 |
| F_cont | 67.8 | 36.8 | 40.3 | 48.7 | 57.8 | 86.3 |
| A_cont | 66.6 | 39.1 | 41.5 | 48.1 | 58.3 | 85.5 |

## Ablation 1 — actor reset vs transplant (paired differences, * = CI excludes 0)

| contrast | AUC | +0 | +10 | +20 | +50 | +150 |
|---|---|---|---|---|---|---|
| total damage: D_online - C_cont | -17.4 [-31.2, -3.6]* | -43.4 [-70.9, -15.9]* | -32.9 [-48.9, -16.9]* | -31.2 [-48.8, -13.7]* | -15.7 [-32.2, +0.7] | -11.0 [-36.4, +14.5] |
| actor reset - D_online | -23.4 [-35.6, -11.1]* | -11.5 [-22.3, -0.8]* | -20.3 [-29.6, -11.0]* | -22.1 [-30.7, -13.4]* | -24.7 [-39.6, -9.7]* | -22.6 [-43.2, -2.1]* |
| actor transplant - D_online | +12.9 [+1.3, +24.5]* | +43.4 [+15.9, +70.9]* | +27.5 [+10.2, +44.8]* | +25.0 [+10.6, +39.4]* | +15.4 [-3.1, +33.9] | +8.4 [-7.4, +24.3] |
| actor+opt reset - D_online | -22.9 [-31.6, -14.1]* | -11.5 [-22.3, -0.8]* | -19.5 [-28.3, -10.8]* | -21.1 [-30.1, -12.1]* | -23.6 [-34.6, -12.7]* | -21.7 [-49.4, +5.9] |
| actor+opt reset - actor reset | +0.5 [-7.9, +8.9] | +0.0 [+0.0, +0.0] | +0.7 [-0.2, +1.7] | +1.0 [+0.1, +1.8]* | +1.0 [-3.9, +6.0] | +0.9 [-14.6, +16.3] |
| transplant - actor reset | +36.3 [+31.2, +41.3]* | +54.9 [+36.6, +73.3]* | +47.8 [+37.1, +58.5]* | +47.0 [+37.8, +56.3]* | +40.0 [+30.4, +49.7]* | +31.1 [+25.6, +36.5]* |
| actor reset - C_cont | -40.8 [-47.3, -34.2]* | -54.9 [-73.3, -36.6]* | -53.2 [-68.9, -37.4]* | -53.3 [-68.9, -37.6]* | -40.4 [-53.0, -27.7]* | -33.6 [-46.4, -20.8]* |
| transplant - C_cont | -4.5 [-6.9, -2.1]* | +0.0 [+0.0, +0.0] | -5.4 [-22.9, +12.2] | -6.2 [-16.0, +3.5] | -0.4 [-7.6, +6.9] | -2.6 [-16.2, +11.1] |

From-scratch reference (Phase 2 recovery-test control, fresh agent): eval 28.2 at update 50, 58.7 at update 150. Actor-reset arm at +50: 28.1, at +150: 62.9 (a fresh actor that keeps the critic + optimizer, trained from the clean env state).

## Ablation 2 — breaking the actor–experience loop (first K updates from a behaviour policy)

| contrast | AUC | +0 | +10 | +20 | +50 | +150 |
|---|---|---|---|---|---|---|
| clean behaviour - D_online | +9.6 [-6.0, +25.3] | +0.0 [+0.0, +0.0] | +19.9 [+4.5, +35.3]* | +13.9 [-4.2, +32.1] | +11.2 [-4.1, +26.6] | +0.4 [-32.2, +33.1] |
| frozen self (fixed pre-intervention data) - D_online | -2.7 [-11.8, +6.4] | +0.0 [+0.0, +0.0] | -8.6 [-15.5, -1.7]* | -4.9 [-12.3, +2.5] | -1.0 [-12.8, +10.7] | -5.2 [-33.3, +22.8] |
| x-seed clean behaviour - D_online | +4.7 [-5.0, +14.4] | +0.0 [+0.0, +0.0] | +6.5 [-3.1, +16.2] | +5.7 [-3.2, +14.7] | +7.4 [-3.2, +18.1] | +6.7 [-19.1, +32.6] |
| tempered self (KL-matched) - D_online | -4.1 [-14.0, +5.8] | +0.0 [+0.0, +0.0] | -18.7 [-27.5, -10.0]* | -9.7 [-14.0, -5.5]* | -1.6 [-15.0, +11.7] | -5.1 [-24.9, +14.7] |
| clean - x-seed clean (matchedness) | +4.9 [-7.5, +17.3] | +0.0 [+0.0, +0.0] | +13.4 [-0.0, +26.8] | +8.2 [-9.4, +25.8] | +3.8 [-2.4, +10.0] | -6.3 [-23.7, +11.1] |
| clean - tempered (quality at matched KL) | +13.7 [+3.8, +23.7]* | +0.0 [+0.0, +0.0] | +38.6 [+25.5, +51.8]* | +23.7 [+7.5, +39.9]* | +12.9 [+3.2, +22.6]* | +5.6 [-8.8, +19.9] |
| x-seed clean - tempered | +8.8 [+6.1, +11.5]* | +0.0 [+0.0, +0.0] | +25.3 [+9.5, +41.0]* | +15.5 [+3.9, +27.1]* | +9.1 [+3.2, +14.9]* | +11.9 [+2.3, +21.4]* |
| clean behaviour - actor transplant | -3.3 [-9.0, +2.4] | -43.4 [-70.9, -15.9]* | -7.6 [-18.1, +2.9] | -11.0 [-15.2, -6.8]* | -4.1 [-7.9, -0.4]* | -8.0 [-29.4, +13.4] |

Tempered behaviour: T per seed [3.85, 3.37, 4.2, 3.45]; KL target (KL(pi_C||pi_D) on bank) [0.499, 0.414, 0.436, 0.42], achieved [0.499, 0.414, 0.436, 0.42].

### Update diagnostics, mean over the K = 10 intervention updates (and the next K, after the swap ends)

Training rollouts are collected by the behaviour policy (masked logits never apply here: no deficit). episode_return = return of completed training episodes (i.e. behaviour-policy return during the swap); behaviour_kl = KL(pi_b || pi_learner) on visited states; ratio = pi_learner/pi_b (stored log-prob).

| arm | window | episode_return | behaviour_kl | ratio_mean | ratio_absdev | clip_frac | approx_kl | entropy |
|---|---|---|---|---|---|---|---|---|
| D_online | 1..10 | 25.286 | 0.000 | 0.997 | 0.065 | 0.081 | 0.007 | 0.704 |
| D_online | 11..20 | 30.854 | 0.000 | 0.998 | 0.053 | 0.059 | 0.005 | 0.632 |
| D_beh_clean | 1..10 | 57.716 | 0.144 | 0.987 | 0.247 | 0.354 | 0.109 | 0.627 |
| D_beh_clean | 11..20 | 51.139 | 0.000 | 0.998 | 0.050 | 0.057 | 0.005 | 0.570 |
| D_beh_frozen_self | 1..10 | 18.857 | 0.016 | 0.996 | 0.108 | 0.167 | 0.016 | 0.775 |
| D_beh_frozen_self | 11..20 | 25.198 | 0.000 | 0.998 | 0.056 | 0.063 | 0.005 | 0.679 |
| D_beh_xseed_clean | 1..10 | 56.857 | 0.448 | 1.017 | 0.504 | 0.526 | 0.433 | 0.662 |
| D_beh_xseed_clean | 11..20 | 40.012 | 0.000 | 0.998 | 0.049 | 0.052 | 0.004 | 0.607 |
| D_beh_tempered | 1..10 | 8.476 | 0.058 | 0.999 | 0.163 | 0.245 | 0.037 | 1.217 |
| D_beh_tempered | 11..20 | 16.266 | 0.000 | 0.998 | 0.066 | 0.056 | 0.005 | 1.016 |

## Ablation 3 — policy drift during the mask (fixed onset bank)

State at removal (update 150); KL = KL(pi_onset || pi_t) on the bank; p(masked) = summed prob of the masked actions [3];  repr drift = 1 - linear CKA vs onset; dead_a1 = fraction of a1 units never active on the bank.

| fork | eval | p(masked) | kl_from_onset | entropy | value | repr_drift_a0 | repr_drift_a1 | dead_a1 |
|---|---|---|---|---|---|---|---|---|
| onset (u100) | 45.4 | 0.495 | 0.000 | 0.573 | 8.936 | 0.000 | 0.000 | 0.002 |
| C | 58.7 | 0.488 | 0.239 | 0.555 | 10.647 | 0.035 | 0.184 | 0.000 |
| D | 15.2 | 0.364 | 0.322 | 0.750 | 1.461 | 0.067 | 0.389 | 0.000 |
| F | 36.8 | 0.509 | 0.067 | 0.578 | 1.461 | 0.000 | 0.000 | 0.002 |
| A | 39.1 | 0.513 | 0.028 | 0.609 | 1.448 | 0.013 | 0.048 | 0.000 |

Manipulation check and outcome at removal (paired differences):

- F - D: KL_onset -0.255 [-0.286, -0.225], p(masked) +0.145 [+0.058, +0.233], repr drift a1 -0.389 [-0.570, -0.209], eval at removal +21.6 [+1.1, +42.1]
- A - D: KL_onset -0.294 [-0.414, -0.174], p(masked) +0.149 [+0.082, +0.217], repr drift a1 -0.342 [-0.521, -0.163], eval at removal +23.9 [+9.7, +38.2]
- C - D: KL_onset -0.083 [-0.219, +0.054], p(masked) +0.124 [+0.033, +0.215], repr drift a1 -0.206 [-0.449, +0.037], eval at removal +43.4 [+15.9, +70.9]

## Ablation 3 — recovery after drift interventions

| contrast | AUC | +0 | +10 | +20 | +50 | +150 |
|---|---|---|---|---|---|---|
| F_cont (frozen a0/a1 during mask) - D_online | +5.9 [-6.1, +17.9] | +21.6 [+1.1, +42.1]* | +12.0 [+0.1, +24.0]* | +12.8 [-0.7, +26.4] | +5.2 [-12.9, +23.4] | +1.4 [-18.8, +21.7] |
| A_cont (anchored during mask) - D_online | +4.7 [-9.3, +18.8] | +23.9 [+9.7, +38.2]* | +13.3 [+2.6, +23.9]* | +12.2 [-4.7, +29.1] | +5.8 [-10.3, +21.8] | +0.5 [-31.8, +32.9] |
| F_cont - C_cont | -11.5 [-19.6, -3.5]* | -21.9 [-41.6, -2.1]* | -20.9 [-25.1, -16.6]* | -18.4 [-31.8, -5.0]* | -10.5 [-26.5, +5.5] | -9.6 [-18.5, -0.6]* |
| A_cont - C_cont | -12.7 [-14.3, -11.1]* | -19.5 [-39.6, +0.5] | -19.6 [-28.1, -11.2]* | -19.0 [-30.7, -7.3]* | -10.0 [-19.2, -0.7]* | -10.4 [-19.5, -1.4]* |

Across masked forks (D, F, A) x seeds (n = 12): Spearman(eval at removal, KL_onset) = -0.94 (p = 0.000); Spearman(eval at removal, p(masked)) = +0.62 (p = 0.033). Descriptive only.

KL_onset along recovery (mean over seeds) at +0 / +10 / +50 / +R:

- C_cont: 0.239 / 0.290 / 0.395 / 0.688
- D_online: 0.322 / 0.285 / 0.433 / 0.657
- D_actor_reset: 0.813 / 0.581 / 0.589 / 0.780
- D_actor_transplant: 0.239 / 0.365 / 0.457 / 0.727
- D_actor_opt_reset: 0.813 / 0.580 / 0.635 / 0.915
- D_beh_clean: 0.322 / 0.201 / 0.352 / 0.653
- D_beh_frozen_self: 0.322 / 0.287 / 0.319 / 0.594
- D_beh_xseed_clean: 0.322 / 0.347 / 0.481 / 0.749
- D_beh_tempered: 0.322 / 0.408 / 0.336 / 0.656
- F_cont: 0.067 / 0.187 / 0.326 / 0.576
- A_cont: 0.028 / 0.141 / 0.320 / 0.599

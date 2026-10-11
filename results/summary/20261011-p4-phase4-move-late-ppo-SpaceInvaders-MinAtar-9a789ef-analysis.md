# Phase 4 causal ablations — 20261011-p4-phase4-move-late-ppo-SpaceInvaders-MinAtar-9a789ef.json

Seeds [0, 1, 2, 3]; mask window updates [100, 150) (movement removed, allowed [0, 3]); recovery 150 updates; eval every 10; K = 10 behaviour updates; anchor beta = 1.0. All arms evaluated deficit-free with common random numbers. CIs: paired t over seeds.

Bit-identity of the C/D forks vs the Phase 3 saved states: {'C_params_identical': True, 'D_params_identical': True, 'C_opt_identical': True, 'D_opt_identical': True}

## Arm scores (mean over seeds)

| arm | AUC | +0 | +10 | +20 | +50 | +150 |
|---|---|---|---|---|---|---|
| C_cont | 79.3 | 58.7 | 61.2 | 67.1 | 68.3 | 95.9 |
| D_online | 65.0 | 18.5 | 34.2 | 36.7 | 53.2 | 88.2 |
| D_actor_reset | 37.7 | 3.7 | 7.7 | 13.0 | 27.4 | 61.7 |
| D_actor_transplant | 75.9 | 58.7 | 51.2 | 58.3 | 68.7 | 92.6 |
| D_actor_opt_reset | 40.5 | 3.7 | 8.1 | 13.5 | 28.1 | 67.7 |
| D_beh_clean | 73.2 | 18.5 | 47.1 | 52.4 | 64.1 | 93.8 |
| D_beh_frozen_self | 63.0 | 18.5 | 23.6 | 42.0 | 51.0 | 83.0 |
| D_beh_xseed_clean | 67.4 | 18.5 | 32.5 | 37.9 | 55.9 | 91.5 |
| D_beh_tempered | 59.8 | 18.5 | 12.1 | 31.8 | 48.6 | 86.3 |
| F_cont | 66.1 | 38.7 | 40.5 | 46.1 | 54.8 | 85.4 |
| A_cont | 64.9 | 39.6 | 35.9 | 42.0 | 54.1 | 85.3 |

## Ablation 1 — actor reset vs transplant (paired differences, * = CI excludes 0)

| contrast | AUC | +0 | +10 | +20 | +50 | +150 |
|---|---|---|---|---|---|---|
| total damage: D_online - C_cont | -14.3 [-17.5, -11.2]* | -40.1 [-58.7, -21.6]* | -26.9 [-37.9, -15.9]* | -30.4 [-42.7, -18.1]* | -15.1 [-23.9, -6.3]* | -7.6 [-18.8, +3.5] |
| actor reset - D_online | -27.2 [-39.4, -15.1]* | -14.8 [-31.3, +1.7] | -26.6 [-35.5, -17.7]* | -23.7 [-29.2, -18.1]* | -25.7 [-32.7, -18.8]* | -26.6 [-47.0, -6.2]* |
| actor transplant - D_online | +10.9 [+7.6, +14.2]* | +40.1 [+21.6, +58.7]* | +17.0 [+9.2, +24.7]* | +21.6 [+12.6, +30.7]* | +15.5 [+6.3, +24.8]* | +4.3 [-14.0, +22.7] |
| actor+opt reset - D_online | -24.5 [-26.8, -22.1]* | -14.8 [-31.3, +1.7] | -26.2 [-35.3, -17.1]* | -23.1 [-28.3, -18.0]* | -25.1 [-30.4, -19.8]* | -20.6 [-31.8, -9.4]* |
| actor+opt reset - actor reset | +2.8 [-7.6, +13.1] | +0.0 [+0.0, +0.0] | +0.4 [-0.0, +0.8] | +0.6 [-0.1, +1.2] | +0.6 [-1.8, +3.1] | +6.0 [-20.5, +32.6] |
| transplant - actor reset | +38.2 [+27.9, +48.4]* | +54.9 [+36.6, +73.3]* | +43.6 [+27.5, +59.6]* | +45.3 [+35.2, +55.5]* | +41.3 [+35.8, +46.7]* | +30.9 [-0.5, +62.3] |
| actor reset - C_cont | -41.6 [-55.6, -27.6]* | -54.9 [-73.3, -36.6]* | -53.5 [-68.8, -38.2]* | -54.1 [-70.3, -37.8]* | -40.8 [-54.9, -26.7]* | -34.2 [-63.8, -4.7]* |
| transplant - C_cont | -3.4 [-7.5, +0.7] | +0.0 [+0.0, +0.0] | -9.9 [-25.2, +5.3] | -8.7 [-19.0, +1.5] | +0.4 [-14.6, +15.4] | -3.3 [-14.5, +7.9] |

From-scratch reference (Phase 2 recovery-test control, fresh agent): eval 28.2 at update 50, 58.7 at update 150. Actor-reset arm at +50: 26.9, at +150: 59.9 (a fresh actor that keeps the critic + optimizer, trained from the clean env state).

## Ablation 2 — breaking the actor–experience loop (first K updates from a behaviour policy)

| contrast | AUC | +0 | +10 | +20 | +50 | +150 |
|---|---|---|---|---|---|---|
| clean behaviour - D_online | +8.3 [+5.6, +11.0]* | +0.0 [+0.0, +0.0] | +12.9 [+3.7, +22.1]* | +15.8 [+10.7, +20.8]* | +10.9 [+3.2, +18.7]* | +5.6 [-7.6, +18.7] |
| frozen self (fixed pre-intervention data) - D_online | -1.9 [-6.3, +2.4] | +0.0 [+0.0, +0.0] | -10.6 [-21.5, +0.3] | +5.3 [-0.9, +11.6] | -2.1 [-12.2, +8.0] | -5.3 [-19.5, +9.0] |
| x-seed clean behaviour - D_online | +2.4 [-5.6, +10.4] | +0.0 [+0.0, +0.0] | -1.8 [-13.4, +9.9] | +1.2 [-1.0, +3.4] | +2.7 [-10.0, +15.4] | +3.2 [-4.6, +11.0] |
| tempered self (KL-matched) - D_online | -5.1 [-14.3, +4.1] | +0.0 [+0.0, +0.0] | -22.1 [-30.7, -13.6]* | -4.9 [-9.7, -0.2]* | -4.6 [-9.2, +0.1] | -2.0 [-23.6, +19.7] |
| clean - x-seed clean (matchedness) | +5.9 [-3.4, +15.1] | +0.0 [+0.0, +0.0] | +14.7 [+1.8, +27.6]* | +14.5 [+7.7, +21.4]* | +8.2 [-2.4, +18.9] | +2.3 [-8.8, +13.5] |
| clean - tempered (quality at matched KL) | +13.4 [+3.4, +23.4]* | +0.0 [+0.0, +0.0] | +35.0 [+26.9, +43.2]* | +20.7 [+12.2, +29.2]* | +15.5 [+12.4, +18.6]* | +7.5 [-5.3, +20.3] |
| x-seed clean - tempered | +7.5 [-7.9, +23.0] | +0.0 [+0.0, +0.0] | +20.3 [+4.0, +36.7]* | +6.1 [+2.1, +10.2]* | +7.3 [-3.4, +18.0] | +5.2 [-13.7, +24.0] |
| clean behaviour - actor transplant | -2.6 [-6.2, +0.9] | -40.1 [-58.7, -21.6]* | -4.1 [-19.3, +11.2] | -5.9 [-18.5, +6.8] | -4.6 [-12.4, +3.2] | +1.2 [-6.3, +8.7] |

Tempered behaviour: T per seed [2.7, 3.14, 4.01, 4.59]; KL target (KL(pi_C||pi_D) on bank) [0.359, 0.452, 0.768, 0.992], achieved [0.359, 0.452, 0.768, 0.992].

### Update diagnostics, mean over the K = 10 intervention updates (and the next K, after the swap ends)

Training rollouts are collected by the behaviour policy (masked logits never apply here: no deficit). episode_return = return of completed training episodes (i.e. behaviour-policy return during the swap); behaviour_kl = KL(pi_b || pi_learner) on visited states; ratio = pi_learner/pi_b (stored log-prob).

| arm | window | episode_return | behaviour_kl | ratio_mean | ratio_absdev | clip_frac | approx_kl | entropy |
|---|---|---|---|---|---|---|---|---|
| D_online | 1..10 | 30.608 | 0.000 | 0.996 | 0.063 | 0.088 | 0.008 | 0.491 |
| D_online | 11..20 | 35.070 | 0.000 | 0.997 | 0.059 | 0.075 | 0.007 | 0.555 |
| D_beh_clean | 1..10 | 57.716 | 0.145 | 0.988 | 0.237 | 0.346 | 0.102 | 0.619 |
| D_beh_clean | 11..20 | 48.075 | 0.000 | 0.998 | 0.058 | 0.073 | 0.006 | 0.596 |
| D_beh_frozen_self | 1..10 | 21.441 | 0.024 | 0.995 | 0.101 | 0.159 | 0.021 | 0.452 |
| D_beh_frozen_self | 11..20 | 31.778 | 0.000 | 0.997 | 0.059 | 0.075 | 0.006 | 0.532 |
| D_beh_xseed_clean | 1..10 | 56.857 | 0.456 | 1.015 | 0.512 | 0.543 | 0.431 | 0.668 |
| D_beh_xseed_clean | 11..20 | 36.534 | 0.000 | 0.998 | 0.059 | 0.071 | 0.006 | 0.653 |
| D_beh_tempered | 1..10 | 11.448 | 0.094 | 0.998 | 0.161 | 0.251 | 0.056 | 0.950 |
| D_beh_tempered | 11..20 | 20.626 | 0.000 | 0.998 | 0.067 | 0.069 | 0.006 | 0.890 |

## Ablation 3 — policy drift during the mask (fixed onset bank)

State at removal (update 150); KL = KL(pi_onset || pi_t) on the bank; p(masked) = summed prob of the masked actions [1, 2];  repr drift = 1 - linear CKA vs onset; dead_a1 = fraction of a1 units never active on the bank.

| fork | eval | p(masked) | kl_from_onset | entropy | value | repr_drift_a0 | repr_drift_a1 | dead_a1 |
|---|---|---|---|---|---|---|---|---|
| onset (u100) | 45.4 | 0.460 | 0.000 | 0.573 | 8.936 | 0.000 | 0.000 | 0.002 |
| C | 58.7 | 0.461 | 0.239 | 0.555 | 10.647 | 0.035 | 0.184 | 0.000 |
| D | 18.5 | 0.383 | 0.383 | 0.476 | 4.151 | 0.098 | 0.246 | 0.002 |
| F | 38.7 | 0.416 | 0.037 | 0.529 | 4.042 | 0.000 | 0.000 | 0.002 |
| A | 39.6 | 0.444 | 0.020 | 0.576 | 4.062 | 0.018 | 0.017 | 0.002 |

Manipulation check and outcome at removal (paired differences):

- F - D: KL_onset -0.346 [-0.713, +0.021], p(masked) +0.033 [-0.119, +0.186], repr drift a1 -0.246 [-0.444, -0.048], eval at removal +20.1 [+8.2, +32.0]
- A - D: KL_onset -0.363 [-0.799, +0.072], p(masked) +0.061 [-0.124, +0.246], repr drift a1 -0.229 [-0.434, -0.023], eval at removal +21.1 [+13.1, +29.0]
- C - D: KL_onset -0.144 [-0.584, +0.296], p(masked) +0.078 [-0.144, +0.300], repr drift a1 -0.062 [-0.291, +0.166], eval at removal +40.1 [+21.6, +58.7]

## Ablation 3 — recovery after drift interventions

| contrast | AUC | +0 | +10 | +20 | +50 | +150 |
|---|---|---|---|---|---|---|
| F_cont (frozen a0/a1 during mask) - D_online | +1.1 [-3.4, +5.7] | +20.1 [+8.2, +32.0]* | +6.3 [-4.2, +16.8] | +9.4 [+2.0, +16.7]* | +1.6 [-9.6, +12.8] | -2.9 [-10.3, +4.6] |
| A_cont (anchored during mask) - D_online | -0.0 [-1.8, +1.7] | +21.1 [+13.1, +29.0]* | +1.7 [-4.7, +8.0] | +5.3 [+1.5, +9.2]* | +0.9 [-4.5, +6.3] | -3.0 [-12.4, +6.4] |
| F_cont - C_cont | -13.2 [-20.0, -6.4]* | -20.0 [-33.5, -6.5]* | -20.6 [-27.7, -13.6]* | -21.0 [-36.7, -5.3]* | -13.5 [-26.3, -0.7]* | -10.5 [-19.4, -1.6]* |
| A_cont - C_cont | -14.4 [-18.5, -10.3]* | -19.0 [-36.0, -2.1]* | -25.2 [-34.1, -16.4]* | -25.0 [-40.9, -9.2]* | -14.2 [-26.7, -1.7]* | -10.6 [-19.8, -1.5]* |

Across masked forks (D, F, A) x seeds (n = 12): Spearman(eval at removal, KL_onset) = -0.80 (p = 0.002); Spearman(eval at removal, p(masked)) = +0.46 (p = 0.134). Descriptive only.

KL_onset along recovery (mean over seeds) at +0 / +10 / +50 / +R:

- C_cont: 0.239 / 0.290 / 0.395 / 0.688
- D_online: 0.383 / 0.384 / 0.353 / 0.651
- D_actor_reset: 0.813 / 0.618 / 0.630 / 1.014
- D_actor_transplant: 0.239 / 0.317 / 0.410 / 0.692
- D_actor_opt_reset: 0.813 / 0.626 / 0.703 / 1.072
- D_beh_clean: 0.383 / 0.195 / 0.354 / 0.623
- D_beh_frozen_self: 0.383 / 0.330 / 0.312 / 0.629
- D_beh_xseed_clean: 0.383 / 0.349 / 0.504 / 0.809
- D_beh_tempered: 0.383 / 0.267 / 0.298 / 0.685
- F_cont: 0.037 / 0.181 / 0.315 / 0.587
- A_cont: 0.020 / 0.154 / 0.289 / 0.578

# Phase 7 point-of-no-return sweep — p7_ponr_pilot

Source: `20261013-p7-ponr-pilot-move-ppo-SpaceInvaders-MinAtar-ba1802f.json`. Seeds [0, 1, 2, 3] (n = 4). Onset w0 = 100; durations [50, 100, 200, 300]; budget R = 250 updates; eval every 10. Restriction: allowed actions [0, 3].

Delay-adjusted reference = clean trajectory at the same number of updates *after onset* (pause-equivalent: what the agent would have if training had simply been paused during the window). Negative = damage beyond lost time. Paired-t 95% CIs; * = CI excludes 0.

## Rescue curve (own recovery and transplant rescue)

| d (updates) | eval at removal D vs C | **L_d late delay-adj.** | A_d delay-adj. AUC | raw AUC vs C(w1+k) | transplant − D_online AUC | transplant vs C(w1+k) AUC | transplant late delay-adj. |
|---|---|---|---|---|---|---|---|
| 50 | 18.5 vs 58.7 | **-0.4 [-9.5, +8.7]** | -0.3 [-3.7, +3.1] | -11.5 [-14.2, -8.9]* | +8.3 [+3.4, +13.3]* | -3.2 [-6.8, +0.5] | +2.9 [-6.6, +12.3] |
| 100 | 15.3 vs 70.4 | **+4.8 [-4.3, +13.8]** | -1.5 [-7.0, +4.0] | -22.6 [-32.0, -13.2]* | +18.2 [+3.7, +32.7]* | -4.4 [-9.5, +0.7] | +11.1 [+1.4, +20.9]* |
| 200 | 13.4 vs 99.3 | **+2.3 [-7.2, +11.7]** | -0.8 [-3.0, +1.3] | -33.9 [-45.4, -22.4]* | +26.7 [+18.8, +34.7]* | -7.1 [-14.2, -0.1]* | +17.1 [+8.5, +25.7]* |
| 300 | 7.3 vs 111.3 | **-1.4 [-8.2, +5.5]** | -7.6 [-10.6, -4.6]* | -48.8 [-63.2, -34.5]* | +42.8 [+27.4, +58.1]* | -6.1 [-14.2, +2.1] | +26.4 [+17.0, +35.9]* |

Per-seed L_d:

- d = 50: +0.8, +5.1, +1.0, -8.4
- d = 100: +12.8, -0.1, +4.7, +1.7
- d = 200: +10.3, +1.9, +0.8, -4.0
- d = 300: +3.1, -6.3, -3.5, +1.2

## Stage 0 gate (rule fixed in LOG.md before the run)

Seed-mean L_d by d: 50: -0.4, 100: +4.8, 200: +2.3, 300: -1.4
Monotone decrease: False.

Decision: **STOP** (no persistent impairment beyond lost time within the budget)

## Predictors at removal: D and clean-matched difference D − C (seed means)

| probe | D, d=50 | D, d=100 | D, d=200 | D, d=300 | D−C, d=50 | D−C, d=100 | D−C, d=200 | D−C, d=300 |
|---|---|---|---|---|---|---|---|---|
| kl_onset | 0.383 | 0.793 | 0.530 | 0.843 | +0.144 | +0.398 | -0.158 | -0.084 |
| entropy | 0.476 | 0.394 | 0.472 | 0.424 | -0.079 | -0.150 | -0.059 | -0.105 |
| p_masked | 0.383 | 0.311 | 0.372 | 0.357 | -0.078 | -0.135 | -0.093 | -0.085 |
| kl_masked_cond | 0.115 | 0.114 | 0.178 | 0.274 | -0.120 | -0.239 | -0.409 | -0.452 |
| masked_logit_corr | 0.926 | 0.893 | 0.830 | 0.780 | +0.027 | +0.037 | +0.038 | +0.020 |
| decode_r2_h1 | 0.977 | 0.962 | 0.932 | 0.905 | +0.064 | +0.100 | +0.117 | +0.132 |
| decode_r2_h0 | 0.978 | 0.969 | 0.953 | 0.936 | +0.057 | +0.094 | +0.116 | +0.138 |
| repr_drift_a0 | 0.098 | 0.160 | 0.155 | 0.180 | +0.063 | +0.094 | +0.060 | +0.059 |
| repr_drift_a1 | 0.246 | 0.366 | 0.473 | 0.548 | +0.062 | +0.110 | +0.130 | +0.170 |
| srank_a1 | 72.203 | 72.597 | 75.573 | 74.568 | -1.151 | -1.784 | +0.210 | -1.871 |
| dead_a1 | 0.002 | 0.002 | 0.000 | 0.000 | +0.002 | -0.002 | -0.006 | +0.000 |
| value_corr | 0.193 | 0.193 | 0.186 | 0.171 | -0.770 | -0.733 | -0.709 | -0.699 |
| value_nmse | 3.707 | 3.706 | 3.747 | 3.825 | +3.286 | +2.592 | +1.527 | +0.856 |
| actor_rel_change | 0.077 | 0.111 | 0.174 | 0.210 | -0.154 | -0.215 | -0.292 | -0.364 |
| critic_rel_change | 0.079 | 0.079 | 0.083 | 0.095 | -0.188 | -0.299 | -0.425 | -0.505 |

## Descriptive association with outcomes over seed × d units (n = 16; nested, NOT inferential)

Spearman rho. 'within-d' = rho after removing the per-duration mean from both variables (does the probe explain seed-to-seed differences at the same duration?).

| variable | rho with L_d | rho with A_d | within-d rho with A_d |
|---|---|---|---|
| d | -0.08 | -0.62 | +nan |
| acute | +0.09 | +0.64 | +0.17 |
| D_kl_onset | -0.37 | -0.44 | -0.19 |
| D_entropy | -0.07 | +0.09 | -0.07 |
| D_p_masked | +0.09 | -0.06 | -0.14 |
| D_kl_masked_cond | -0.04 | -0.64 | -0.30 |
| D_masked_logit_corr | +0.16 | +0.63 | +0.14 |
| D_decode_r2_h1 | +0.22 | +0.56 | -0.07 |
| D_decode_r2_h0 | +0.21 | +0.55 | +0.20 |
| D_repr_drift_a0 | -0.08 | -0.30 | +0.01 |
| D_repr_drift_a1 | -0.18 | -0.45 | -0.10 |
| D_srank_a1 | +0.11 | +0.05 | +0.45 |
| D_dead_a1 | +0.08 | +0.04 | -0.15 |
| D_value_corr | -0.02 | +0.14 | -0.09 |
| D_value_nmse | -0.03 | -0.09 | +0.14 |
| D_actor_rel_change | +0.07 | -0.56 | +0.01 |
| D_critic_rel_change | +0.10 | -0.48 | -0.05 |
| dC_kl_onset | -0.11 | +0.18 | -0.16 |
| dC_entropy | +0.13 | +0.02 | -0.07 |
| dC_p_masked | +0.03 | -0.17 | -0.13 |
| dC_kl_masked_cond | +0.25 | +0.32 | -0.19 |
| dC_masked_logit_corr | -0.07 | +0.27 | +0.05 |
| dC_decode_r2_h1 | -0.10 | -0.44 | -0.23 |
| dC_decode_r2_h0 | -0.16 | -0.63 | -0.12 |
| dC_repr_drift_a0 | +0.05 | -0.01 | +0.03 |
| dC_repr_drift_a1 | +0.01 | -0.27 | +0.00 |
| dC_srank_a1 | +0.31 | +0.32 | +0.31 |
| dC_dead_a1 | +0.20 | +0.07 | +0.31 |
| dC_value_corr | -0.15 | -0.13 | -0.19 |
| dC_value_nmse | +0.18 | +0.47 | +0.16 |
| dC_actor_rel_change | +0.21 | +0.59 | +0.00 |
| dC_critic_rel_change | +0.09 | +0.62 | +0.28 |

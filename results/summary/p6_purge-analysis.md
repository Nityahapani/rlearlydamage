# Phase 6 replay purge — p6_purge

Manifests: 20261012-p6-purge-s0-pqnr-1000k-SpaceInvaders-MinAtar-864697e.json, 20261012-p6-purge-s1-pqnr-1000k-SpaceInvaders-MinAtar-864697e.json, 20261012-p6-purge-s2-pqnr-1000k-SpaceInvaders-MinAtar-864697e.json, 20261012-p6-purge-s3-pqnr-1000k-SpaceInvaders-MinAtar-864697e.json, 20261012-p6-purge-s4-pqnr-1000k-SpaceInvaders-MinAtar-864697e.json, 20261012-p6-purge-s5-pqnr-1000k-SpaceInvaders-MinAtar-864697e.json, 20261012-p6-purge-s6-pqnr-1000k-SpaceInvaders-MinAtar-864697e.json, 20261012-p6-purge-s7-pqnr-1000k-SpaceInvaders-MinAtar-864697e.json.
Seeds (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Metric: eval_eps05_return_mean. Removal at u350; recovery 650; eval every 10; delayed purge at +50.
Consistency checks: [{'D_clean_matches_1M_control_run': {'seeds': [0], 'identical': True}, 'forks_match_phase6_surgery': {'0': True}}, {'D_clean_matches_1M_control_run': {'seeds': [1], 'identical': True}, 'forks_match_phase6_surgery': {'1': True}}, {'D_clean_matches_1M_control_run': {'seeds': [2], 'identical': True}, 'forks_match_phase6_surgery': {'2': True}}, {'D_clean_matches_1M_control_run': {'seeds': [3], 'identical': True}, 'forks_match_phase6_surgery': {'3': True}}, {'D_clean_matches_1M_control_run': {'seeds': [4], 'identical': True}, 'forks_match_phase6_surgery': {'4': True}}, {'D_clean_matches_1M_control_run': {'seeds': [5], 'identical': True}, 'forks_match_phase6_surgery': {'5': True}}, {'D_clean_matches_1M_control_run': {'seeds': [6], 'identical': True}, 'forks_match_phase6_surgery': {'6': True}}, {'D_clean_matches_1M_control_run': {'seeds': [7], 'identical': True}, 'forks_match_phase6_surgery': {'7': True}}].
Replacement buffer (frozen clean actor, eps 0.05): collection-episode mean return [[27.23900032043457], [29.875999450683594], [26.448999404907227], [28.61199951171875], [30.350000381469727], [26.34600067138672], [27.2189998626709], [26.94300079345703]].

Windows (post-removal updates): AUC +10..+650; early +10..+150; mid +160..+240; late +250..+650; post-delay +60..+150. +h readouts average checkpoints h-E..h+E.

## Arm scores (mean over seeds)

| arm | AUC | early | mid | late | post-delay | +10 | +100 |
|---|---|---|---|---|---|---|---|
| A_retain | 33.3 | 24.3 | 27.8 | 37.8 | 25.8 | 18.0 | 24.8 |
| B_purge | 37.8 | 24.2 | 31.2 | 44.3 | 27.9 | 9.5 | 27.8 |
| C_delay50 | 36.8 | 24.3 | 29.3 | 43.0 | 25.8 | 18.0 | 25.6 |
| D_clean | 37.9 | 28.0 | 30.6 | 43.2 | 28.5 | 27.3 | 29.4 |
| E_clean_purge | 39.7 | 27.1 | 33.5 | 45.7 | 27.6 | 26.0 | 27.7 |

## Primary contrasts (paired; positive = first arm better)

| contrast | AUC | early | mid | late | post-delay | +10 | +100 | AUC p (Holm) |
|---|---|---|---|---|---|---|---|---|
| purge rescue: B - A | +4.6 [+2.2, +6.9]* | -0.1 [-0.7, +0.5] | +3.3 [+1.3, +5.4]* | +6.5 [+3.1, +10.0]* | +2.1 [+1.3, +2.8]* | -8.5 [-12.2, -4.8]* | +3.0 [+0.8, +5.3]* | 0.002 (0.007) |
| delayed-purge effect: B - C | +1.0 [-0.1, +2.1] | -0.1 [-1.1, +1.0] | +1.8 [-0.5, +4.1] | +1.2 [-0.5, +3.0] | +2.1 [+0.3, +3.9]* | -8.5 [-12.2, -4.8]* | +2.2 [-0.3, +4.8] | 0.075 (0.149) |
| residual impairment: B - D | -0.1 [-2.5, +2.3] | -3.8 [-5.1, -2.4]* | +0.5 [-1.6, +2.7] | +1.1 [-2.1, +4.3] | -0.6 [-2.6, +1.3] | -17.8 [-22.3, -13.3]* | -1.5 [-3.0, -0.1]* | 0.906 (0.906) |

## Secondary contrasts

| contrast | AUC | early | mid | late | post-delay | +10 | +100 |
|---|---|---|---|---|---|---|---|
| purge procedure on clean net: E - D | +1.8 [-1.2, +4.8] | -0.9 [-2.3, +0.5] | +2.8 [-1.2, +6.9] | +2.5 [-1.3, +6.3] | -0.9 [-2.7, +1.0] | -1.2 [-5.7, +3.2] | -1.7 [-3.8, +0.4] |
| damage-specific purge benefit: (B - A) - (E - D) | +2.8 [-0.2, +5.7] | +0.8 [-1.1, +2.7] | +0.5 [-2.6, +3.5] | +4.0 [-0.0, +8.1] | +3.0 [+1.2, +4.7]* | -7.2 [-13.8, -0.7]* | +4.7 [+2.4, +7.0]* |
| C - A | +3.5 [+1.6, +5.5]* | -0.0 [-0.8, +0.8] | +1.5 [-1.9, +4.9] | +5.3 [+2.3, +8.3]* | -0.0 [-1.2, +1.2] | +0.0 [+0.0, +0.0] | +0.8 [-1.5, +3.1] |
| total damage, retained buffer: A - D | -4.7 [-6.9, -2.4]* | -3.7 [-5.3, -2.1]* | -2.8 [-3.8, -1.8]* | -5.5 [-8.7, -2.2]* | -2.7 [-4.3, -1.1]* | -9.3 [-15.0, -3.6]* | -4.6 [-6.1, -3.0]* |

## Seed robustness (AUC, n = 8)

| contrast | mean | between-seed SD | seeds < 0 | 95% prediction interval | per-seed (0, 1, 2, 3, 4, 5, 6, 7) |
|---|---|---|---|---|---|
| purge rescue: B - A | +4.6 | 2.8 | 0/8 | [-2.4, +11.5] | +0.0, +3.7, +4.9, +5.1, +8.2, +6.6, +6.6, +1.3 |
| delayed-purge effect: B - C | +1.0 | 1.4 | 3/8 | [-2.4, +4.4] | -0.2, +1.5, -0.1, +1.9, +2.4, -0.6, +3.0, +0.1 |
| residual impairment: B - D | -0.1 | 2.9 | 6/8 | [-7.3, +7.1] | -2.0, +1.2, -0.4, -1.9, +6.5, -1.0, -1.7, -1.7 |
| purge procedure on clean net: E - D | +1.8 | 3.6 | 3/8 | [-7.2, +10.8] | +1.4, +4.4, +0.2, -0.4, +9.3, +1.8, -1.4, -1.3 |
| damage-specific purge benefit: (B - A) - (E - D) | +2.8 | 3.5 | 3/8 | [-6.1, +11.6] | -1.4, -0.7, +4.7, +5.5, -1.1, +4.8, +8.0, +2.5 |
| C - A | +3.5 | 2.4 | 0/8 | [-2.4, +9.5] | +0.2, +2.2, +5.0, +3.2, +5.9, +7.2, +3.6, +1.2 |
| total damage, retained buffer: A - D | -4.7 | 2.7 | 8/8 | [-11.4, +2.1] | -2.0, -2.5, -5.3, -6.9, -1.8, -7.6, -8.3, -3.0 |

## Restriction-era share of each buffer at eval checkpoints (mean over seeds)

| arm | +0 | +10 | +50 | +60 | +100 | +150 | +200 | +250 | +300 |
|---|---|---|---|---|---|---|---|---|---|
| A_retain | 0.410 | 0.410 | 0.410 | 0.410 | 0.410 | 0.386 | 0.181 | 0.000 | 0.000 |
| B_purge | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C_delay50 | 0.410 | 0.410 | 0.410 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D_clean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| E_clean_purge | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D_clean (window-era, unrestricted) | 0.410 | 0.410 | 0.410 | 0.410 | 0.410 | 0.386 | 0.181 | 0.000 | 0.000 |
| E_clean_purge (window-era, unrestricted) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Diagnostic pattern (AUC means; CIs above)

- B > A: True (37.8 vs 33.3)
- A <= C <= B: True (C 36.8)
- B < D: True (D 37.9)

TD loss, first 30 post-removal updates (mean): A_retain 0.885, B_purge 0.696, C_delay50 0.885, D_clean 0.074, E_clean_purge 0.121

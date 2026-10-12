# Phase 6c identity x residence 2x2 — p6c_2x2

Phase 6b manifests: 20261012-p6b-tpurge-s0-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s1-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s10-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s11-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s2-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s3-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s4-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s5-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s6-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s7-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s8-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s9-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json.
Phase 6c manifests: 20261012-p6c-res2x2-s0-pqnr-1000k-SpaceInvaders-MinAtar-2559f2e.json, 20261012-p6c-res2x2-s1-pqnr-1000k-SpaceInvaders-MinAtar-2559f2e.json, 20261012-p6c-res2x2-s10-pqnr-1000k-SpaceInvaders-MinAtar-5ca0172.json, 20261012-p6c-res2x2-s11-pqnr-1000k-SpaceInvaders-MinAtar-5ca0172.json, 20261012-p6c-res2x2-s2-pqnr-1000k-SpaceInvaders-MinAtar-eb35453.json, 20261012-p6c-res2x2-s3-pqnr-1000k-SpaceInvaders-MinAtar-eb35453.json, 20261012-p6c-res2x2-s4-pqnr-1000k-SpaceInvaders-MinAtar-eb35453.json, 20261012-p6c-res2x2-s5-pqnr-1000k-SpaceInvaders-MinAtar-3ddb28f.json, 20261012-p6c-res2x2-s6-pqnr-1000k-SpaceInvaders-MinAtar-3ddb28f.json, 20261012-p6c-res2x2-s7-pqnr-1000k-SpaceInvaders-MinAtar-3ddb28f.json, 20261012-p6c-res2x2-s8-pqnr-1000k-SpaceInvaders-MinAtar-5ca0172.json, 20261012-p6c-res2x2-s9-pqnr-1000k-SpaceInvaders-MinAtar-5ca0172.json.
Seeds (n = 12): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Metric: eval_eps05_return_mean. Removal u350; recovery 650; eval every 10.
Consistency checks vs Phase 6b (all must be True): {'forks_match_p6b': {'0': True, '1': True, '10': True, '11': True, '2': True, '3': True, '4': True, '5': True, '6': True, '7': True, '8': True, '9': True}, 'pool_stats_match_p6b': {'0': True, '1': True, '10': True, '11': True, '2': True, '3': True, '4': True, '5': True, '6': True, '7': True, '8': True, '9': True}, 'blocks_match_p6b': {'0': True, '1': True, '10': True, '11': True, '2': True, '3': True, '4': True, '5': True, '6': True, '7': True, '8': True, '9': True}, 'B_targeted_matches_p6b': {'0': True}}.

Layout (positions P2 = rows written u150-249, P1 = newest rows u250-349): B = [P | pool], Bp = [pool | P], Cp = [R | pool], C = [pool | R].

## Cell scores (mean over seeds)

| cell | removed | replacement | AUC | early | middle | late |
|---|---|---|---|---|---|---|
| B_targeted | R | long | 36.7 | 25.2 | 29.7 | 42.4 |
| Bp_targeted_short | R | short | 35.2 | 25.9 | 28.7 | 40.0 |
| Cp_sham_long | P | long | 35.5 | 24.5 | 29.0 | 41.0 |
| C_sham | P | short | 34.1 | 23.6 | 27.9 | 39.2 |
| A_retain | none (ref) | - | 33.1 | 24.2 | 27.6 | 37.6 |
| D_cleannet | R, clean net (ref) | long | 39.5 | 28.3 | 32.5 | 45.2 |

## Primary contrasts (Holm across the three AUC tests)

| contrast | AUC | early | middle | late | AUC t p | Holm p | sign-flip p | seeds > 0 |
|---|---|---|---|---|---|---|---|---|
| identity at matched long residence: B - Cp | +1.1 [-0.2, +2.5] | +0.7 [+0.1, +1.4]* | +0.7 [-1.0, +2.3] | +1.4 [-0.7, +3.5] | 0.0899 | 0.1797 | 0.0972 | 8/12 |
| identity at matched short residence: Bp - C | +1.1 [-0.8, +3.0] | +2.2 [+1.2, +3.2]* | +0.9 [-0.7, +2.4] | +0.7 [-1.9, +3.3] | 0.2259 | 0.2259 | 0.2505 | 8/12 |
| residence main effect: ((B - Bp) + (Cp - C)) / 2 | +1.5 [+0.2, +2.7]* | +0.0 [-0.4, +0.5] | +1.1 [+0.2, +1.9]* | +2.1 [+0.1, +4.1]* | 0.0242 | 0.0726 | 0.0269 | 8/12 |

## Secondary (descriptive)

| contrast | AUC | early | middle | late | sign-flip p | seeds > 0 |
|---|---|---|---|---|---|---|
| identity main effect: ((B - Cp) + (Bp - C)) / 2 | +1.1 [+0.0, +2.2]* | +1.5 [+0.8, +2.2]* | +0.8 [-0.3, +1.9] | +1.0 [-0.4, +2.5] | 0.0479 | 9/12 |
| interaction: (B - Cp) - (Bp - C) | +0.0 [-2.3, +2.4] | -1.5 [-2.5, -0.6]* | -0.2 [-2.4, +2.1] | +0.7 [-3.0, +4.4] | 0.9746 | 7/12 |
| residence | R removed: B - Bp | +1.5 [-0.1, +3.1] | -0.7 [-1.3, -0.2]* | +1.0 [-0.6, +2.6] | +2.4 [-0.0, +4.9] | 0.0674 | 8/12 |
| residence | P removed: Cp - C | +1.5 [-0.4, +3.3] | +0.8 [+0.1, +1.5]* | +1.2 [-0.0, +2.3] | +1.8 [-1.2, +4.7] | 0.1079 | 8/12 |
| Phase 6b reference: B - C | +2.6 [+0.9, +4.3]* | +1.5 [+0.7, +2.3]* | +1.9 [+0.5, +3.2]* | +3.2 [+0.7, +5.6]* | 0.0088 | 9/12 |

Decomposition of Phase 6b B - C (+2.6) = identity at long residence B - Cp (+1.1) + residence given P removed Cp - C (+1.5).

## Paired seed-level differences (AUC)

| seed | B - Cp | Bp - C | ((B - Bp) + (Cp - C)) / 2 | ((B - Cp) + (Bp - C)) / 2 | (B - Cp) - (Bp - C) | B - Bp | Cp - C | B - C |
|---|---|---|---|---|---|---|---|---|
| 0 | +3.5 | +0.4 | +3.0 | +2.0 | +3.1 | +4.6 | +1.5 | +5.0 |
| 1 | +4.4 | +4.3 | -0.5 | +4.4 | +0.1 | -0.4 | -0.5 | +3.9 |
| 2 | -0.2 | -1.5 | +1.5 | -0.9 | +1.3 | +2.1 | +0.8 | +0.6 |
| 3 | -1.2 | +8.7 | +2.8 | +3.8 | -9.9 | -2.1 | +7.8 | +6.6 |
| 4 | +1.7 | +0.8 | +1.9 | +1.3 | +0.9 | +2.4 | +1.5 | +3.2 |
| 5 | +1.0 | -0.3 | +3.4 | +0.3 | +1.3 | +4.0 | +2.8 | +3.7 |
| 6 | +0.7 | -2.8 | -0.5 | -1.1 | +3.4 | +1.3 | -2.2 | -1.5 |
| 7 | +0.6 | +1.3 | -2.1 | +1.0 | -0.7 | -2.4 | -1.8 | -1.1 |
| 8 | -0.1 | +0.8 | +4.7 | +0.3 | -0.9 | +4.2 | +5.1 | +5.0 |
| 9 | -2.0 | +0.6 | -0.1 | -0.7 | -2.6 | -1.4 | +1.3 | -0.8 |
| 10 | +0.6 | +0.9 | +2.7 | +0.7 | -0.3 | +2.5 | +2.8 | +3.4 |
| 11 | +4.7 | -0.2 | +1.0 | +2.2 | +4.9 | +3.4 | -1.5 | +3.2 |
| between-seed SD | 2.1 | 2.9 | 2.0 | 1.7 | 3.8 | 2.5 | 2.9 | 2.7 |

## Exposure: restricted-era share / replacement share at checkpoints (mean over seeds)

| cell | +0 | +20 | +40 | +60 | +100 | +140 | +150 | +200 | +250 |
|---|---|---|---|---|---|---|---|---|---|
| B_targeted | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.39 | 0.00 / 0.18 | 0.00 / 0.00 |
| Bp_targeted_short | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.34 | 0.00 / 0.18 | 0.00 / 0.02 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Cp_sham_long | 0.41 / 0.41 | 0.41 / 0.41 | 0.41 / 0.41 | 0.34 / 0.41 | 0.18 / 0.41 | 0.02 / 0.41 | 0.00 / 0.39 | 0.00 / 0.18 | 0.00 / 0.00 |
| C_sham | 0.41 / 0.41 | 0.41 / 0.41 | 0.41 / 0.41 | 0.41 / 0.34 | 0.41 / 0.18 | 0.41 / 0.02 | 0.39 / 0.00 | 0.18 / 0.00 | 0.00 / 0.00 |
| A_retain | 0.41 / 0.00 | 0.41 / 0.00 | 0.41 / 0.00 | 0.41 / 0.00 | 0.41 / 0.00 | 0.41 / 0.00 | 0.39 / 0.00 | 0.18 / 0.00 | 0.00 / 0.00 |
| D_cleannet | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.39 | 0.00 / 0.18 | 0.00 / 0.00 |

Cumulative expected sampled transitions over +1..+650 (millions): B_targeted: restricted-era 0.00, replacement 0.65; Bp_targeted_short: restricted-era 0.00, replacement 0.31; Cp_sham_long: restricted-era 0.31, replacement 0.65; C_sham: restricted-era 0.65, replacement 0.31.
Max |replacement-share difference| B_targeted vs Cp_sham_long: 0.0000 (residence-matched pair).
Max |replacement-share difference| Bp_targeted_short vs C_sham: 0.0000 (residence-matched pair).

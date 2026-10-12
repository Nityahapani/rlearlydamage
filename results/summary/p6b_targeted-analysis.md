# Phase 6b targeted vs sham replay purge — p6b_targeted

Manifests: 20261012-p6b-tpurge-s0-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s1-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s2-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s3-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s4-pqnr-1000k-SpaceInvaders-MinAtar-250cec4.json, 20261012-p6b-tpurge-s5-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s6-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s7-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s8-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s9-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s10-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json, 20261012-p6b-tpurge-s11-pqnr-1000k-SpaceInvaders-MinAtar-1c06f18.json.
Seeds (n = 12): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Metric: eval_eps05_return_mean. Removal u350; recovery 650; eval every 10.
Replaced rows: K = 3200 rows x 128 envs; B = rows written u[250, 349], C (sham) = rows written u[150, 249].
Consistency checks: [{'A_retain_matches_phase6_purge': {'0': True}, 'forks_match_phase6_surgery': {'0': True}}, {'A_retain_matches_phase6_purge': {'1': True}, 'forks_match_phase6_surgery': {'1': True}}, {'A_retain_matches_phase6_purge': {'2': True}, 'forks_match_phase6_surgery': {'2': True}}, {'A_retain_matches_phase6_purge': {'3': True}, 'forks_match_phase6_surgery': {'3': True}}, {'A_retain_matches_phase6_purge': {'4': True}, 'forks_match_phase6_surgery': {'4': True}}, {'A_retain_matches_phase6_purge': {'5': True}, 'forks_match_phase6_surgery': {'5': True}}, {'A_retain_matches_phase6_purge': {'6': True}, 'forks_match_phase6_surgery': {'6': True}}, {'A_retain_matches_phase6_purge': {'7': True}, 'forks_match_phase6_surgery': {'7': True}}, {}, {}, {}, {}].
Pool collection-episode mean return per seed: [[26.753000259399414], [29.139999389648438], [26.01300048828125], [28.11199951171875], [30.07699966430664], [26.113000869750977], [26.92799949645996], [26.43600082397461], [25.702999114990234], [24.739999771118164], [28.0939998626709], [27.06800079345703]].

## Arm scores (mean over seeds)

| arm | AUC | early | middle | late |
|---|---|---|---|---|
| A_retain | 33.1 | 24.2 | 27.6 | 37.6 |
| B_targeted | 36.7 | 25.2 | 29.7 | 42.4 |
| C_sham | 34.1 | 23.6 | 27.9 | 39.2 |
| D_cleannet | 39.5 | 28.3 | 32.5 | 45.2 |

## Primary contrasts (paired; positive = first arm better). Primary endpoint AUC; windows secondary.

| contrast | AUC | early | middle | late | AUC t-test p | Holm p | sign-flip p | seeds > 0 |
|---|---|---|---|---|---|---|---|---|
| targeted vs sham: B - C | +2.6 [+0.9, +4.3]* | +1.5 [+0.7, +2.3]* | +1.9 [+0.5, +3.2]* | +3.2 [+0.7, +5.6]* | 0.0063 | 0.0063 | 0.0088 | 9/12 |
| targeted purge vs retain: B - A | +3.5 [+1.9, +5.2]* | +1.0 [+0.2, +1.8]* | +2.2 [+0.6, +3.7]* | +4.8 [+2.3, +7.3]* | 0.0006 | 0.0012 | 0.0024 | 10/12 |
| residual network impairment: B - D | -2.9 [-4.0, -1.8]* | -3.2 [-4.0, -2.4]* | -2.8 [-4.7, -0.9]* | -2.8 [-4.4, -1.1]* | 0.0001 | 0.0004 | 0.0010 | 1/12 |

## Secondary

| contrast | AUC | early | middle | late |
|---|---|---|---|---|
| sham refresh vs retain: C - A | +0.9 [-0.8, +2.7] | -0.5 [-1.2, +0.2] | +0.3 [-0.8, +1.4] | +1.6 [-0.9, +4.1] |

Share of the targeted-purge benefit attributable to targeting (B - C) / (B - A), AUC means: 73%.

## Paired seed-level differences (AUC)

| seed | targeted vs sham: B - C | targeted purge vs retain: B - A | residual network impairment: B - D | sham refresh vs retain: C - A |
|---|---|---|---|---|
| 0 | +5.0 | +1.6 | +1.9 | -3.5 |
| 1 | +3.9 | +3.3 | -3.7 | -0.6 |
| 2 | +0.6 | +2.4 | -3.0 | +1.8 |
| 3 | +6.6 | +2.9 | -3.6 | -3.7 |
| 4 | +3.2 | +5.4 | -3.3 | +2.3 |
| 5 | +3.7 | +5.7 | -2.5 | +2.0 |
| 6 | -1.5 | +4.6 | -3.8 | +6.1 |
| 7 | -1.1 | -1.2 | -3.4 | -0.0 |
| 8 | +5.0 | +6.7 | -3.8 | +1.6 |
| 9 | -0.8 | -0.5 | -5.1 | +0.3 |
| 10 | +3.4 | +5.8 | -1.9 | +2.4 |
| 11 | +3.2 | +5.8 | -2.0 | +2.6 |
| between-seed SD | 2.7 | 2.6 | 1.7 | 2.7 |

## Buffer composition at eval checkpoints (mean over seeds): restricted-era share / replacement share

| arm | +0 | +10 | +50 | +100 | +140 | +150 | +200 | +250 | +300 |
|---|---|---|---|---|---|---|---|---|---|
| A_retain | 0.41 / 0.00 | 0.41 / 0.00 | 0.41 / 0.00 | 0.41 / 0.00 | 0.41 / 0.00 | 0.39 / 0.00 | 0.18 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| B_targeted | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.39 | 0.00 / 0.18 | 0.00 / 0.00 | 0.00 / 0.00 |
| C_sham | 0.41 / 0.41 | 0.41 / 0.41 | 0.41 / 0.39 | 0.41 / 0.18 | 0.41 / 0.02 | 0.39 / 0.00 | 0.18 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| D_cleannet | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.41 | 0.00 / 0.39 | 0.00 / 0.18 | 0.00 / 0.00 | 0.00 / 0.00 |

Mean replacement share over the AUC window (sampling is uniform, so = expected sampled share): A_retain 0.000, B_targeted 0.119, C_sham 0.056, D_cleannet 0.119.

## Addendum: replacement-data exposure (descriptive; added after the user's note, before viewing results)

Expected sampled replacement share = buffer replacement share (uniform sampling; seam correction <= 128/1M; realised per-update share is binomial around it, SD <= 0.0055).

| checkpoint | +0 | +20 | +40 | +60 | +80 | +100 | +120 | +140 | +160 | +180 | +200 | +220 | +240 | +260 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_targeted | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.34 | 0.26 | 0.18 | 0.10 | 0.02 | 0.00 |
| C_sham | 0.41 | 0.41 | 0.41 | 0.34 | 0.26 | 0.18 | 0.10 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| D_cleannet | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.34 | 0.26 | 0.18 | 0.10 | 0.02 | 0.00 |

Cumulative sampled replacement transitions over +1..+650 (mean over seeds): B_targeted 0.65M, C_sham 0.31M, D_cleannet 0.65M; ratio B/C = 2.07. Exposure identical in B and C up to ~+41.
B - C over the exposure-matched checkpoints +10..+40 (descriptive): +2.5 [+0.2, +4.8]* (9/12 seeds > 0). Later B - C differences combine data identity with the residence mismatch.

TD loss, first 30 post-removal updates (mean): A_retain 0.906, B_targeted 0.870, C_sham 0.741, D_cleannet 0.078

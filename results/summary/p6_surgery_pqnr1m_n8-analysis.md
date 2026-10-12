# PQN-R params x replay-buffer surgery — 20261012-p5-surgery-late-s0-pqnr-1000k-SpaceInvaders-MinAtar-85a0033.json, 20261012-p5-surgery-late-s1-pqnr-1000k-SpaceInvaders-MinAtar-5bf7d2b.json, 20261012-p5-surgery-late-s2-pqnr-1000k-SpaceInvaders-MinAtar-5bf7d2b.json, 20261012-p5-surgery-late-s3-pqnr-1000k-SpaceInvaders-MinAtar-5bf7d2b.json, 20261012-p5-surgery-late-s4-pqnr-1000k-SpaceInvaders-MinAtar-5bf7d2b.json, 20261012-p5-surgery-late-s5-pqnr-1000k-SpaceInvaders-MinAtar-5bf7d2b.json, 20261012-p5-surgery-late-s6-pqnr-1000k-SpaceInvaders-MinAtar-5bf7d2b.json, 20261012-p5-surgery-late-s7-pqnr-1000k-SpaceInvaders-MinAtar-5bf7d2b.json

Seeds (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Variant: pqnr-1000k.

Metric: eval_eps05_return_mean. Removal at update 350; recovery 650 updates; eval every 10. Assignment rule: network = params + optimizer state (together); buffer = all buffer arrays; env state, obs, RNG, update counter, exploration schedule, deficit bounds from C; no target network. Buffer turnover: 244.1 updates; fraction of D's buffer written under the restriction at removal: 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41. Buffer pointer/size identical in C and D: [True, True, True, True, True, True, True, True]. Consistency vs recovery run: [{}, {'C_fork_matches_recovery_control': {'seeds': [1], 'identical': True}}, {'C_fork_matches_recovery_control': {'seeds': [2], 'identical': True}}, {'C_fork_matches_recovery_control': {'seeds': [3], 'identical': True}}, {'C_fork_matches_recovery_control': {'seeds': [4], 'identical': True}}, {'C_fork_matches_recovery_control': {'seeds': [5], 'identical': True}}, {'C_fork_matches_recovery_control': {'seeds': [6], 'identical': True}}, {'C_fork_matches_recovery_control': {'seeds': [7], 'identical': True}}].

Readouts at +h > 0 average checkpoints h-E, h, h+E; end = mean of last 2.

## Arm scores (mean over seeds)

| arm | AUC | +0 | +10 | +20 | +30 | +100 | +650 |
|---|---|---|---|---|---|---|---|
| CC | 37.9 | 27.8 | 27.3 | 27.1 | 26.5 | 29.4 | 48.9 |
| DC | 35.4 | 14.1 | 15.6 | 18.7 | 23.0 | 28.1 | 45.6 |
| CD | 36.2 | 27.8 | 27.4 | 26.3 | 26.7 | 27.3 | 49.0 |
| DD | 32.3 | 14.1 | 18.1 | 19.7 | 22.0 | 27.1 | 46.8 |

## Contrasts (paired; negative = harm)

| contrast | AUC | +0 | +10 | +20 | +30 | +100 | +650 |
|---|---|---|---|---|---|---|---|
| total damage: DD - CC | -5.6 [-7.0, -4.2]* | -13.8 [-19.7, -7.9]* | -9.2 [-14.7, -3.7]* | -7.4 [-11.7, -3.1]* | -4.5 [-7.9, -1.1]* | -2.2 [-4.0, -0.5]* | -2.2 [-10.5, +6.2] |
| parameter transfer: DC - CC | -2.6 [-3.6, -1.6]* | -13.8 [-19.7, -7.9]* | -11.6 [-19.1, -4.2]* | -8.4 [-12.6, -4.2]* | -3.4 [-6.2, -0.6]* | -1.3 [-3.5, +0.9] | -3.4 [-8.9, +2.2] |
| data transfer: CD - CC | -1.7 [-3.3, -0.2]* | +0.0 [+0.0, +0.0] | +0.1 [-2.8, +3.0] | -0.7 [-1.9, +0.5] | +0.2 [-1.5, +1.9] | -2.1 [-4.2, +0.1] | +0.1 [-8.7, +8.9] |
| network main effect | -3.2 [-4.2, -2.2]* | -13.8 [-19.7, -7.9]* | -10.5 [-14.7, -6.2]* | -7.5 [-11.3, -3.8]* | -4.1 [-6.8, -1.3]* | -0.7 [-3.0, +1.5] | -2.8 [-5.3, -0.3]* |
| buffer main effect | -2.4 [-3.4, -1.4]* | +0.0 [+0.0, +0.0]* | +1.3 [-2.8, +5.3] | +0.1 [-2.4, +2.7] | -0.4 [-2.3, +1.4] | -1.5 [-3.3, +0.3] | +0.6 [-5.6, +6.9] |
| interaction (DD-CD)-(DC-CC) | -1.3 [-3.2, +0.6] | +0.0 [+0.0, +0.0] | +2.3 [-5.6, +10.3] | +1.7 [-2.4, +5.8] | -1.2 [-4.4, +1.9] | +1.2 [-0.4, +2.7] | +1.2 [-9.9, +12.3] |
| rescue by clean network: CD - DD | +3.9 [+2.2, +5.5]* | +13.8 [+7.9, +19.7]* | +9.3 [+5.8, +12.8]* | +6.6 [+2.3, +11.0]* | +4.7 [+1.2, +8.2]* | +0.1 [-2.4, +2.7] | +2.2 [-4.3, +8.8] |
| rescue by clean buffer: DC - DD | +3.0 [+1.8, +4.2]* | +0.0 [+0.0, +0.0] | -2.4 [-9.9, +5.1] | -1.0 [-5.5, +3.5] | +1.1 [-1.9, +4.0] | +0.9 [-0.7, +2.6] | -1.2 [-9.2, +6.7] |

Share of total AUC damage: network main effect 57%, buffer main effect 43%.

## Seed robustness (AUC, n = 8)

| contrast | mean | between-seed SD | seeds < 0 | 95% prediction interval (new seed) | per-seed (0, 1, 2, 3, 4, 5, 6, 7) |
|---|---|---|---|---|---|
| total damage: DD - CC | -5.6 | 1.7 | 8/8 | [-9.9, -1.3] | -5.9, -4.6, -4.3, -7.2, -3.9, -5.4, -8.9, -4.6 |
| parameter transfer: DC - CC | -2.6 | 1.2 | 8/8 | [-5.6, +0.4] | -2.4, -0.4, -1.9, -3.0, -2.5, -4.3, -3.8, -2.1 |
| data transfer: CD - CC | -1.7 | 1.9 | 6/8 | [-6.5, +3.0] | +1.0, +0.0, -0.7, -4.5, -1.1, -2.5, -2.6, -3.8 |
| network main effect | -3.2 | 1.2 | 8/8 | [-6.2, -0.2] | -4.7, -2.5, -2.8, -2.9, -2.7, -3.6, -5.1, -1.5 |
| buffer main effect | -2.4 | 1.2 | 8/8 | [-5.5, +0.7] | -1.2, -2.1, -1.5, -4.3, -1.2, -1.8, -3.9, -3.1 |
| interaction (DD-CD)-(DC-CC) | -1.3 | 2.3 | 5/8 | [-7.1, +4.5] | -4.5, -4.2, -1.7, +0.3, -0.3, +1.4, -2.6, +1.3 |

## Overwrite of restriction-era data after removal (mean over tagged seeds [0, 1, 2, 3, 4, 5, 6, 7])

buf = share of valid buffer slots written during the window; samp = share of sampled replay transitions from it. Arms with the D buffer carry restricted data; arms with the C buffer carry the control's same-era data.

| arm | +1 buf | +1 samp | first update buf < 50% | < 10% | = 0 | AUC-window mean samp |
|---|---|---|---|---|---|---|
| CC | 0.41 | 0.41 | 1 | 220 | 245 | 0.122 |
| CD | 0.41 | 0.41 | 1 | 220 | 245 | 0.122 |
| DD | 0.41 | 0.41 | 1 | 220 | 245 | 0.122 |
| DC | 0.41 | 0.41 | 1 | 220 | 245 | 0.122 |

TD loss over the first 30 post-removal updates (mean): CC 0.074, DC 0.690, CD 0.058, DD 0.858

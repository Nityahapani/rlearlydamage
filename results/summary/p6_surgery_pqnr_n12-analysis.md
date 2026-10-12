# PQN-R params x replay-buffer surgery — 20261011-p5-surgery-late-pqnr-SpaceInvaders-MinAtar-77afd57.json, 20261012-p5-surgery-late-s4to7-pqnr-SpaceInvaders-MinAtar-c75175b.json, 20261012-p5-surgery-late-s8to11-pqnr-SpaceInvaders-MinAtar-28909a2.json

Seeds (n = 12): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Variant: pqnr.

Metric: eval_eps05_return_mean. Removal at update 350; recovery 650 updates; eval every 10. Assignment rule: network = params + optimizer state (together); buffer = all buffer arrays; env state, obs, RNG, update counter, exploration schedule, deficit bounds from C; no target network. Buffer turnover: 24.4 updates; fraction of D's buffer written under the restriction at removal: 1.00, 1.00, 1.00. Buffer pointer/size identical in C and D: [True, True, True]. Consistency vs recovery run: [{'C_fork_matches_recovery_control': True, 'D_fork_matches_recovery_late': True}, {}, {}].

Readouts at +h > 0 average checkpoints h-E, h, h+E; end = mean of last 2.

## Arm scores (mean over seeds)

| arm | AUC | +0 | +10 | +20 | +30 | +100 | +650 |
|---|---|---|---|---|---|---|---|
| CC | 48.9 | 32.0 | 36.3 | 37.5 | 38.2 | 40.2 | 58.8 |
| DC | 44.3 | 4.2 | 18.9 | 21.9 | 26.7 | 31.0 | 53.4 |
| CD | 48.9 | 32.0 | 31.9 | 33.6 | 36.0 | 42.9 | 58.5 |
| DD | 42.2 | 4.2 | 16.0 | 18.0 | 22.3 | 28.3 | 55.1 |

## Contrasts (paired; negative = harm)

| contrast | AUC | +0 | +10 | +20 | +30 | +100 | +650 |
|---|---|---|---|---|---|---|---|
| total damage: DD - CC | -6.7 [-8.0, -5.3]* | -27.8 [-32.9, -22.8]* | -20.3 [-23.7, -17.0]* | -19.5 [-22.3, -16.7]* | -15.9 [-18.1, -13.7]* | -11.9 [-15.2, -8.6]* | -3.6 [-7.2, -0.1]* |
| parameter transfer: DC - CC | -4.6 [-6.1, -3.0]* | -27.8 [-32.9, -22.8]* | -17.4 [-21.7, -13.2]* | -15.5 [-18.7, -12.4]* | -11.5 [-13.8, -9.2]* | -9.2 [-13.0, -5.3]* | -5.4 [-11.8, +1.0] |
| data transfer: CD - CC | +0.0 [-1.0, +1.0] | +0.0 [+0.0, +0.0] | -4.5 [-9.1, +0.1] | -3.9 [-7.5, -0.3]* | -2.2 [-4.8, +0.3] | +2.7 [-1.1, +6.5] | -0.3 [-6.8, +6.3] |
| network main effect | -5.6 [-6.3, -4.9]* | -27.8 [-32.9, -22.8]* | -16.6 [-19.9, -13.4]* | -15.6 [-17.8, -13.4]* | -12.6 [-14.2, -11.0]* | -11.9 [-14.7, -9.1]* | -4.4 [-7.3, -1.4]* |
| buffer main effect | -1.0 [-2.0, -0.1]* | -0.0 [-0.0, -0.0]* | -3.7 [-7.0, -0.4]* | -3.9 [-6.5, -1.4]* | -3.3 [-5.1, -1.5]* | -0.0 [-2.3, +2.3] | +0.7 [-2.9, +4.3] |
| interaction (DD-CD)-(DC-CC) | -2.1 [-4.5, +0.2] | +0.0 [+0.0, +0.0] | +1.6 [-4.0, +7.2] | -0.1 [-4.2, +4.0] | -2.2 [-5.1, +0.8] | -5.4 [-9.9, -1.0]* | +2.0 [-7.9, +11.9] |
| rescue by clean network: CD - DD | +6.7 [+5.5, +7.8]* | +27.8 [+22.8, +32.9]* | +15.8 [+11.6, +20.1]* | +15.6 [+12.7, +18.6]* | +13.7 [+11.7, +15.7]* | +14.6 [+11.3, +17.9]* | +3.4 [-1.7, +8.5] |
| rescue by clean buffer: DC - DD | +2.1 [+0.2, +4.0]* | +0.0 [+0.0, +0.0] | +2.9 [-1.1, +6.9] | +4.0 [+1.1, +6.9]* | +4.4 [+2.3, +6.5]* | +2.8 [+0.2, +5.3]* | -1.7 [-7.4, +3.9] |

Share of total AUC damage: network main effect 84%, buffer main effect 16%.

## Seed robustness (AUC, n = 12)

| contrast | mean | between-seed SD | seeds < 0 | 95% prediction interval (new seed) | per-seed (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) |
|---|---|---|---|---|---|
| total damage: DD - CC | -6.7 | 2.1 | 12/12 | [-11.4, -1.9] | -5.0, -8.5, -6.9, -7.7, -5.4, -5.6, -9.6, -9.2, -7.3, -6.8, -5.7, -2.1 |
| parameter transfer: DC - CC | -4.6 | 2.5 | 12/12 | [-10.2, +1.1] | -5.0, -2.2, -4.6, -8.2, -2.9, -2.2, -2.9, -6.7, -6.5, -8.1, -0.7, -4.6 |
| data transfer: CD - CC | +0.0 | 1.6 | 6/12 | [-3.7, +3.8] | +1.1, +0.4, -1.2, -0.7, +2.2, +1.5, -1.0, -1.9, -1.5, -2.1, +2.6, +0.7 |
| network main effect | -5.6 | 1.1 | 12/12 | [-8.1, -3.1] | -5.5, -5.6, -5.2, -7.6, -5.3, -4.7, -5.7, -7.0, -6.2, -6.4, -4.5, -3.7 |
| buffer main effect | -1.0 | 1.5 | 10/12 | [-4.5, +2.4] | +0.5, -2.9, -1.7, -0.1, -0.2, -0.9, -3.9, -2.2, -1.1, -0.4, -1.2, +1.6 |
| interaction (DD-CD)-(DC-CC) | -2.1 | 3.7 | 8/12 | [-10.6, +6.3] | -1.2, -6.7, -1.2, +1.2, -4.7, -4.9, -5.7, -0.7, +0.8, +3.5, -7.7, +1.8 |

## Overwrite of restriction-era data after removal (mean over tagged seeds [4, 5, 6, 7, 8, 9, 10, 11])

buf = share of valid buffer slots written during the window; samp = share of sampled replay transitions from it. Arms with the D buffer carry restricted data; arms with the C buffer carry the control's same-era data.

| arm | +1 buf | +1 samp | first update buf < 50% | < 10% | = 0 | AUC-window mean samp |
|---|---|---|---|---|---|---|
| CC | 0.96 | 0.96 | 13 | 22 | 25 | 0.018 |
| CD | 0.96 | 0.96 | 13 | 22 | 25 | 0.018 |
| DD | 0.96 | 0.96 | 13 | 22 | 25 | 0.018 |
| DC | 0.96 | 0.96 | 13 | 22 | 25 | 0.018 |

TD loss over the first 30 post-removal updates (mean): CC 0.152, DC 0.518, CD 0.129, DD 0.176

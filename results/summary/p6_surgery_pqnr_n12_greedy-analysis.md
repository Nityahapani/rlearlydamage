# PQN-R params x replay-buffer surgery — 20261011-p5-surgery-late-pqnr-SpaceInvaders-MinAtar-77afd57.json, 20261012-p5-surgery-late-s4to7-pqnr-SpaceInvaders-MinAtar-c75175b.json, 20261012-p5-surgery-late-s8to11-pqnr-SpaceInvaders-MinAtar-28909a2.json

Seeds (n = 12): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Variant: pqnr.

Metric: eval_return_mean. Removal at update 350; recovery 650 updates; eval every 10. Assignment rule: network = params + optimizer state (together); buffer = all buffer arrays; env state, obs, RNG, update counter, exploration schedule, deficit bounds from C; no target network. Buffer turnover: 24.4 updates; fraction of D's buffer written under the restriction at removal: 1.00, 1.00, 1.00. Buffer pointer/size identical in C and D: [True, True, True]. Consistency vs recovery run: [{'C_fork_matches_recovery_control': True, 'D_fork_matches_recovery_late': True}, {}, {}].

Readouts at +h > 0 average checkpoints h-E, h, h+E; end = mean of last 2.

## Arm scores (mean over seeds)

| arm | AUC | +0 | +10 | +20 | +30 | +100 | +650 |
|---|---|---|---|---|---|---|---|
| CC | 86.6 | 52.8 | 63.2 | 57.8 | 52.9 | 69.6 | 110.8 |
| DC | 75.6 | 4.0 | 20.1 | 25.4 | 32.4 | 40.9 | 95.0 |
| CD | 83.4 | 52.8 | 44.5 | 44.6 | 48.7 | 67.3 | 103.3 |
| DD | 71.6 | 4.0 | 17.8 | 21.4 | 25.8 | 32.6 | 102.6 |

## Contrasts (paired; negative = harm)

| contrast | AUC | +0 | +10 | +20 | +30 | +100 | +650 |
|---|---|---|---|---|---|---|---|
| total damage: DD - CC | -15.0 [-19.7, -10.2]* | -48.8 [-69.6, -28.1]* | -45.4 [-62.2, -28.6]* | -36.4 [-51.1, -21.6]* | -27.1 [-38.3, -15.9]* | -37.0 [-57.8, -16.2]* | -8.2 [-34.0, +17.6] |
| parameter transfer: DC - CC | -11.0 [-15.9, -6.0]* | -48.8 [-69.6, -28.1]* | -43.1 [-60.0, -26.3]* | -32.4 [-45.9, -18.9]* | -20.4 [-31.1, -9.8]* | -28.7 [-51.6, -5.7]* | -15.9 [-46.3, +14.5] |
| data transfer: CD - CC | -3.2 [-8.7, +2.4] | +0.0 [+0.0, +0.0] | -18.8 [-41.9, +4.3] | -13.2 [-34.8, +8.5] | -4.2 [-21.6, +13.2] | -2.3 [-29.7, +25.1] | -7.5 [-40.4, +25.3] |
| network main effect | -11.4 [-15.7, -7.0]* | -48.8 [-69.6, -28.1]* | -34.9 [-44.7, -25.0]* | -27.8 [-36.5, -19.0]* | -21.7 [-29.2, -14.1]* | -31.7 [-44.5, -18.9]* | -8.3 [-27.1, +10.5] |
| buffer main effect | -3.6 [-7.8, +0.7] | +0.0 [+0.0, +0.0] | -10.5 [-22.5, +1.4] | -8.6 [-20.1, +3.0] | -5.4 [-15.2, +4.3] | -5.3 [-20.5, +9.8] | +0.1 [-17.3, +17.4] |
| interaction (DD-CD)-(DC-CC) | -0.8 [-7.4, +5.7] | +0.0 [+0.0, +0.0] | +16.5 [-6.4, +39.4] | +9.2 [-12.0, +30.5] | -2.5 [-20.3, +15.3] | -6.0 [-33.5, +21.4] | +15.2 [-27.1, +57.5] |
| rescue by clean network: CD - DD | +11.8 [+5.9, +17.6]* | +48.8 [+28.1, +69.6]* | +26.6 [+13.5, +39.8]* | +23.2 [+9.1, +37.2]* | +22.9 [+10.3, +35.6]* | +34.7 [+21.3, +48.1]* | +0.7 [-25.4, +26.7] |
| rescue by clean buffer: DC - DD | +4.0 [-1.1, +9.1] | +0.0 [+0.0, +0.0] | +2.3 [-2.0, +6.6] | +4.0 [-0.9, +8.9] | +6.7 [-0.0, +13.4] | +8.3 [-0.9, +17.6] | -7.7 [-28.0, +12.7] |

Share of total AUC damage: network main effect 76%, buffer main effect 24%.

## Seed robustness (AUC, n = 12)

| contrast | mean | between-seed SD | seeds < 0 | 95% prediction interval (new seed) | per-seed (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) |
|---|---|---|---|---|---|
| total damage: DD - CC | -15.0 | 7.4 | 12/12 | [-32.0, +2.1] | -14.3, -18.6, -17.3, -23.6, -3.6, -20.5, -23.8, -15.8, -13.0, -20.1, -7.9, -0.9 |
| parameter transfer: DC - CC | -11.0 | 7.8 | 12/12 | [-28.9, +6.9] | -10.3, -2.8, -8.9, -30.0, -12.2, -10.3, -13.1, -16.9, -1.2, -15.4, -2.7, -7.6 |
| data transfer: CD - CC | -3.2 | 8.8 | 8/12 | [-23.3, +16.9] | +3.0, -18.3, -5.4, -13.1, +9.1, -5.2, -5.4, +4.2, -4.2, -0.1, +9.6, -12.3 |
| network main effect | -11.4 | 6.8 | 11/12 | [-27.0, +4.3] | -13.8, -1.5, -10.4, -20.2, -12.5, -12.8, -15.7, -18.5, -5.0, -17.7, -10.1, +1.9 |
| buffer main effect | -3.6 | 6.7 | 9/12 | [-18.9, +11.7] | -0.5, -17.1, -6.9, -3.4, +8.8, -7.7, -8.0, +2.7, -8.0, -2.4, +2.2, -2.8 |
| interaction (DD-CD)-(DC-CC) | -0.8 | 10.3 | 9/12 | [-24.4, +22.7] | -7.0, +2.5, -3.1, +19.6, -0.5, -5.0, -5.2, -3.1, -7.7, -4.6, -14.8, +19.0 |

## Overwrite of restriction-era data after removal (mean over tagged seeds [4, 5, 6, 7, 8, 9, 10, 11])

buf = share of valid buffer slots written during the window; samp = share of sampled replay transitions from it. Arms with the D buffer carry restricted data; arms with the C buffer carry the control's same-era data.

| arm | +1 buf | +1 samp | first update buf < 50% | < 10% | = 0 | AUC-window mean samp |
|---|---|---|---|---|---|---|
| CC | 0.96 | 0.96 | 13 | 22 | 25 | 0.018 |
| CD | 0.96 | 0.96 | 13 | 22 | 25 | 0.018 |
| DD | 0.96 | 0.96 | 13 | 22 | 25 | 0.018 |
| DC | 0.96 | 0.96 | 13 | 22 | 25 | 0.018 |

TD loss over the first 30 post-removal updates (mean): CC 0.152, DC 0.518, CD 0.129, DD 0.176

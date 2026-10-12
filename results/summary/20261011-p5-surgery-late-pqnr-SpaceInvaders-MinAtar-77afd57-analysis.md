# PQN-R params x replay-buffer surgery — 20261011-p5-surgery-late-pqnr-SpaceInvaders-MinAtar-77afd57.json

Seeds (n = 4): [0, 1, 2, 3]. Variant: pqnr.

Metric: eval_eps05_return_mean. Removal at update 350; recovery 650 updates; eval every 10. Assignment rule: network = params + optimizer state (together); buffer = all buffer arrays; env state, obs, RNG, update counter, exploration schedule, deficit bounds from C; no target network. Buffer turnover: 24.4 updates; fraction of D's buffer written under the restriction at removal: 1.00. Buffer pointer/size identical in C and D: [True]. Consistency vs recovery run: [{'C_fork_matches_recovery_control': True, 'D_fork_matches_recovery_late': True}].

Readouts at +h > 0 average checkpoints h-E, h, h+E; end = mean of last 2.

## Arm scores (mean over seeds)

| arm | AUC | +0 | +10 | +20 | +30 | +100 | +650 |
|---|---|---|---|---|---|---|---|
| CC | 49.0 | 26.7 | 37.3 | 38.9 | 39.9 | 40.0 | 60.3 |
| DC | 44.0 | 3.2 | 16.8 | 20.6 | 26.9 | 29.5 | 49.3 |
| CD | 48.9 | 26.7 | 31.9 | 34.1 | 35.8 | 46.9 | 56.6 |
| DD | 42.0 | 3.2 | 16.5 | 18.4 | 23.3 | 26.7 | 55.1 |

## Contrasts (paired; negative = harm)

| contrast | AUC | +0 | +10 | +20 | +30 | +100 | +650 |
|---|---|---|---|---|---|---|---|
| total damage: DD - CC | -7.0 [-9.4, -4.7]* | -23.4 [-32.5, -14.4]* | -20.8 [-29.7, -11.9]* | -20.5 [-29.8, -11.2]* | -16.6 [-24.5, -8.7]* | -13.3 [-18.8, -7.9]* | -5.2 [-11.6, +1.3] |
| parameter transfer: DC - CC | -5.0 [-8.9, -1.1]* | -23.4 [-32.5, -14.4]* | -20.5 [-37.0, -4.0]* | -18.3 [-30.5, -6.1]* | -13.0 [-17.5, -8.6]* | -10.6 [-20.7, -0.4]* | -11.0 [-26.9, +4.8] |
| data transfer: CD - CC | -0.1 [-1.7, +1.6] | +0.0 [+0.0, +0.0] | -5.4 [-16.6, +5.9] | -4.8 [-17.4, +7.8] | -4.1 [-13.3, +5.0] | +6.9 [-4.6, +18.3] | -3.7 [-18.0, +10.5] |
| network main effect | -6.0 [-7.7, -4.2]* | -23.4 [-32.5, -14.4]* | -18.0 [-29.4, -6.5]* | -17.0 [-25.2, -8.8]* | -12.7 [-17.6, -7.9]* | -15.4 [-19.7, -11.1]* | -6.2 [-15.8, +3.3] |
| buffer main effect | -1.1 [-3.6, +1.4] | +0.0 [+0.0, +0.0]* | -2.9 [-8.8, +3.1] | -3.5 [-10.7, +3.7] | -3.9 [-11.1, +3.4] | +2.0 [-3.8, +7.9] | +1.0 [-7.1, +9.2] |
| interaction (DD-CD)-(DC-CC) | -1.9 [-7.3, +3.4] | +0.0 [+0.0, +0.0] | +5.1 [-16.8, +27.0] | +2.6 [-13.8, +19.0] | +0.6 [-6.7, +7.9] | -9.6 [-24.3, +5.0] | +9.6 [-9.2, +28.4] |
| rescue by clean network: CD - DD | +6.9 [+4.7, +9.2]* | +23.4 [+14.4, +32.5]* | +15.4 [+0.3, +30.5]* | +15.7 [+4.7, +26.8]* | +12.4 [+5.1, +19.8]* | +20.2 [+13.8, +26.6]* | +1.4 [-9.0, +11.8] |
| rescue by clean buffer: DC - DD | +2.0 [-2.9, +6.9] | +0.0 [+0.0, +0.0] | +0.3 [-13.2, +13.9] | +2.2 [-6.7, +11.1] | +3.6 [-3.4, +10.6] | +2.8 [-3.9, +9.5] | -5.8 [-16.2, +4.5] |

Share of total AUC damage: network main effect 85%, buffer main effect 15%.

## Seed robustness (AUC, n = 4)

| contrast | mean | between-seed SD | seeds < 0 | 95% prediction interval (new seed) | per-seed (0, 1, 2, 3) |
|---|---|---|---|---|---|
| total damage: DD - CC | -7.0 | 1.5 | 4/4 | [-12.3, -1.7] | -5.0, -8.5, -6.9, -7.7 |
| parameter transfer: DC - CC | -5.0 | 2.5 | 4/4 | [-13.8, +3.8] | -5.0, -2.2, -4.6, -8.2 |
| data transfer: CD - CC | -0.1 | 1.0 | 2/4 | [-3.8, +3.6] | +1.1, +0.4, -1.2, -0.7 |
| network main effect | -6.0 | 1.1 | 4/4 | [-9.9, -2.1] | -5.5, -5.6, -5.2, -7.6 |
| buffer main effect | -1.1 | 1.6 | 3/4 | [-6.6, +4.5] | +0.5, -2.9, -1.7, -0.1 |
| interaction (DD-CD)-(DC-CC) | -1.9 | 3.4 | 3/4 | [-13.9, +10.0] | -1.2, -6.7, -1.2, +1.2 |

TD loss over the first 30 post-removal updates (mean): CC 0.156, DC 0.541, CD 0.132, DD 0.226

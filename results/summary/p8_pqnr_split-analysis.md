# Phase 8 J4 — PQN-R parameters vs optimizer state (100k buffer, damaged buffer held fixed)

New: 20261013-p5-p8-split-late-pqnr-SpaceInvaders-MinAtar-3f952c0.json, 20261013-p5-p8-split-late-pqnr-SpaceInvaders-MinAtar-bfca538.json
Old (CD = CCD, DD, CC): 20261011-p5-surgery-late-pqnr-SpaceInvaders-MinAtar-77afd57.json, 20261012-p5-surgery-late-s4to7-pqnr-SpaceInvaders-MinAtar-c75175b.json
Seeds (n = 8): [0, 1, 2, 3, 4, 5, 6, 7]. Metric eval_eps05_return_mean. Removal u350, recovery 650.
Bit-identity check Phase 8 DDD == earlier DD: {0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True} -> PASS

## Cell means

| cell (params, opt, buffer) | AUC | early +10..+150 | late +250..+650 | +10 |
|---|---|---|---|---|
| CCC | 49.3 | 41.1 | 52.8 | 38.1 |
| CCD | 49.4 | 40.1 | 53.6 | 29.9 |
| CDD | 48.5 | 36.3 | 53.6 | 11.1 |
| DCD | 43.4 | 26.8 | 50.3 | 15.4 |
| DDD | 42.1 | 26.5 | 48.8 | 13.7 |

## Contrasts (paired; positive = rescue)

| contrast | AUC | early +10..+150 | late +250..+650 | +10 | AUC p | Holm p |
|---|---|---|---|---|---|---|
| parameter rescue CDD - DDD (primary) | +6.4 [+4.8, +8.0]* | +9.9 [+7.9, +11.9]* | +4.8 [+2.8, +6.7]* | -2.6 [-6.5, +1.3] | 0.0000 | 0.0001 |
| optimizer rescue DCD - DDD (primary) | +1.3 [-0.4, +3.1] | +0.4 [-1.9, +2.7] | +1.5 [-0.3, +3.3] | +1.6 [-2.9, +6.2] | 0.1156 | 0.1156 |
| network rescue CCD - DDD (= earlier CD - DD) | +7.3 [+6.4, +8.2]* | +13.7 [+12.2, +15.2]* | +4.8 [+2.8, +6.8]* | +16.2 [+8.7, +23.6]* | 0.0000 |  |
| params main effect (2x2 at D buffer) | +6.2 [+5.1, +7.3]* | +11.6 [+10.4, +12.8]* | +4.1 [+2.4, +5.7]* | +5.9 [+1.7, +10.2]* | 0.0000 |  |
| opt main effect (2x2 at D buffer) | +1.1 [+0.4, +1.9]* | +2.1 [+0.9, +3.2]* | +0.8 [-0.1, +1.6] | +10.2 [+4.9, +15.5]* | 0.0089 |  |
| interaction (CCD - CDD) - (DCD - DDD) | -0.4 [-3.1, +2.2] | +3.4 [+0.6, +6.3]* | -1.4 [-4.6, +1.8] | +17.1 [+9.1, +25.2]* | 0.7152 |  |

Share of the network rescue (CCD - DDD, AUC +7.31) recovered by clean parameters alone: 87%; by clean optimizer state alone: 18%.

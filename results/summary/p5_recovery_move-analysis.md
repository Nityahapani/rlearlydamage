# Phase 5 — PQN / PQN-R recovery replication (SpaceInvaders-MinAtar, movement restriction {noop, fire})

Deficit-free greedy eval (full action set, clean obs, true reward, common random numbers); paired t-CIs over seeds; * = CI excludes 0. gap = control − arm (positive = damage); delay-adjusted = control(u − L) − arm(u).

## PQN  (manifests: 20261011-p5-recovery-pqn-control-SpaceInvaders-MinAtar-3bc24c1.json, 20261011-p5-recovery-pqn-early-late-SpaceInvaders-MinAtar-fd9d190.json)

Exposure per update: 4096 env steps, 64 grad steps of 128, replay ratio 2.0, buffer 0 transitions (turnover 0.0 updates); linear 1.0 -> 0.05 over 1000000 env steps (244 updates), then constant; LR 0.0005 constant (RAdam, grad clip 10.0).

Control eval reliability after the eps floor (mean |checkpoint-to-checkpoint change| / mean level; lower = better): eval_return_mean 0.379, eval_eps05_return_mean 0.156. **Primary metric: eval_eps05_return_mean.** Readouts at +h > 0 average the checkpoints h-E, h, h+E (where available).

Control eval: u0 1.6, u100 23.0, u250 46.8, u500 70.0, u750 79.5, u1000 76.7; final (last 50 updates) 81.3 ± 6.4 sd. Eval gain per 100-update block: 0-100: +21.4; 100-200: +0.3; 200-300: +35.1; 300-400: +6.5; 400-500: +5.0; 500-600: +2.7; 600-700: +9.3; 700-800: -1.1; 800-900: -5.0; 900-1000: +0.8.

| arm | window (updates) | horizon | gap | delay-adjusted |
|---|---|---|---|---|
| early | [0, 100) | +0 (u100) | +19.0 [+16.9, +21.1]* | -2.4 [-5.9, +1.0] |
| early | [0, 100) | +10 (u110) | +17.4 [+15.0, +19.8]* | -1.8 [-4.5, +0.9] |
| early | [0, 100) | +100 (u200) | +2.5 [-2.7, +7.7] | -5.6 [-12.6, +1.4] |
| early | [0, 100) | +500 (u600) | +0.6 [-16.9, +18.1] | -9.9 [-31.7, +11.9] |
| early | [0, 100) | +900 (u1000) | +0.8 [-5.1, +6.6] | -2.1 [-24.9, +20.6] |
| early | [0, 100) | AUC +10..+900 | +0.5 [-3.5, +4.5] | -6.4 [-10.5, -2.4]* |
| early | | at onset: control 1.6; at removal arm 4.0 vs control 23.0 | | |
| late | [250, 350) | +0 (u350) | +58.3 [+45.9, +70.6]* | +43.2 [+19.2, +67.3]* |
| late | [250, 350) | +10 (u360) | +25.0 [+19.8, +30.2]* | +12.8 [-0.6, +26.2] |
| late | [250, 350) | +100 (u450) | +5.7 [-3.9, +15.3] | -1.1 [-10.4, +8.2] |
| late | [250, 350) | +500 (u850) | +7.8 [-1.6, +17.2] | -2.0 [-16.3, +12.2] |
| late | [250, 350) | +650 (u1000) | +6.5 [-5.0, +18.0] | +3.6 [-12.2, +19.4] |
| late | [250, 350) | AUC +10..+650 | +4.6 [+2.0, +7.2]* | +1.4 [-0.7, +3.5] |
| late | | at onset: control 46.8; at removal arm 3.6 vs control 61.8 | | |

## PQNR  (manifests: 20261011-p5-recovery-pqnr-control-SpaceInvaders-MinAtar-3bc24c1.json, 20261011-p5-recovery-pqnr-early-late-SpaceInvaders-MinAtar-8456a9b.json)

Exposure per update: 4096 env steps, 64 grad steps of 128, replay ratio 2.0, buffer 100000 transitions (turnover 24.4 updates); linear 1.0 -> 0.05 over 1000000 env steps (244 updates), then constant; LR 0.0005 constant (RAdam, grad clip 10.0).

Control eval reliability after the eps floor (mean |checkpoint-to-checkpoint change| / mean level; lower = better): eval_return_mean 0.508, eval_eps05_return_mean 0.176. **Primary metric: eval_eps05_return_mean.** Readouts at +h > 0 average the checkpoints h-E, h, h+E (where available).

Control eval: u0 1.6, u100 11.8, u250 26.6, u500 44.8, u750 48.4, u1000 59.3; final (last 50 updates) 58.3 ± 4.4 sd. Eval gain per 100-update block: 0-100: +10.3; 100-200: +14.6; 200-300: +4.3; 300-400: +8.6; 400-500: +5.4; 500-600: +8.6; 600-700: -2.0; 700-800: +5.5; 800-900: +1.1; 900-1000: +1.3.

| arm | window (updates) | horizon | gap | delay-adjusted |
|---|---|---|---|---|
| early | [0, 100) | +0 (u100) | +7.4 [-1.5, +16.4] | -2.8 [-7.2, +1.6] |
| early | [0, 100) | +10 (u110) | +12.2 [+2.0, +22.3]* | -1.0 [-3.3, +1.3] |
| early | [0, 100) | +100 (u200) | +9.2 [+7.5, +10.9]* | -2.2 [-6.9, +2.4] |
| early | [0, 100) | +500 (u600) | -1.3 [-5.9, +3.2] | -7.1 [-20.4, +6.1] |
| early | [0, 100) | +900 (u1000) | +2.0 [-9.4, +13.4] | -0.9 [-11.7, +9.8] |
| early | [0, 100) | AUC +10..+900 | +2.7 [+0.1, +5.2]* | -2.7 [-5.5, +0.2] |
| early | | at onset: control 1.6; at removal arm 4.4 vs control 11.8 | | |
| late | [250, 350) | +0 (u350) | +23.4 [+14.4, +32.5]* | +23.4 [+20.0, +26.8]* |
| late | [250, 350) | +10 (u360) | +26.4 [+20.0, +32.8]* | +15.3 [+12.6, +18.0]* |
| late | [250, 350) | +100 (u450) | +14.4 [+6.3, +22.5]* | +5.7 [-0.5, +12.0] |
| late | [250, 350) | +500 (u850) | +7.7 [-2.8, +18.2] | +0.5 [-11.5, +12.6] |
| late | [250, 350) | +650 (u1000) | +3.2 [-12.0, +18.4] | +0.2 [-7.3, +7.7] |
| late | [250, 350) | AUC +10..+650 | +8.6 [+5.6, +11.6]* | +4.3 [+1.6, +7.0]* |
| late | | at onset: control 26.6; at removal arm 3.2 vs control 26.7 | | |


# POST-HOC: dependence of recovery conclusions on the counterfactual baseline (Phase 8, reviewer response)

Not pre-registered. Negative = restricted agent worse than the baseline; paired 95% t-CI over seeds; * = CI excludes 0.
B1 same-update C(w1+k) (all lost time counted as damage); B2 pause C(w1+k-L) (paper's delay-adjusted baseline); B3 competence-matched C(t*+k), t* = first checkpoint where C's 3-checkpoint moving average reaches D's competence at removal (asks whether D learns as fast as a clean agent from equal competence).

| agent | arm | n | window | K | t* (mean, range) | B1 AUC | B2 AUC | B3 AUC | B1 end | B2 end | B3 end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PPO | move_early | 4 | [0, 50) | 250 | 8 (0-10) | -0.1 [-15.6, +15.4] | +15.0 [-0.2, +30.1] | +12.5 [-5.1, +30.1] | +7.4 [-9.7, +24.5] | +22.0 [+3.9, +40.1]* | +19.2 [-1.2, +39.7] |
| PPO | move_late | 4 | [100, 150) | 250 | 35 (0-70) | -10.1 [-13.0, -7.2]* | +1.1 [-2.9, +5.2] | +18.0 [+3.8, +32.2]* | -8.5 [-13.5, -3.5]* | +0.2 [-5.8, +6.1] | +11.9 [-5.2, +29.1] |
| PPO | fire_early | 4 | [0, 50) | 250 | 0 (0-0) | -9.8 [-22.4, +2.7] | +5.3 [-6.5, +17.1] | +5.3 [-6.5, +17.1] | -4.5 [-17.4, +8.5] | +10.1 [-8.2, +28.4] | +10.1 [-8.2, +28.4] |
| PPO | fire_late | 4 | [100, 150) | 250 | 30 (20-50) | -15.2 [-24.8, -5.6]* | -4.0 [-16.4, +8.4] | +14.4 [+7.0, +21.8]* | -10.9 [-18.8, -3.0]* | -2.3 [-15.6, +11.0] | +8.4 [+1.3, +15.5]* |
| PQN | early | 4 | [0, 100) | 900 | 15 (0-30) | -0.5 [-4.5, +3.5] | +6.4 [+2.4, +10.5]* | +5.4 [+1.8, +8.9]* | +0.6 [-15.2, +16.4] | +4.1 [-6.6, +14.7] | +8.5 [+1.0, +16.1]* |
| PQN | late | 4 | [250, 350) | 650 | 12 (0-40) | -4.6 [-7.2, -2.0]* | -1.4 [-3.5, +0.7] | +17.5 [+16.6, +18.3]* | -8.7 [-24.4, +7.0] | -5.2 [-17.8, +7.3] | +1.5 [-4.0, +7.0] |
| PQN-R | early | 4 | [0, 100) | 900 | 40 (30-50) | -2.7 [-5.2, -0.1]* | +2.7 [-0.2, +5.5] | +0.4 [-2.3, +3.1] | -0.3 [-10.8, +10.3] | +4.6 [-1.1, +10.3] | +3.7 [-5.4, +12.8] |
| PQN-R | late | 4 | [250, 350) | 650 | 25 (0-40) | -8.6 [-11.6, -5.6]* | -4.3 [-7.0, -1.6]* | +7.2 [+3.5, +11.0]* | -2.3 [-11.5, +7.0] | +2.6 [-3.6, +8.7] | +5.7 [-2.5, +13.9] |

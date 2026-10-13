# POST-HOC: minimum detectable effects of small-n null results

MDE = (t_.975 + t_.80) * SE with SE from the 95% CI half-width; 'harm excluded beyond' = CI lower bound.

| contrast | n | estimate [95% CI] | SE | MDE (80% power) | harm excluded beyond | source |
|---|---|---|---|---|---|---|
| PPO move late, delay-adj. at +250 | 4 | +2.0 [-0.1, +4.1] | 0.66 | 2.7 | -0.1 | p2_recovery_all-analysis.md |
| PPO fire late, delay-adj. at +250 | 4 | +2.4 [-3.6, +8.3] | 1.87 | 7.8 | -3.6 | p2_recovery_all-analysis.md |
| PQN late, delay-adj. AUC | 4 | -1.4 [-3.5, +0.7] | 0.66 | 2.7 | -3.5 | p5_recovery_move-analysis.md |
| PQN-R late, delay-adj. at +650 | 4 | -0.2 [-7.7, +7.3] | 2.36 | 9.8 | -7.7 | p5_recovery_move-analysis.md |
| PPO drift suppression F - D, AUC | 4 | +1.1 [-3.4, +5.7] | 1.43 | 5.9 | -3.4 | 20261011-p4-phase4-move-late-...-analysis.md |
| PPO drift suppression A - D, AUC | 4 | -0.0 [-1.8, +1.7] | 0.55 | 2.3 | -1.8 | 20261011-p4-phase4-move-late-...-analysis.md |
| PPO critic main effect, AUC (Phase 3) | 4 | -2.3 [-6.2, +1.7] | 1.24 | 5.2 | -6.2 | 20261011-p3-surgery-move-late-analysis.md |
| PPO optimizer main effect, AUC (Phase 3) | 4 | -0.1 [-2.8, +2.6] | 0.85 | 3.5 | -2.8 | 20261011-p3-surgery-move-late-analysis.md |
| PONR pilot L_300 | 4 | -1.4 [-8.2, +5.5] | 2.15 | 9.0 | -8.2 | p7_ponr_pilot-analysis.md |

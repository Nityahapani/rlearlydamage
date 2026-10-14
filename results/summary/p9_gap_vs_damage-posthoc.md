# POST-HOC: competence lost at removal vs. damage across PPO surgery settings (descriptive)

Gap = DDD − CCC at removal; relative gap = gap / C(w1); clean progress = C(w1) − C(w0) during the window; total = DDD − CCC recovery AUC; delay-adjusted = DDD(w1+k) − C(w1+k−L) AUC. Negative = harm.

| setting | C(w0) | C(w1) | D(w1) | clean progress (rel.) | gap at removal | relative gap | total damage | total / C(w1) | delay-adjusted AUC |
|---|---|---|---|---|---|---|---|---|---|
| SpaceInvaders, movement, 2x128 (n=12) (C, D from Phase 2 seeds 0–3) | 45.40 | 58.70 | 18.50 | +13.30 (+29%) | -37.0 [-45.1, -28.9] | -63% | -15.1 [-17.2, -13.0] | -26% | – |
| SpaceInvaders, movement, 2x256 (n=8) | 46.94 | 68.19 | 18.25 | +21.25 (+45%) | -49.9 [-56.9, -43.0] | -73% | -19.7 [-23.7, -15.6] | -29% | -5.1 [-8.3, -1.8] |
| SpaceInvaders, reward flip (n=8) | 45.59 | 58.22 | 0.00 | +12.63 (+28%) | -58.2 [-66.4, -50.1] | -100% | -32.3 [-40.7, -23.9] | -55% | -18.6 [-27.2, -9.9] |
| Asterix, movement (n=8) | 5.36 | 5.41 | 4.38 | +0.05 (+1%) | -1.0 [-1.5, -0.5] | -18% | -0.1 [-0.5, +0.4] | -2% | +0.5 [+0.0, +1.0] |
| Asterix, reward flip (n=4) | 6.01 | 5.98 | 1.76 | -0.03 (-0%) | -4.2 [-6.0, -2.4] | -70% | -2.4 [-3.6, -1.2] | -40% | -1.8 [-2.6, -1.0] |

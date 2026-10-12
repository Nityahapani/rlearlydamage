# Onset rule — p6_pqnr1m_control

Manifests: 20261012-p5-recovery-pqnr-1000k-control-s01-SpaceInvaders-MinAtar-c79dad6.json, 20261012-p5-recovery-pqnr-1000k-control-s23-SpaceInvaders-MinAtar-8fec61b.json, 20261012-p5-recovery-pqnr-1000k-control-s45-SpaceInvaders-MinAtar-8fec61b.json, 20261012-p5-recovery-pqnr-1000k-control-s67-SpaceInvaders-MinAtar-8fec61b.json. Seeds (n = 8): [0, 1, 2, 3, 4, 5, 6, 7].

Control eval reliability (mean |checkpoint diff| / level, after u244): eval_return_mean 0.597, eval_eps05_return_mean 0.178. **Metric: eval_eps05_return_mean.**

Final (mean of checkpoints in the last 50 updates): 48.3 (per-seed sd 3.9). Threshold 40% of final = 19.3.

- Exploration floor: u244
- Mean control first reaches the threshold at u170; per seed: [100, 180, 60, 130, 60, 130, 160, 150]
- **Onset by rule: u250** (control there = 24.7 = 51% of final)

Mean control eval and gain per 100 updates:

| block | mean at end | gain |
|---|---|---|
| 0-100 | 10.84 | +8.49 |
| 100-200 | 21.58 | +10.74 |
| 200-300 | 26.21 | +4.63 |
| 300-400 | 27.24 | +1.03 |
| 400-500 | 28.01 | +0.77 |
| 500-600 | 33.57 | +5.56 |
| 600-700 | 35.18 | +1.60 |
| 700-800 | 44.49 | +9.31 |
| 800-900 | 44.15 | -0.34 |
| 900-1000 | 51.52 | +7.37 |

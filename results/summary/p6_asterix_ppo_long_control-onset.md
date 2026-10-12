# Onset rule — p6_asterix_ppo_long_control

Manifests: 20261012-p2-asterix-long-control-ppo-Asterix-MinAtar-f12598d.json. Seeds (n = 4): [0, 1, 2, 3].

Control eval reliability (mean |checkpoint diff| / level, after u0): eval_return_mean 0.096. **Metric: eval_return_mean.**

Final (mean of checkpoints in the last 50 updates): 14.0 (per-seed sd 3.5). Threshold 40% of final = 5.6.

- Exploration floor: u0 (PPO: none)
- Mean control first reaches the threshold at u390; per seed: [470, 410, 390, 270]
- **Onset by rule: u390** (control there = 6.0 = 43% of final)

Mean control eval and gain per 100 updates:

| block | mean at end | gain |
|---|---|---|
| 0-100 | 1.76 | +1.39 |
| 100-200 | 2.84 | +1.09 |
| 200-300 | 4.35 | +1.50 |
| 300-400 | 5.90 | +1.55 |
| 400-500 | 6.70 | +0.80 |
| 500-600 | 8.11 | +1.41 |
| 600-700 | 9.88 | +1.77 |
| 700-800 | 10.38 | +0.50 |
| 800-900 | 11.27 | +0.88 |
| 900-1000 | 12.82 | +1.55 |
| 1000-1100 | 14.02 | +1.21 |
| 1100-1200 | 13.84 | -0.18 |
| 1200-1220 | 13.59 | -0.25 |

# Onset rule — p12_cartpole

Manifests: 20261014-p2-p12-longctl-ppo-CartPole-v1-7138fe2.json. Seeds (n = 4): [0, 1, 2, 3].

Control eval reliability (mean |checkpoint diff| / level, after u0): eval_return_mean 0.113. **Metric: eval_return_mean.**

Final (mean of checkpoints in the last 50 updates): 470.4 (per-seed sd 41.8). Threshold 40% of final = 188.2.

- Exploration floor: u0 (PPO: none)
- Mean control first reaches the threshold at u10; per seed: [10, 10, 10, 30]
- **Onset by rule: u10** (control there = 224.5 = 48% of final)

Mean control eval and gain per 100 updates:

| block | mean at end | gain |
|---|---|---|
| 0-100 | 490.54 | +468.86 |
| 100-200 | 488.18 | -2.36 |

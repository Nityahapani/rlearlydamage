# Research Log

Format: newest entries at the bottom. Each entry lists **what / why / outcome / next**.
Run IDs refer to manifests in `runs/`.

---

## 2026-10-11 — Project start

**What.**
- Chose topic and title: *Where Does Early Damage Live? Weight, Optimizer, and Data Scars in Reinforcement Learning.*
- Target venue: TMLR (rolling submission; judged on correctness of claims and interest to some readers).
- Finished the literature review (`docs/lit_review.md`).

**Key lit findings that shape the design.**
- "Critical periods exist in RL" is **already claimed** (Park et al. 2021, ICMI; de Kleijn et al. 2022; Park et al. 2022), but only for the timing of reward guidance. We do **not** claim to discover critical periods.
- Nikishin et al. 2022 (primacy bias): a fresh network trained on the old buffer recovers. That is one cell of our factorial, run without a deficit. It predicts weight scars dominate for off-policy agents.
- Igl et al. 2021 (ITER): transient non-stationarity permanently affects representations.
- Xu 2026 (arXiv 2608.20442): Adam's first moment carries a perturbation after its source is removed, which motivates **optimizer state as a third carrier**.
- Achille et al. 2019: low-level deficits are permanent and high-level deficits are recoverable. Does this hold for RL-specific deficits?

**Contribution (working).** Causal decomposition of lasting early damage into weight / optimizer / data scars via state surgery, across perceptual, action, and reward deficits, for off-policy and on-policy agents.

**Decisions.**
- Stack: JAX + PureJaxRL-style single-file agents, gymnax/MinAtar first, Craftax-Classic later.
- Stats: rliable IQM + 95% stratified bootstrap CIs; at least 10 seeds for pilots and 30 for headline results.
- Constant learning rate in all critical-period sweeps, to rule out an LR-decay explanation (as Achille did).

**Next.**
- Phase 1: implement and validate baseline PQN + PPO on MinAtar Breakout/SpaceInvaders/Asterix; reproduce reference returns.
- Remaining lit checks (see `docs/lit_review.md` §4).

## 2026-10-11 — Phase 1 baseline #1 (PPO, Breakout-MinAtar): FAILED decision point 1

- Run: `runs/20261011-p1-baseline-ppo-Breakout-MinAtar-3bff72c.json`, summary in `results/summary/`.
- Config: PPO, hidden 128, LR 5e-4 constant, gamma 0.99, 64 envs x 128 steps, 4 seeds, 5M steps. Wall-clock 516 s on 2 CPUs.
- Result: every seed reaches about 7 within 0.5M steps and then plateaus. Seed 0 escapes only at about 4.5M (11.3 at 5M). Final returns: [11.3, 7.0, 7.0, 6.9].
- Sanity checks: random policy 0.42, constant actions 0.0–0.50 (`/tmp/envcheck.py`). So ~7 is real learning, but a local optimum.
- Reference: the gymnax PPO checkpoint reaches R = 28 (gymnax README). Its config (gymnax-blines `agents/Breakout-MinAtar/ppo.yaml`) uses gamma 0.999, 2x256 MLP, LR 5e-4 constant, 10M steps.
- Diagnosis to test: gamma 0.99 is too myopic. Next: a gamma 0.999 pilot.
- Script change: `train_baseline.py --set KEY=VALUE` for config overrides.

## 2026-10-11 — Phase 1 baseline #2 (gamma 0.999): same plateau

- Run: `runs/20261011-p1-baseline-g999-ppo-Breakout-MinAtar-abdb7af.json`. Same config as #1 except gamma 0.999. Wall-clock 547 s.
- Final returns: [7.0, 7.0, 14.7, 7.0]. Seed 2 escaped the plateau at about 3.5–4M steps; the others stayed at about 7.
- Conclusion: gamma is not the cause. PPO on Breakout-MinAtar at this budget is bimodal (plateau at ~7, rare escape). gymnax issue #120 (2026-10-06) also reports "heavy seed-to-seed variance in untruncated Breakout".
- Bimodal baselines would swamp deficit effects at a 4–10 seed budget. Next: pilot the other MinAtar games for smooth, seed-consistent learning.

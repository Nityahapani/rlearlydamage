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

## 2026-10-11 — Phase 1 env pilot (PPO, 2M steps, 4 seeds, hidden 128, gamma 0.99)

| env | return/seed at ~2M | wall-clock | shape |
|---|---|---|---|
| Asterix-MinAtar | 3.1 / 4.4 / 3.7 / 4.1 | 296 s | smooth, seed-consistent, still climbing, slow |
| SpaceInvaders-MinAtar | 83.6 / 81.6 / 85.7 / 71.0 | 337 s | smooth, seed-consistent, steep: best signal per CPU-second |
| Freeway-MinAtar | 42.2 / 43.9 / 46.5 / 45.9 | 412 s | smooth, but slowest and low variance across conditions likely |
| Breakout-MinAtar (runs #1/#2) | plateau ~7, rare escape | ~530 s / 5M | bimodal: unsuitable as primary |

Decision: primary env = **SpaceInvaders-MinAtar**, replication env = **Asterix-MinAtar**. Breakout is reported as a bimodal case.
Caveat: gymnax SpaceInvaders dynamics differ from the original MinAtar (gymnax issue #120: enemy-bullet hit test uses the pre-move cannon position; first wave has 28 aliens). We study *relative* effects of deficits inside one fixed MDP, so this does not threaten validity. It must be disclosed, and absolute returns must not be compared with MinAtar papers.
Also fixed: provenance no longer treats untracked run outputs (runs/, results/) as a dirty tree (commit 7e1f3f4).

## 2026-10-11 — Phase 2 pilot: obs_noise on SpaceInvaders (no critical period; damage = lost time)

- Run: `runs/20261011-p2-pilot-obs_noise-ppo-SpaceInvaders-MinAtar-86ccb44.json`.
  - obs_noise sigma 1.0, duration 0.2 (400k steps), onsets 0/0.1/0.2/0.3/0.4, plus a control. 2M steps, seeds 0–3, wall-clock 2216 s.
- **Reproducibility:** the control is bit-identical to the Phase 1 envpilot SpaceInvaders run (same seeds and config; max abs diff 0.0). Same-seed determinism on CPU holds, which state surgery relies on.
- Analysis: `scripts/analyze_sweep.py` → `figures/p2_pilot_obs_noise.png`.

| window | final | raw damage [95% CI] | equal-recovery damage | delay-adjusted damage |
|---|---|---|---|---|
| control | 81.0 | – | – | – |
| on0 | 72.7 | +8.4 [-4.5, +19.0] | +16.5 [+5.4, +25.8] | -1.1 [-13.8, +11.6] |
| on0.1 | 71.4 | +9.6 [-3.0, +23.4] | +15.6 [+3.4, +30.2] | +0.1 [-12.3, +14.7] |
| on0.2 | 71.5 | +9.6 [+5.8, +15.3] | +6.9 [-1.2, +14.6] | +0.1 [-5.8, +7.6] |
| on0.3 | 68.7 | +12.3 [+8.2, +16.7] | +14.9 [+11.1, +18.4] | +2.8 [+0.3, +4.5] |
| on0.4 | 70.9 | +10.1 [+7.9, +12.3] | +7.9 [+4.7, +10.5] | +0.6 [-1.0, +3.4] |

- **Interpretation:**
  - Every window costs about 10 points at T = 2M, with no dependence on onset.
  - The control is still improving fast (68.7 at 1.6M → 80.8 at 2.0M). Measured against the control at T − 400k ("the deficit only cost lost time"), damage is about 0. A full-resolution check from curves.npz gives −2 to −7, i.e. the noisy agents still learned during the deficit.
  - At this budget and severity, obs_noise leaves **no lasting scar**: decision point 2 is *not* met for obs_noise.
- **Methodological lesson:** raw "final return" damage is confounded with lost training time while the control is still improving. Delay-adjusted damage (now a standard readout) or training to a plateau is required.

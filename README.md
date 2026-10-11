# Where Does Early Damage Live? Weight, Optimizer, and Data Scars in Reinforcement Learning

Research code and reproducible log for a TMLR submission.

**Question.** Deep networks have *critical learning periods*: a temporary deficit early in training
causes permanent damage, while the same deficit later is harmless (Achille et al., 2019).
In RL the agent produces its own training data, so lasting damage could be stored in three places:

| Carrier | What it is | How we test it |
|---|---|---|
| **Weight scar** | damaged network parameters | transplant clean vs. damaged params |
| **Optimizer scar** | damaged Adam moments (m, v, step count) | transplant clean vs. damaged optimizer state |
| **Data scar** | damaged replay buffer / behaviour policy | transplant clean vs. damaged buffer (off-policy) or swap the behaviour policy (on-policy) |

The core experiment is **state surgery**: when the deficit ends, we build hybrid agents from a
*damaged* run and a *clean* run, using all 2×2×2 combinations of {weights, optimizer, data}, and measure final performance.

## Repo layout
```
configs/            YAML configs (base + one file per deficit type)
src/rled/           library code
  deficits.py       deficit windows: onset/duration schedule + deficit transforms
  surgery.py        trainer-state composition (the 2x2x2 transplant)
  metrics.py        plasticity diagnostics (dormant ratio, feature rank, ...)
  provenance.py     records git hash, config, seed, env for every run
  agents/           PQN/DQN (off-policy) and PPO (on-policy), PureJaxRL-style
scripts/            entry points (sweeps, surgery runs, analysis)
tests/              unit tests (run before every experiment commit)
docs/               literature review, design notes
runs/               run registry (small JSON manifests only, committed)
results/            large outputs (git-ignored; summaries go in results/summary/)
LOG.md              dated research log: every decision, run, and result
```

## Reproducibility rules
1. Every run writes a manifest to `runs/` with its git commit, config, seed, and package versions (`rled.provenance`).
2. Runs from a dirty working tree are refused unless `--allow-dirty` is passed, and that flag is recorded in the manifest.
3. Every decision, run batch, and result gets a dated entry in `LOG.md`.
4. Statistics use `rliable` (IQM + stratified bootstrap 95% CIs); never report a mean over fewer than 10 seeds.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # CPU
pip install -e ".[dev,gpu]"        # CUDA 12 JAX
pytest -q
```

## Status
Phases 0–5 done (scaffold, baselines, deficit pilots, state surgery, causal ablations, PQN replication + replay-buffer surgery). See `LOG.md`.

## Status and results so far (2026-10-11)

Full details, tables and caveats are in [`LOG.md`](LOG.md). Every number traces to a manifest in `runs/` and a summary in `results/summary/`.

**Phase 1: baselines (PPO, 2x128 MLP, CPU-only)**
- Breakout-MinAtar is bimodal: most seeds plateau at about 7 (the gymnax reference is 28), and gamma 0.999 does not fix it. Not used as the primary env.
- Primary env: **SpaceInvaders-MinAtar** (smooth and seed-consistent; about 81 at 2M steps). Replication env: Asterix-MinAtar.
- Same-seed reruns are bit-identical (a prerequisite for state surgery).

**Phase 2 pilot: deficit windows on SpaceInvaders** (duration 20% of 2M steps, onsets 0–40%, 4 seeds)

![Phase 2 pilot](figures/p2_pilot_spaceinvaders.png)

- `obs_noise` (sigma 1): about 10 points of raw damage at every onset, but about 0 once lost training time is accounted for. No lasting scar.
- `action_mask` (2 of 4 actions): early masking is free; damage grows with onset. No window exceeds the lost-time null after the CI correction (latest: +5.4 [-0.8, +11.6]). The per-seed random mask was a design flaw, to be replaced with fixed "no fire" and "no movement" masks.
- **Recovery test** (fixed masks shared by all seeds, deficit-free evaluation, 3.6M steps; `figures/recovery_all.png`):
  - Early movement or fire restriction acts as pure delay and recovers fully.
  - Late restriction causes *unlearning during the mask* (eval falls from 45 to 15–19). Recovery leaves a 6–7% gap at 2M post-removal steps, consistent with lost time rather than a lasting scar.
- **State surgery** (Phase 3 pilot, `figures/surgery_move_late.png`). At removal of the late movement restriction, a 2^3 factorial over actor weights × critic weights × optimizer state:
  - The **actor weights carry 89%** of the recovery damage: -12.7 [-13.8, -11.6] eval-return AUC, consistent across all 4 seeds.
  - The critic effect is not significant, and the optimizer state carries about 0.
  - Swapping just 10 updates of *experience* (behaviour policy) moves recovery by -6.6 / +8.3. In on-policy RL, weight and data scars are entangled.
- **Phase 4 causal ablations** (`figures/phase4_move_late_*.png`, LOG):
  - Resetting the damaged actor is far worse than keeping it (−27 AUC; recovery equals learning from scratch). A clean-actor transplant restores C. The damage is lost competence, not a toxic weight state.
  - 10 updates of clean-actor experience help (+8.3 AUC) only when the behaviour policy is good *and* close to the learner. KL-matched degraded data and good-but-unrelated (cross-seed) data do not help, and frozen self-data ≈ online. This looks like implicit policy guidance, not a separable data scar.
  - Freezing the actor representation or anchoring to the onset policy halves the collapse at removal (+20 eval) but leaves recovery AUC unchanged. Drift is fast-reversible; the residual gap is lost training time.
- **Phase 4 replication, fire restriction:** same picture.
  - Actor reset: −23 AUC. Transplant restores C.
  - No separable data scar.
  - Drift suppression halves the collapse but does not change recovery AUC.
  - An Asterix replication was aborted: PPO has no competence on Asterix at u100, so there was no collapse to ablate.
- **Phase 5 (PQN / PQN-R, `figures/p5_recovery_move.png`, `figures/p5_surgery_pqnr.png`):** the recovery pattern replicates in a value-based agent.
  - Early restriction = delay. Late restriction = collapse to ~3 during the mask.
  - In PQN the residual gap is lost time.
  - In PQN-R (a 100k replay buffer) there is a small excess (+4.3 AUC, delay-adjusted), concentrated early.
  - Params × buffer surgery: damaged network + clean buffer −5.0 [−8.9, −1.1] AUC; clean network + damaged buffer −0.1 [−1.7, +1.6]. The network carries 85% of the damage.
  - Caveat: the buffer turns over in 24 updates.
- Decision point 2 (a peak-then-fall damage curve) is **not met yet**.
- CIs are paired t-intervals over seeds (earlier bootstrap CIs were too narrow at n=4 and were corrected; see LOG).
- Methodological note: while the control is still improving, raw final-return damage is confounded with lost time, so we report delay-adjusted damage.

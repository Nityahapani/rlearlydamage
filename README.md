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
Phase 0: scaffold and literature review. See `LOG.md`.

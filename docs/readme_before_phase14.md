# Superseded README snapshot

**Historical record only. It contains obsolete early-phase claims and sample sizes; use the current README, manuscript and inference map.**

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

## Causal narrative (synthesis, 2026-10-13)

Full synthesis: [`docs/causal_narrative.md`](docs/causal_narrative.md). Summary figure: `figures/causal_summary.png`, built by `scripts/make_causal_summary.py`, which asserts every plotted value verbatim against its `results/summary/` source.

![causal summary](figures/causal_summary.png)

- **The phenomenon (PPO, PQN, PQN-R; SpaceInvaders-MinAtar).** An early action restriction acts as pure delay: no early critical period was found. A late restriction, imposed after the agent is competent, causes unlearning during the mask. After removal the remaining cost is mostly lost time; only PQN-R shows a small excess (+4.3* delay-adjusted AUC).
- **Central 1: in PPO, the damage is in the actor.**
  - Actor main effect −12.7 [−13.8, −11.6] (89%). Critic −2.3 n.s.; Adam state −0.1 n.s.
  - A clean-actor transplant recovers the deficit (+10.9* for movement, +12.9* for fire).
  - Resetting the damaged actor is worse (−27.2*) and tracks learning from scratch: the damage is lost competence, not toxic state.
  - A clean *behaviour* policy repairs almost as well as a transplant (+8.3*). The data channel is guidance, not a data scar.
- **Central 2: in PQN-R, the network carries the damage robustly.**
  - Network main effect −5.6* (84%) with a 100k buffer and −3.2* (57%) with 1M. On an identical buffer the damaged network still trails (6b B−D −2.9 [−4.0, −1.8]*).
  - The buffer contributes only when it outlives the deficit: data transfer +0.0 at 100k, −1.7* at 1M.
  - Its effect persists after the restriction-era data is fully overwritten (−2.9* in the "gone" phase), so it acts as a data→network mediator, not a stored scar.
  - Caveat: the optimizer was bundled with the params by prespecified rule and is tested separately only in PPO.
- **Secondary, qualified: replay composition (PQN-R, 1M buffer).**
  - Purging restriction-era data helps (+4.6*). A targeted purge beats an age-matched sham (+2.6*), but that contrast is confounded with residence time.
  - With exposure fully matched (6d, n=32, pre-registered), the identity of the removed transitions accounts for Δ = +0.46 [−0.19, +1.11]: equivalence at SESOI ±1.0 AUC, but only marginally (TOST p = 0.0495).
- **Limits.**
  - PPO surgery has n=4.
  - One environment (gymnax #120), one deficit family, constant LR.
  - Plasticity was not measured directly; the PQN-R optimizer is not separated.
  - The "early damage" title is not supported. Suggested reframe: *deficit damage after competence*.
  - Recommended next step: extend the PPO surgery to ≥12 seeds.

## Status
Phases 0–6d are done:
- scaffold, baselines and deficit pilots;
- PPO state surgery and causal ablations;
- PQN/PQN-R replication and replay-buffer surgery;
- 12-seed surgery, the 1M-buffer persistence test and the Asterix onset pilot;
- the pre-registered purge, targeted-purge (6b), identity × residence (6c) and matched-identity (6d) tests;
- the cross-algorithm synthesis.

See `LOG.md`.

## Results by phase (last updated 2026-10-13)

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
- **Phase 6 (`figures/p6_surgery_pqnr_n12.png`, `figures/p6_surgery_pqnr1m_n8.png`):**
  - 100k surgery at **12 seeds**: parameter transfer −4.6 [−6.1, −3.0] (12/12 seeds negative), data transfer +0.0 [−1.0, +1.0]; network 84% of damage. Restriction-era data is gone from the buffer 25 updates after removal (write-time tags).
  - **1M buffer** (restricted data persists ~245 updates, 41% of the buffer at removal; 8 seeds): the deficit is milder (D −50% vs −87% at removal), and a small data carrier appears: data transfer −1.7 [−3.3, −0.2], buffer main effect −2.4 [−3.4, −1.4] (8/8), network still larger (57% vs 43%). The data effect is ~0 early and persists after the restricted data is overwritten, i.e. it acts by damaging the network that trains on it, not as an ongoing buffer effect. Prediction intervals span 0.
  - **Replay purge (pre-registered, 8 seeds, `figures/p6_purge.png`):** replacing the damaged agent's 1M buffer with fresh clean-actor data at removal improves recovery by +4.6 [+2.2, +6.9] AUC (Holm p = 0.007), mostly *after* the retained buffer's restricted data is already gone (late +6.5*). Restriction-era replay data leaves a scar after it is removed (via the network trained on it). Purging 50 updates later loses little (B − C +1.0, n.s.). After an immediate purge there is no residual AUC gap to the clean control (−0.1), but on identical data the damaged network still trails (−1.9*). The damage-specific part of the purge benefit (net of replacement-data quality) is +2.8 [−0.2, +5.7].
  - **Targeted vs sham purge (pre-registered, 12 seeds, `figures/p6b_targeted.png`):** replacing only the restriction-era 41% of the buffer beats replacing an equal, adjacent-age block of non-restricted data: B − C +2.6 [+0.9, +4.3] AUC (Holm p = 0.006), and beats retaining it: +3.5 (Holm p = 0.001). The damaged network still trails the clean one on an identical buffer: −2.9 (Holm p = 0.0004). Caveat: the sham's replacement data is overwritten ~100 updates earlier (2.1× less sampled exposure), so B − C is not attributed solely to data identity. While exposure is still identical (+10..+40), B − C is already +2.5 [+0.2, +4.8]. **Identity × residence 2×2 (pre-registered, 12 seeds, `figures/p6c_2x2.png`):** blocks are repositioned so the replacement data stays in the buffer equally long. Restriction-era removal at matched residence gives B − Cp +1.1 [−0.2, +2.5] (Holm p = 0.18). The residence main effect is +1.5 [+0.2, +2.7] (Holm p = 0.07). The Phase 6b gap splits about evenly between data identity and replacement residence, and neither alone is significant after Holm. The identity effect is clear only early in recovery (+0.7*, +2.2* in the +10..150 window). Strong data-identity claims are therefore not supported. The network carrier (B − D −2.9*) is unaffected. **Matched-identity purge (pre-registered, n = 32, SESOI 1.0 AUC, `figures/p6d_idswap.png`):** both arms start from one common buffer in which half the env columns of the restriction-era rows hold restricted data (R) and half hold the clean fork's same-epoch transitions (U). One arm purges R, the other purges U. Both use the same pool, slots, ages, exposure, retained residence, seams and learner state. Δ = +0.46 [−0.19, +1.11] (90% CI upper 0.998). The pre-registered outcome is equivalence: an identity effect of 1 AUC point or more is rejected, but the margin is thin (0.002). Restriction-era data identity is at most a small damage carrier; the network carrier remains the robust result.
  - **Asterix (PPO):** a 10M-step control gives a meaning-matched onset of u390 (43% of final), provisional because the plateau is weak.
- Decision point 2 (a peak-then-fall damage curve) is **not met yet**.
- CIs are paired t-intervals over seeds (earlier bootstrap CIs were too narrow at n=4 and were corrected; see LOG).
- Methodological note: while the control is still improving, raw final-return damage is confounded with lost time, so we report delay-adjusted damage.


## Phase 12: expanded generalization and claim calibration

The updated manuscript is [`paper/main.pdf`](paper/main.pdf), with source [`paper/main.tex`](paper/main.tex). The fourth-round response is [`docs/response_round4.md`](docs/response_round4.md).

Six new settings reached their planned eight seeds; Freeway has only four complete seeds and is descriptive. Optional independent-seed replications did not run. New contrasts on reused IDs are not described as fresh-seed replication. The paper separates same-update cost from harm beyond lost training time and defines “carrier” only through the tested transplantation procedure.

- [Generalization panel](results/summary/p12_panel-analysis.md)
- [PQN-R reward-flip primary and matched-horizon analyses](results/summary/p12_g7_pqnr_rflip-analysis.md)
- [Interim/outlier sensitivity](results/summary/p12_interim_predictors-posthoc.md)
- [Result availability and reset limitations](docs/result_availability.md)
- [File hashes and completion inventory](docs/result_inventory.json)

All surviving saved NPZ arrays are included even though new generated outputs remain ignored by default. Full generated runner-state pickles are not archived; the availability note explains which artifacts must be regenerated.


## Phase 13: reference-aware independent replication and prospective checks

Updated manuscript: **Where Do Recovery Costs Live? Reference-Aware State Surgery in Deep Reinforcement Learning** ([PDF](paper/main.pdf), [source](paper/main.tex)). Two tiers are kept explicit: same-update gap attribution versus directly tested pause-relative recovery cost. A negative finite-window AUC is not an irreversible deficit.

All five registered cohorts completed on two CPU cores: A/B/C/E each8 unused training seeds; D16 held-out seeds with coefficients frozen before its first run. The fixed five-claim Holm family supports the Asterix pause-reference and actuator-noise gap claims. CartPole's pause allocation differs materially from its same-update allocation (actor75% versus96%, critic32% in the pause comparison); its stronger compound dominance test does not pass that correction.

The controlled label assay matches actual transitions, ages, collection and update schedules, then proves all non-learner states equal after washout. Its matched primary is inconclusive, not a mediation success. The prospective frozen acute+KL model has MSE68.78 versus32.27 for acute gap and fails the10% utility rule. Secondary correlations/swaps are not used to rescue failed primaries. These results remain scoped procedural effects, not unique internal encoding or a general deep-RL law.

- [Pre-registration](prereg/phase13.json)
- [Frozen diagnostic coefficients and training data](prereg/p13_frozen_predictors.json)
- [Full primary-family analysis](results/summary/p13_confirmatory-analysis.md)
- [Prospective comparison](results/summary/p13_D_prospective-analysis.md)
- [Fifth-round response](docs/response_round5.md)
- [Result inventory/availability](docs/result_availability.md)

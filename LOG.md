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

## 2026-10-11 — Phase 2 pilot: action_mask on SpaceInvaders (later = worse; no early critical period)

- Run: `runs/20261011-p2-pilot-action_mask-ppo-SpaceInvaders-MinAtar-e9cecf7.json`.
  - allowed_frac 0.5 (2 of 4 actions), duration 0.2, onsets 0–0.4. Wall-clock 1673 s.
  - Control: shared with the obs_noise sweep (identical config with an empty window; the bit-identical determinism check above).
- Combined figure: `figures/p2_pilot_spaceinvaders.png`.

| window | final | raw damage [95% CI] | equal-recovery damage | delay-adjusted damage |
|---|---|---|---|---|
| on0 | 83.4 | -2.4 [-9.2, +3.0] | +1.4 [-10.1, +10.9] | -11.9 [-21.8, -5.3] |
| on0.1 | 76.9 | +4.2 [-3.6, +13.0] | +10.0 [+4.6, +18.5] | -5.3 [-12.4, +5.1] |
| on0.2 | 75.0 | +6.1 [-0.6, +11.6] | +7.2 [+4.2, +11.3] | -3.4 [-10.6, +3.6] |
| on0.3 | 68.9 | +12.1 [+6.8, +17.4] | +12.6 [+7.5, +16.8] | +2.6 [-1.6, +6.9] |
| on0.4 | 66.1 | +14.9 [+12.8, +17.1] | +9.7 [+6.8, +12.6] | +5.4 [+2.1, +9.0] |

- **Interpretation:**
  - The opposite of a critical period: masking at the start costs nothing (not even lost time). Damage grows with onset, and only the latest window exceeds the lost-time null (+5.4 [2.1, 9.0]).
  - Late windows also have the least recovery time (0.6M steps for on0.4), so "slow re-adaptation" and "lasting scar" are not yet separable. A longer-recovery run is needed.
- **Design flaw found:** the allowed-action subset is drawn per seed, so the deficit differs *in kind* across seeds.
  - SpaceInvaders actions are [noop, left, right, fire].
  - Seed 0 kept {left, right}: cannot fire, so it cannot score.
  - Seed 1 kept {noop, left} and seed 2 kept {noop, right}: cannot fire.
  - Seed 3 kept {noop, fire}: cannot move.
  - Fix before any further action_mask runs: use a fixed, semantically chosen mask shared by all seeds (e.g. remove "fire", or remove movement), and report each as a separate deficit.

## 2026-10-11 — Recovery test: fixed action masks, deficit-free evaluation (SpaceInvaders)

**Design (fixes the per-seed-mask flaw)**
- Allowed-action sets are fixed and identical for every seed. SpaceInvaders action indices: 0 noop, 1 left, 2 right, 3 fire.
- Two interventions:
  - **movement restriction**: allowed {noop, fire}.
  - **fire restriction**: allowed {noop, left, right}. Reward is essentially unobtainable, so this is analyzed as a *different intervention*, not a matched-severity replicate.
- Windows, in updates (1 update = 8192 steps):
  - early [0, 50) = 0–0.41M steps.
  - late [100, 150) = 0.82–1.23M steps.
  - All runs: 440 updates (3.6M steps), seeds 0–3.
- **Evaluation protocol:**
  - Every 10 updates, every agent is evaluated by `make_eval`: full action set (no training mask), clean observations, true reward, 64 episodes, stochastic policy.
  - The *same* eval key is used for every agent and checkpoint (common random numbers).
  - Mask boundaries coincide with eval checkpoints. The as-experienced (masked) training return is also logged.
- Runs:
  - `runs/20261011-p2-recovery-move-ppo-SpaceInvaders-MinAtar-fea755e.json` (control, move_early, move_late; 1883 s).
  - `runs/20261011-p2-recovery-fire-ppo-SpaceInvaders-MinAtar-806d05a.json` (fire_early, fire_late; 1254 s).
  - Control shared; late runs are bit-identical to the control before their mask, as expected.
- Analysis: `scripts/analyze_recovery.py`. Figures: `figures/recovery_move.png`, `figures/recovery_fire.png`, `figures/recovery_all.png`.

**Results.** Gap = control − deficit, paired by seed, with a 95% bootstrap CI over 4 seeds. Delay-adjusted gap = control(u − 50) − deficit(u).

| config | Q1a during mask (unmasked eval) | Q1b during mask (as experienced) | Q2 at removal | Q3 +0.41M | Q3 +0.82M | Q4 +2.05M | delay-adj at Q4 |
|---|---|---|---|---|---|---|---|
| move_early | 5.6 vs 15.7 | 3.0 vs 15.6 | 5.4 vs 28.2 (+22.8) | +17.0 [+13.6, +21.5] | +4.5 [-6.8, +15.7] | **-6.6 [-15.3, +1.9]** | -16.4 |
| fire_early | 4.2 vs 15.7 | 0.0 vs 15.6 | 4.5 vs 28.2 (+23.7) | +18.0 [+13.7, +22.3] | +12.9 [+5.0, +19.9] | **+5.5 [-2.0, +11.1]** | -4.3 |
| move_late | 24.5 vs 50.8 | 3.1 vs 49.3 | 18.5 vs 58.7 (+40.1) | +13.6 [+9.2, +20.7] | +10.9 [+8.2, +13.7] | **+7.2 [+3.9, +10.7]** (7%) | -2.0 [-3.1, -0.9] |
| fire_late | 22.6 vs 50.8 | 0.9 vs 49.3 | 15.2 vs 58.7 (+43.4) | +17.4 [+8.3, +26.6] | +16.1 [+9.7, +22.5] | **+6.9 [+2.6, +13.6]** (6%) | -2.4 [-5.4, +0.7] |

Final eval at 3.6M (mean of seeds): control 112.6; move_early 114.9; fire_early 113.7; move_late 106.2; fire_late 105.8.

**Interpretation**
1. **Early masks behave as pure delay.**
   - At removal, both early agents score like an untrained agent (delay-adjusted Q2 about -1). Fire restriction gives zero reward for 0.41M steps, i.e. no learning signal.
   - Both then catch up completely. move_early even ends slightly *above* the control (Q4 -6.6, CI includes 0).
   - No critical-period signature.
2. **Late masks cause active unlearning during the mask.**
   - Deficit-free eval *falls* from 45.4 (update 100, mask onset) to 18.5 (movement) / 15.2 (fire) at removal.
   - Delay-adjusted Q2 is +27 / +30: the agents end the mask worse than they were before it.
   - Hypothesis to test: masked-action logits get no gradient while allowed-action logits and shared features are re-optimized for the restricted game, so the unmasked policy drifts.
3. **The late-mask gap shrinks but has not closed by 2.05M post-removal steps.** It goes about +40 → +11/+16 → **+7 (6–7%)**, with CIs excluding 0.
   - However, the delay-adjusted gap at Q4 is about -2, and the control is still improving at about +15 per 100 updates. So the residual gap equals about 0.37M steps of lag, i.e. roughly the window length.
   - **No evidence of a lasting scar beyond lost (and relearned) time.**
   - A decisive "does it close" answer needs a longer horizon or a plateaued control, plus more seeds.
4. **Fire vs movement:** fire restriction is more destructive *during* the mask (0 vs ~3 reward). But its recovery profile is nearly indistinguishable from movement restriction at both onsets.

**Caveats:** 4 seeds (crude bootstrap); one game; PPO with constant LR; stochastic-policy eval only; Q3/Q4 use 3-checkpoint smoothing.

## 2026-10-11 — Phase 3 pilot: state surgery at removal of the late movement restriction

**Design**
- Run: `runs/20261011-p3-surgery-move-late-ppo-SpaceInvaders-MinAtar-365cf2b.json`, `scripts/surgery_ppo.py`. Wall-clock 2477 s.
- **Fork:** a shared history to update 100. Then clean C trains to 150 with no mask; damaged D trains to 150 with the movement restriction (allowed {noop, fire}). States at removal are saved in `results/<stem>/states_w1.pkl` (git-ignored).
- **Carriers for PPO.** PPO has no replay buffer, so the 2^3 factorial is over **actor weights × critic weights × optimizer state** (Adam moments and count). The actor and critic are separate towers, so the split is clean.
  - The **data** carrier is tested with 2 extra arms: for the first 10 post-surgery updates, rollouts are collected by the *other* run's frozen actor. The learner's own critic is used, and PPO's ratio uses the behaviour log-prob.
  - **This deviates from the original plan and must be stated in the paper.**
- Every arm continues 150 updates (1.23M steps) with no deficit. Env state, obs, RNG and update counter all come from C (identical future randomness).
- Evaluation follows the recovery-test protocol (deficit-free, same eval key).
- **Sanity checks passed:**
  - (a) The fork reproduces the recovery-test evals at removal (max abs diff 0.0004, i.e. rounding of stored values).
  - (b) The CCC arm reproduces the original control's continuation exactly.
  - (c) At +0, the eval depends only on the actor.
- Analysis: `scripts/analyze_surgery.py` → `results/summary/20261011-p3-surgery-move-late-analysis.md`, `figures/surgery_move_late.png`. The primary readout is the recovery AUC (mean eval over +10..+150).

**Results** (paired by seed; 95% paired t-interval, 4 seeds)
- Total damage (DDD − CCC), AUC: -14.3 [-17.5, -11.2].

| carrier | main effect, AUC | share of total | at +50 | at +150 |
|---|---|---|---|---|
| **actor** | **-12.7 [-13.8, -11.6]** (all 4 seeds -12.1 to -13.6) | **89%** | -15.2 [-16.4, -13.9] | -7.8 [-18.1, +2.5] |
| critic | -2.3 [-6.2, +1.7] | 16% (n.s.) | -3.2 [-7.9, +1.6] | -2.3 [-8.6, +3.9] |
| optimizer state | -0.1 [-2.8, +2.6] | ~0% | +0.1 | +1.9 |

- Transplants:
  - Damaged actor alone in a clean agent (DCC − CCC): -13.0 [-19.5, -6.4].
  - Clean actor alone rescues a damaged agent (CDD − DDD): +10.9 [+7.6, +14.2].
  - Critic and optimizer transplants: all CIs include 0. Interactions are small (|AUC| ≤ 1.3).
- **Data arms (10 updates of behaviour swap):**
  - Damaged data into a clean agent (CCC+bD − CCC): **-6.6 [-8.9, -4.3]**. Eval drops from 58.7 to 33.0 at +10.
  - Clean data into a damaged agent (DDD+bC − DDD): **+8.3 [+5.6, +11.0]** (58% of total damage). Eval goes from 18.5 to 47.1 at +10, vs 34.2 for DDD.

**Interpretation**
- The late-mask damage lives in the **actor weights**: the policy drift accumulated under the mask.
- It does **not** live in the optimizer state, despite Xu (2608.20442)'s first-moment carry-over in supervised settings. Constant LR and short Adam memory (beta1 0.9, beta2 0.999 → ~1000 steps vs 150 updates × 32 minibatch steps after surgery) are a plausible reason. To test: a long-beta2 or low-LR variant.
- The critic carries little or nothing.
- In on-policy RL, the "weight" and "data" carriers are **entangled**: the damaged actor harms recovery partly by *collecting poor experience*. Only 10 updates of clean-actor data recover ~58% of the total damage.
  - Caveat: the behaviour swap is also an importance-weighted off-policy PPO update from a different policy, so part of the effect may be implicit policy guidance rather than "data" per se. A cleaner control (e.g. same-quality data from an unrelated policy) is needed before calling it a data scar.

**Correction to earlier entries (CI method).** The earlier tables used a percentile bootstrap over 4 seeds, which is anti-conservative. All analysis scripts now use paired t-intervals when n < 10, and the figures and tables were regenerated. Changed conclusions:
- Recovery test, fire_late at +2.05M: +6.9 [-4.1, +17.9], now **includes 0**. move_late stays significant at +7.2 [+1.3, +13.2]; its delay-adjusted value is -2.0 [-4.1, +0.1].
- Phase 2 pilot, action_mask on0.4 delay-adjusted: +5.4 [-0.8, +11.6], now **includes 0**. obs_noise on0.3 delay-adjusted: +2.8 [-1.1, +6.8], now **includes 0**.
- So no Phase 2 window shows significant damage beyond lost time.

## 2026-10-11 — Phase 4: causal ablations at removal of the late movement mask

**Run.** `scripts/phase4_ablations.py`, manifest `runs/20261011-p4-phase4-move-late-ppo-SpaceInvaders-MinAtar-9a789ef.json`, commit 9a789ef, 2818 s CPU.
- Summary: `results/summary/20261011-p4-phase4-move-late-ppo-SpaceInvaders-MinAtar-9a789ef.json`.
- Analysis: `…-analysis.md` (`scripts/analyze_phase4.py`).
- Figures: `figures/phase4_move_late_{reset_transplant,behaviour,drift}.png`.
- Setup: 4 seeds; mask on updates [100, 150) with allowed actions {noop, fire}; recovery 150 updates; deficit-free eval every 10 updates with common random numbers; K = 10 behaviour updates; anchor β = 1.

**Checks passed.**
- The C/D forks are bit-identical to the Phase 3 saved states (params and optimizer state).
- D_online AUC 65.0 equals Phase 3 DDD, D_actor_transplant 75.9 equals CDD, and D_beh_clean 73.2 equals DDD+bC.
- The tempered-T bisection hits the KL target exactly (T = 2.7–4.6).

**Ablation 1: actor reset vs transplant** (AUC, paired difference vs D_online, 95% t-CI)
- Actor reset (critic and optimizer kept): **−27.2 [−39.4, −15.1]**.
  - Its recovery curve matches a from-scratch agent: 26.9 at +50 vs 28.2 for the Phase 2 control at update 50, and 59.9 vs 58.7 at 150 (cross-run comparison, no CI).
  - So the trained critic and optimizer give a fresh actor nothing.
- Actor + full Adam reset: −24.5 [−26.8, −22.1]. Compared with the actor reset alone: +2.8 [−7.6, +13.1], n.s.
- Clean actor transplant: +10.9 [+7.6, +14.2]. Compared with C_cont: −3.4 [−7.5, +0.7], n.s.
- **Answer:** recovery is about *preserving a good learned policy*, not removing damaged parameters. The damaged actor is far better than a fresh one, so it keeps most of its competence. We see no sign of a "toxic" weight state that blocks learning.

**Ablation 2: breaking the actor–experience loop** (learner = D; the first K = 10 updates are collected by the behaviour policy; AUC vs D_online)

| Behaviour policy | AUC vs D_online | +10 | Behaviour KL | Clip fraction |
|---|---|---|---|---|
| Matched clean actor | **+8.3 [+5.6, +11.0]** | +12.9* | 0.145 | 0.35 |
| Frozen self (fixed pre-intervention data distribution) | −1.9 [−6.3, +2.4] n.s. | −10.6, borderline | 0.024 | 0.16 |
| Cross-seed clean actor (good but unrelated) | +2.4 [−5.6, +10.4] n.s. | | 0.46 | 0.54 |
| Tempered self (KL-matched to clean, low quality) | −5.1 [−14.3, +4.1] n.s. | −22.1* | 0.094 | 0.25 |

- Clean − tempered: **+13.4 [+3.4, +23.4]**.
- Clean − cross-seed: +5.9 [−3.4, +15.1] on AUC (n.s.) but +14.7* at +10.
- Clean behaviour − actor transplant: −2.6 [−6.2, +0.9].
- **Answer:**
  - The off-policy update itself is roughly neutral: frozen self ≈ online.
  - Mismatch alone is not the benefit: the tempered and cross-seed arms do not help.
  - High-return experience from an *unrelated* good policy helps little, because the large ratios get clipped (clip fraction 0.54).
  - The benefit needs experience from a good policy that is *close to the learner*. This is consistent with implicit policy guidance through the clipped ratio (behaviour cloning of a nearby good policy), not with a distinct data carrier.
  - The effects are transient: all behaviour-arm differences at +150 include 0.
  - **No evidence for a data scar separable from the actor.** In on-policy PPO, experience is a mediator of the actor's state, not an independent carrier.
- Caveat: the tempered arm also changes exploration (entropy 0.95 vs 0.49).

**Ablation 3: drift as the mechanism** (fixed onset bank, about 2000 obs per seed)

| At removal | C | D | F (a0/a1 frozen) | A (anchor) |
|---|---|---|---|---|
| Eval | 58.7 | 18.5 | 38.7 | 39.6 |
| KL(onset‖t) | 0.24 | 0.38 | 0.04 | 0.02 |
| 1 − CKA(a1) | 0.18 | 0.25 | 0 | 0.02 |
| p(left+right) | 0.46 | 0.38 | 0.42 | 0.44 |
| Mean V on bank | 10.6 | 4.2 | 4.0 | 4.1 |

- The manipulation worked:
  - repr drift (1 − CKA(a1)): F − D −0.25 [−0.44, −0.05], A − D −0.23 [−0.43, −0.02];
  - eval at removal: F − D **+20.1 [+8.2, +32.0]**, A − D **+21.1 [+13.1, +29.0]**.
- KL_onset differences have wide CIs: one D seed has KL 0.79 and eval 4.3.
- Across D/F/A × seeds, Spearman(eval at removal, KL_onset) = −0.80 (descriptive).
- **Drift suppression prevents about half of the collapse.** F/A hold the onset level (45 → 39) but miss the improvement C made during the window (45 → 59).
- **But recovery AUC is unchanged:**
  - F_cont − D_online: +1.1 [−3.4, +5.7];
  - A_cont − D_online: −0.0 [−1.8, +1.7];
  - both remain about 13–14 AUC below C_cont, about the same as D.
  - D regains the drift-lost competence within about 10–20 updates.
- The masked actions' probabilities barely move (masked logits get no direct gradient). The drift is a redistribution among the allowed actions plus representation change, not suppression of the masked actions.
- The critic's value on the bank falls by the same amount in D/F/A: it tracks the restricted policy.
- **Answer:** drift causes the collapse *at removal*, but it is quickly reversible. The persistent recovery gap is the learning missed during the window (lost time), not drift.

**Overall revision of the story.** The late-mask damage is real and lives in the actor, but it behaves as *lost and temporarily overwritten competence*, not as a scar:
- resets hurt;
- drift is fast-reversible;
- data effects are transient guidance;
- the residual gap is lost training.

This agrees with the delay-adjusted Phase 2 / recovery results. Caveats: 1 env, 1 deficit, 4 seeds, a single β and K. The freeze restores a0/a1 after each step, but Adam moments for a0/a1 still accumulate during the window.

## 2026-10-11 — Phase 4 replication: fire restriction (SpaceInvaders, allowed {noop, left, right})

**Run.**
- Manifest: `runs/20261011-p4-phase4-fire-late-ppo-SpaceInvaders-MinAtar-8456a9b.json`.
- Analysis: `results/summary/…fire-late…-analysis.md`.
- Figures: `figures/phase4_fire_late_*.png`.
- Same protocol as the movement run. D at removal is 15.2, equal to the Phase 2 fire_late value. Shared history and C fork are identical to the movement run.
- (A first launch was OOM-killed when a PQN smoke test ran concurrently. The sandbox has ~2 GB RAM and one Phase 4 process uses ~1.2 GB, so all runs are now strictly sequential.)

**Ablation 1 (AUC vs D_online).**
- Actor reset: **−23.4 [−35.6, −11.1]**.
- Actor + Adam reset vs actor reset: +0.5, n.s.
- Transplant: **+12.9 [+1.3, +24.5]**.
- Transplant vs C: −4.5 [−6.9, −2.1].
- The reset arm again tracks learning from scratch: 28.1 at +50 vs 28.2.
- → Replicates: keeping the learned policy matters, and removing damaged weights hurts.

**Ablation 2.**
- Clean behaviour: +9.6 [−6.0, +25.3] AUC, n.s.; at +10: +19.9 [+4.5, +35.3].
- Frozen self: −2.7 n.s.
- Cross-seed clean: +4.7 n.s.
- Tempered: −4.1 n.s. on AUC; at +10: −18.7*.
- Clean − tempered: +13.7 [+3.8, +23.7]*.
- Cross-seed − tempered: +8.8 [+6.1, +11.5]*. Here unrelated good data helps more than in the movement run.
- Clean − cross-seed: +4.9 n.s.
- → Qualitatively replicates: no separable data scar, and transient guidance from good nearby data. The matched-vs-unrelated distinction is weaker here.

**Ablation 3.**
- Unlike movement, the masked action's own probability *does* fall (p(fire) 0.50 → 0.36). The critic's value collapses (8.9 → 1.5), because without fire there is no reward.
- F/A prevent the drift: KL_onset −0.26*/−0.29*, p(fire) +0.15*.
- They also halve the collapse: eval at removal +21.6*/+23.9*.
- But AUC: +5.9 [−6.1, +17.9] and +4.7 [−9.3, +18.8], n.s. Both remain about −12 vs C.
- Spearman(eval at removal, KL_onset) = −0.94.
- → Replicates: drift causes the collapse, which is quickly reversible; the persistent gap is lost training.

## 2026-10-11 — Phase 5: PQN recovery replication and replay-buffer surgery (SpaceInvaders, movement restriction {noop, fire})

**Agents** (`src/rled/agents/pqn.py`, commit faf0917; tests in `tests/test_pqn.py`).
- **PQN** (Gallici et al., ICLR 2025, purejaxql MinAtar config): LayerNorm Q-network, no target network, no replay buffer.
  - Each update: 128 envs × 32 steps = 4096 transitions; Q(λ = 0.65) targets from the pre-update network.
  - 2 epochs × 32 minibatches of 128 (64 gradient steps); RAdam, gradient clip 10.
  - ε 1 → 0.05 over 1M env steps (244 updates).
- **PQN-R** (the user chose to run both): identical network, exploration, collection and 64 gradient steps per update. Transitions go to a FIFO buffer of 100k (bit-packed obs; s′ is the next slot per env), and batches of 128 are sampled uniformly.
  - 1-step targets with the online network (no target network).
  - Replay ratio 2, equal to PQN's 2 epochs. Buffer turnover is 24.4 updates.
- **Deviations from purejaxql:**
  - MLP 2×128 torso instead of the CNN, matching our PPO and the CPU budget;
  - constant LR 5e-4 instead of linear decay (critical-period design);
  - the off-by-one bootstrap at step T−2 of the λ-recursion is corrected;
  - action mask.
- **Mask semantics:** while active, ε-greedy acts only over allowed actions (random draws included), and bootstrap maxima are over allowed actions.

**Matching to PPO** (recorded in each manifest under `exposure`).
- The window is matched in **env steps**: 409,600 = PPO 50 updates × 8192 = PQN 100 updates × 4096.
- Training exposure differs:

| Per window | PQN | PPO |
|---|---|---|
| Gradient steps | 6,400 | 1,600 |
| Sample-gradients | 819k | 1.64M |
| Exploration | ε-greedy schedule | Entropy-regularised stochastic policy |

- **Late onset by meaning, rule fixed before the arms were run:** the first 10-update multiple at which (i) ε has reached its floor (≥ u245) and (ii) the control's mean eval is ≥ 40% of its final value. The 40% figure is PPO's control at u100: 45.4 / 112.6.
  - Both variants reach 40% earlier (u180 / u190), so the ε condition binds: **late = [250, 350), early = [0, 100)**.
  - At u250 the control is at ~58% (PQN) and ~46% (PQN-R) of its final level, a later stage than PPO's onset.
- **Horizon:** 1000 updates (4.1M steps); 650 post-removal updates (2.66M steps; PPO used 2.38M).
- The controls' curves flatten:
  - PQN gains −1.1, −5.0, +0.8 per 100 updates over u700–1000;
  - PQN-R +5.5, +1.1, +1.3.

**Evaluation metric (protocol change, decided on the control only).**
- Greedy (ε = 0) eval of PQN was very brittle from checkpoint to checkpoint. For example, one control scored [50, 27, 134, 26] at u800 and [151, 137, 143, 147] at u900. With common random numbers, all 64 episodes flip together when the argmax flips.
- Every checkpoint now records greedy *and* ε = 0.05 eval (the training floor).
- Prespecified rule: primary = the metric with the lower control checkpoint-to-checkpoint noise after the ε floor.

| Variant | ε = 0.05 noise | Greedy noise | Primary |
|---|---|---|---|
| PQN | 0.156 | 0.379 | ε = 0.05 |
| PQN-R | 0.176 | 0.508 | ε = 0.05 |

- Greedy results are reported as secondary.

**Recovery** (`results/summary/p5_recovery_move-analysis.md`, `figures/p5_recovery_move.png`; 4 seeds; t-CIs; readouts are 3-checkpoint means; gap = control − arm, positive = damage; delay-adjusted = control(u − 100) − arm(u)).

| Variant / arm | At removal | +10 gap (delay-adj.) | AUC gap | AUC delay-adjusted |
|---|---|---|---|---|
| PQN early [0, 100) | 4.0 vs 23.0 | +17.4* (−1.8) | +0.5 [−3.5, +4.5] | **−6.4 [−10.5, −2.4]*** (catches up faster than a delayed control) |
| PQN late [250, 350) | **3.6 vs 61.8** | +25.0* (+12.8 n.s.) | +4.6 [+2.0, +7.2]* | **+1.4 [−0.7, +3.5]** |
| PQN-R early | 4.4 vs 11.8 | +12.2* (−1.0) | +2.7 [+0.1, +5.2]* | −2.7 [−5.5, +0.2] |
| PQN-R late | **3.2 vs 26.7** | +26.4* (+15.3*) | +8.6 [+5.6, +11.6]* | **+4.3 [+1.6, +7.0]*** |

- The PPO pattern replicates in both variants:
  - early restriction = pure delay;
  - late restriction = *unlearning during the mask* (the agent collapses to ~3, the score of a noop/fire-only policy).
- **PQN:** the residual gap is consistent with lost time; the gap is closed by +100.
- **PQN-R:** a small excess beyond lost time, concentrated in the first ~100 post-removal updates; delay-adjusted +5.7 n.s. at +100 and ≈ 0 at +500. This is what motivated the surgery.

**Params × replay-buffer surgery** (PQN-R, at u350).
- Run: `runs/20261011-p5-surgery-late-pqnr-SpaceInvaders-MinAtar-77afd57.json`, `figures/p5_surgery_pqnr.png`.
- **Saved state** (`results/<stem>/states_removal.pkl`, git-ignored), for C and D: params, RAdam state, buffer (obs / actions / rewards / dones / pointer / size), env state, obs, RNG, update counter.
- **Prespecified assignment rule:**
  - network carrier = params + optimizer state, together;
  - buffer carrier = all buffer arrays; pointer and size are identical in C and D;
  - env state, obs, RNG, update counter (hence the ε schedule, already at its floor) and deficit bounds come from C;
  - no target network exists.
- **Checks:** the C/D forks reproduce the recovery-run control/late eval curves exactly. 100% of D's buffer was written under the restriction at removal.

| Contrast | AUC +10..+650 | +10 | +30 | +100 |
|---|---|---|---|---|
| Total damage DD − CC | **−7.0 [−9.4, −4.7]*** | −20.8* | −16.6* | −13.3* |
| **Parameter transfer DC − CC** | **−5.0 [−8.9, −1.1]*** | −20.5* | −13.0* | −10.6* |
| **Data transfer CD − CC** | **−0.1 [−1.7, +1.6]** | −5.4 n.s. | −4.1 n.s. | +6.9 n.s. |
| Network main effect | −6.0 [−7.7, −4.2]* (85% of total) | | | |
| Buffer main effect | −1.1 [−3.6, +1.4] (15%, n.s.) | | | |
| Interaction | −1.9 n.s. | | | |
| Rescue by clean network (CD − DD) | +6.9 [+4.7, +9.2]* | | | |
| Rescue by clean buffer (DC − DD) | +2.0 n.s. | | | |

- TD loss over the first 30 updates:

| Arm | CC | DC | CD | DD |
|---|---|---|---|---|
| TD loss | 0.156 | **0.541** | 0.132 | 0.226 |

  The damaged network's Q-values are miscalibrated for the clean policy's data.
- Greedy eval (secondary, very noisy): parameter transfer −13.0 [−31.8, +5.8]; data transfer −8.4 [−23.3, +6.4]; network 62% of total. Same ordering, CIs include 0.

**Answers.**
- *Parameter transfer:* yes. The damaged network impairs recovery even with clean replay, early and over the full horizon.
- *Data transfer:* no detectable effect. A damaged buffer paired with clean parameters does not impair recovery (AUC −0.1 [−1.7, +1.6]); at most a transient, n.s. dip in the first ~30 updates.
- Main caveat: with 4096 new transitions per update, a 100k buffer is fully replaced in 24 updates. The buffer can only act briefly, so this design cannot detect a data carrier that needs a long-lived buffer. A larger buffer or lower collection throughput (e.g. 1M, or fewer envs) is the obvious stronger test.
- Across PPO (Phase 3/4), PQN and PQN-R, the late-restriction damage lives in the network.

**Process notes.**
- The sandbox restarted once while waiting on a user question. The venv and `.git/config` (the remote) were recreated, and package versions match the earlier manifests exactly.
- The first fire launch was OOM-killed, as noted above.

## 2026-10-11 — Phase 4 replication on Asterix: aborted (window not "late" on Asterix)

- Run `runs/20261011-p4-phase4-asterix-move-late-ppo-Asterix-MinAtar-208a6f2.json` (allowed {noop, up, down}, window [100, 150)) was stopped after the forks and one arm (partial summary with forks + C_cont committed for transparency).
- On Asterix, PPO at update 100 scores only ~2 (Phase 1: ~4 at 2M steps, still climbing). The restriction caused no collapse: D 1.8 vs C 2.2 at removal, so there is nothing to ablate.
- PPO's u100 is "late" on SpaceInvaders (40% of final competence) but not on Asterix.
- A meaning-matched Asterix replication needs a long control to locate the 40%-of-final point (likely ≫ 2M steps). Alternatively, use PQN, which learns Asterix faster (purejaxql default env). Pending user decision.

## 2026-10-12 — Phase 6: seed extension, 1M-buffer persistence test, Asterix onset pilot

User directives (2026-10-11): (1) more seeds on the key PQN-R surgery contrasts before expanding mechanistic claims, prioritised over new ablation arms; (2) a stronger replay-persistence test only if the data-carrier claim is central (judged central: it is the "data scar" in the title) — 1M buffer, same transfer factorial, measure how fast the transferred buffer is overwritten; (3) locate a meaning-matched Asterix onset before any surgery there.

**Code (c75175b, bit-identical to 5a915ec on a PQN-R check run).**
- Replay slots carry an int16 write-time tag (update index). Diagnostics per update, for a runner-state window [l0, l1) (set to the restriction window in surgery runs): `buf_frac_window` (share of valid slots written in the window) and `sample_frac_window` (share of sampled replay transitions from it). Learning is unaffected.
- Actions stored as uint8; the jitted run fn donates its state (surgery arms run on private copies, so C/D are never invalidated); `--buffer-size`; 1M runs do not pickle buffers (snapshot cap).
- `analyze_pqn_surgery.py` pools seed groups and adds a seed-robustness block (between-seed SD, #seeds < 0, 95% prediction interval for a new seed, mean ± t·sd·√(1+1/n)) and an overwrite block. `scripts/onset_rule.py` applies the Phase 5 onset rule to pooled controls (reproduces PQN-R u250 / 46% and PPO SpaceInvaders u100 / 40%; "final" = mean of checkpoints in the last 50 updates). `scripts/surgery_phase_contrasts.py` splits contrasts by overwrite phase.
- Bug fixed during the 1M queue: the surgery consistency check kept only the last recovery summary per arm, so seed 0 of the 1M run was not checked online; all 8 C forks were then verified post hoc (identical to their 1M control runs).

### 6a. PQN-R 100k surgery at n = 12 (seeds 0–11; same window u250–350, recovery 650; ε = 0.05 primary)
Summaries: `…p5-surgery-late-pqnr-…-77afd57.json` (0–3), `…surgery-late-s4to7-…-c75175b.json`, `…surgery-late-s8to11-…-28909a2.json`. Analysis: `results/summary/p6_surgery_pqnr_n12-analysis.md` (greedy secondary: `p6_surgery_pqnr_n12_greedy-analysis.md`), `figures/p6_surgery_pqnr_n12.png`.

| AUC contrast | n = 4 | n = 12 | seeds < 0 | between-seed SD | 95% PI (new seed) |
|---|---|---|---|---|---|
| total DD − CC | −7.0* | −6.7 [−8.0, −5.3]* | 12/12 | 2.1 | [−11.4, −1.9] |
| parameter transfer DC − CC | −5.0* | −4.6 [−6.1, −3.0]* | 12/12 | 2.5 | [−10.2, +1.1] |
| data transfer CD − CC | −0.1 | +0.0 [−1.0, +1.0] | 6/12 | 1.6 | [−3.7, +3.8] |
| network main effect | −6.0* | −5.6 [−6.3, −4.9]* | 12/12 | 1.1 | [−8.1, −3.1] |
| buffer main effect | −1.1 | −1.0 [−2.0, −0.1]* | 10/12 | 1.5 | [−4.5, +2.4] |
| interaction | −1.9 | −2.1 [−4.5, +0.2] | 8/12 | 3.7 | [−10.6, +6.3] |

- Network main effect = 84% of total damage. Data transfer is a short transient only (+20: −3.9 [−7.5, −0.3]*; +10: −4.5, CI touches 0), zero over the horizon.
- Overwrite (tagged seeds 4–11): restriction-era data < 50% of the buffer by +13, < 10% by +22, gone at +25 updates; mean sampled share over the AUC window 1.8%.
- Greedy (secondary): parameter −11.0*, data −3.2 [−8.7, +2.4], network main −11.4*.
- With 12 seeds the n = 4 conclusions hold; the small buffer main effect becomes significant (−1.0) but its prediction interval spans 0.

### 6b. PQN-R 1M-buffer persistence test
**Controls** (8 seeds, groups of 2; `…p5-recovery-pqnr-1000k-control-s{01,23,45,67}-…json`; `results/summary/p6_pqnr1m_control-onset.md`).
- Metric rule: ε = 0.05 again (noise 0.178 vs greedy 0.597).
- Onset rule: 40%-of-final reached at u170 (per seed 60–180); the ε-floor condition binds → **onset u250** (control there 24.7 = 51% of final 48.3). Same window as 100k: [250, 350), recovery 650.
- The 1M control learns slower (final 48.3 vs 58.3) and is still climbing at u1000 (+7.4 over the last 100 updates).

**Surgery** (8 seeds, one per process for memory; `…p5-surgery-late-s{0..7}-pqnr-1000k-…json`; `results/summary/p6_surgery_pqnr1m_n8-analysis.md`, `…-phases.md`, `figures/p6_surgery_pqnr1m_n8.png`).
- Composition at removal (all seeds): D buffer full, 41.0% restricted-era data (newest); 59% pre-onset data shared with C. Restricted data stays at 41% until +150, drains over +160..+240, gone at +245; mean sampled share over the AUC window 12.2% (vs 1.8% at 100k).
- **The deficit itself is milder with 1M:** at removal D 14.1 vs C 27.8 (−50%; 100k: 4.2 vs 32.0, −87%). The persisting pre-onset data (including left/right actions) dilutes the restriction. So the 1M test changes severity as well as persistence.

| AUC contrast (n = 8) | 1M | seeds < 0 | 95% PI | 100k (n = 12) |
|---|---|---|---|---|
| total DD − CC | −5.6 [−7.0, −4.2]* | 8/8 | [−9.9, −1.3] | −6.7* |
| parameter transfer DC − CC | −2.6 [−3.6, −1.6]* | 8/8 | [−5.6, +0.4] | −4.6* |
| data transfer CD − CC | **−1.7 [−3.3, −0.2]*** | 6/8 | [−6.5, +3.0] | +0.0 |
| network main effect | −3.2 [−4.2, −2.2]* | 8/8 | [−6.2, −0.2] | −5.6* |
| buffer main effect | **−2.4 [−3.4, −1.4]*** | 8/8 | [−5.5, +0.7] | −1.0* |
| interaction | −1.3 [−3.2, +0.6] | 5/8 | [−7.1, +4.5] | −2.1 |

- Share of total damage: network 57%, buffer 43% (100k: 84% / 16%).
- Early readouts: data transfer ≈ 0 at +10/+20/+30 (+0.1, −0.7, +0.2) — no immediate effect of handing a clean network a 41%-restricted buffer.
- **By overwrite phase** (`p6_surgery_pqnr1m_n8-phases.md`): data transfer −1.2 [−2.8, +0.5] while carried (+10..+150), −2.5 [−5.0, +0.0] while draining (+160..+240), −1.8 [−3.9, +0.3] after it is gone (+250..+650); buffer main effect −1.4*, −1.7, −2.9*. The data effect does **not** track the presence of the restricted data: it accrues while the clean network trains on it and then persists after the data is overwritten.
- TD loss over the first 30 updates: CC 0.074, DC 0.690, CD 0.058, DD 0.858.

**Answer to the persistence question.** When the transferred buffer persists (~245 updates instead of 25), a data carrier appears: small, cumulative, and significant (data transfer −1.7*, buffer main effect −2.4*, 8/8 seeds), but a new seed could plausibly show none (PIs span 0). It is not an immediate "poisoned buffer" effect; the harm is consistent with the restricted data being written into a clean network during training and persisting as network damage once the data is gone. The network remains the larger carrier at both buffer sizes. Caveats: n = 8; the 1M deficit is milder (so absolute effects are not directly comparable to 100k); one buffer size beyond 100k; same game and restriction.

### 6c. PPO Asterix long control (onset pilot only; no surgery)
Run `runs/20261012-p2-asterix-long-control-ppo-Asterix-MinAtar-f12598d.json` (1220 updates = 10M steps, 4 seeds); `results/summary/p6_asterix_ppo_long_control-onset.md`.
- Eval rises almost linearly: u100 1.8, u200 2.8, u300 4.4, u400 5.9, u600 8.1, u800 10.4, u1000 12.8, u1100 14.0, u1220 13.6.
- Final (last 50 updates) 14.0 (per-seed sd 3.5); 40% = 5.6; mean crosses at u390 (per seed 270–470) → **meaning-matched onset u390** (43% of final).
- Plateau is weak: only the last ~120 updates are flat (−0.18, −0.25 per 100). If the true asymptote is higher, the 40% point moves later (e.g. final 18 → ~u550). The onset is provisional.
- Proposed PPO Asterix replication (not run): restriction [390, 440) (50 updates = 409,600 steps), recovery 290 updates (u730; within the observed control) — or a longer control first to confirm the plateau.

## 2026-10-12 — Phase 6 (cont.): replay-purge experiment — PRE-REGISTRATION (written and committed before any real run)

**Question (user):** does replay data leave a scar after it is removed? Setting: SpaceInvaders-MinAtar movement restriction ({noop, fire}), PQN-R, 1M buffer, late window [250, 350) (prespecified onset rule, unchanged), recovery 650 updates, seeds 0–7 (same as the Phase 6 1M surgery), ε = 0.05 deficit-free eval primary (greedy secondary). Script `scripts/pqn_purge.py`, analysis `scripts/analyze_pqn_purge.py`; one seed per process (memory).

**Arms at removal (u350):**
- **A_retain:** damaged agent continues with its own 1M buffer (restriction-era data, 41% at removal, overwritten by normal collection by ~+245).
- **B_purge:** damaged agent; the entire buffer is replaced at u350 by the replacement buffer.
- **C_delay50:** damaged agent keeps its buffer for 50 updates (bit-identical to A up to u400), then the entire buffer is replaced by the same replacement buffer.
- **D_clean:** clean agent continues with its own buffer (= the 1M control run; checked for bit-identity against the control summaries).
- **E_clean_purge (secondary, added by me):** clean agent with the same purge at u350. B − A mixes "remove restriction-era data" with "replace a mixed-age buffer by fresh clean-actor data"; E − D measures the latter on a clean network, and (B − A) − (E − D) isolates the damage-specific purge benefit.
- Non-buffer state (params, RAdam state, env state, current obs, RNG, update counter) is the arm's own fork state (A/B/C from the restricted fork, D/E from the clean fork).

**Replacement buffer (fixed protocol; one per seed, identical for B, C and E):** real transitions, not relabelled old data. A **frozen clean actor** (the clean fork's parameters at u350; it does not keep learning) acts ε-greedy with ε = 0.05 (training floor) over the full action set in 128 fresh unrestricted envs (same wrappers), key `fold_in(PRNGKey(seed), 2026)`, separate from every agent's env state and RNG; 7,812 × 128 transitions (fills the buffer) in the PQN-R schema, time-ordered, ptr = 0, size = full (FIFO overwrites the oldest replacement data first); slot tags = purge update. Splice: after any swap 128 of 1M transitions (0.013%) have the agent's next obs as successor (as in all surgery arms).

**Primary estimands** (paired by seed, 95% t-CI; Holm correction across the three primary AUC tests):
1. Purge rescue **B − A**. 2. Delayed-purge effect **B − C**. 3. Residual impairment after purge **B − D**.

**Windows** (post-removal updates; AUC as in Phase 5 = mean over +10..+650): early +10..+150 (A's restricted share flat), mid +160..+240 (draining), late +250..+650 (A's restricted data gone), post-delay +60..+150; readouts +10, +100 (3-checkpoint means). Restriction-era share of every buffer logged at every eval checkpoint.

**Secondary:** E − D; (B − A) − (E − D); C − A; A − D. Seed-robustness block (SD, #seeds < 0, prediction interval) for all.

**Interpretation rule (user):** most diagnostic = B > A, C between them, B < D → old data actively harms recovery while consumed, and impairment remains after removal. B ≈ A weakens the active replay-data claim (does not affect the network-transfer result). E − D ≠ 0 means part of B − A is a replacement-data-quality effect, read through (B − A) − (E − D).

**Expectation (from the Phase 6 phase analysis; not a hypothesis test):** small B − A (≈ +1–2 AUC, possibly n.s. at n = 8), clearly negative B − D (network damage).

## 2026-10-12 — Phase 6 replay purge: RESULTS (n = 8, analysis exactly as pre-registered in 864697e)

Summaries `results/summary/20261012-p6-purge-s{0..7}-pqnr-1000k-SpaceInvaders-MinAtar-864697e.json`; analysis `results/summary/p6_purge-analysis.md`; post-hoc `results/summary/p6_purge-posthoc.md`; figure `figures/p6_purge.png`.
- Checks: D_clean bit-identical to the 1M control run and both forks bit-identical to the Phase 6 1M surgery, all 8 seeds. Replacement-buffer collection returns 26.3–30.4 per episode (frozen clean actor, ε 0.05).

| arm | AUC | early +10..150 | mid +160..240 | late +250..650 | +10 |
|---|---|---|---|---|---|
| A_retain | 33.3 | 24.3 | 27.8 | 37.8 | 18.0 |
| B_purge | 37.8 | 24.2 | 31.2 | 44.3 | 9.5 |
| C_delay50 | 36.8 | 24.3 | 29.3 | 43.0 | 18.0 |
| D_clean | 37.9 | 28.0 | 30.6 | 43.2 | 27.3 |
| E_clean_purge | 39.7 | 27.1 | 33.5 | 45.7 | 26.0 |

**Primary (AUC; Holm across the three):**
- **Purge rescue B − A: +4.6 [+2.2, +6.9]*** (p = 0.002, Holm 0.007; 0/8 seeds negative; PI [−2.4, +11.5]). Early −0.1 (an immediate purge shock, +10 −8.5*, cancels later gains), mid +3.3*, **late +6.5 [+3.1, +10.0]***.
- **Delayed-purge effect B − C: +1.0 [−0.1, +2.1]** (p = 0.075, Holm 0.149; n.s.). Post-delay window +2.1*. C − A = +3.5* (secondary): purging after 50 updates recovers ~78% of the purge benefit.
- **Residual impairment B − D: −0.1 [−2.5, +2.3]** (Holm 0.906). Early −3.8*, +100 −1.5*, late +1.1 (n.s.).

**Secondary:** E − D +1.8 [−1.2, +4.8] (replacement data mildly better than the clean agent's own mixed-age buffer, n.s.; one seed +9.3). Damage-specific purge benefit (B − A) − (E − D) +2.8 [−0.2, +5.7] (late +4.0 [−0.0, +8.1], +100 +4.7*). A − D −4.7* (8/8; late −5.5*).

**Post-hoc (labelled, not pre-registered):** B − E (damaged vs clean network on the identical buffer) −1.9 [−3.0, −0.8]* AUC (early −2.9*, late −1.4 n.s.), matching the surgery's parameter transfer (−2.6*). Purge shock: B − A at +10 = −14.8* (single checkpoint), gone by +20; C's purge at u400 causes no significant dip (+60/+70: −1.1, −3.0 n.s.), nor does E's purge of a clean agent (−2.0 n.s.). TD loss (first 30 updates): A 0.885, B 0.696, D 0.074, E 0.121.

**Reading against the pre-registered rule.** The pattern holds descriptively: B > A (significant), A ≤ C ≤ B (C close to B; B − C n.s.), B < D only by −0.1 (n.s.).
1. **Replay data leaves a scar after it is removed.** In A the restriction-era data is completely overwritten by +245, yet over +250..+650 A trails B by +6.5* and D by −5.5*. Up to ~40% of B − A could be replacement-data quality (E − D +1.8, n.s.); the damage-specific part is +2.8 AUC (late +4.0), borderline at n = 8. This is the persistent data→network effect seen in the Phase 6 phase analysis, now shown causally by removing the data.
2. **Most of the harm comes from consuming the old data after the first 50 updates:** delaying the purge by 50 updates costs only +1.0 (n.s.), while letting the data drain naturally (~245 updates) costs +4.6.
3. **No residual impairment after an immediate purge on AUC (B ≈ D).** This is not "no network damage": on the identical buffer the damaged network trails the clean one (B − E −1.9*). B matches D because the fresh replacement data is slightly better than D's own buffer.
4. **Purge shock:** swapping the whole buffer right at removal makes the damaged agent briefly much worse (+10: −14.8*). The same swap 50 updates later, or applied to a clean agent, does not. The just-damaged Q-function is fragile to an abrupt data shift.

Caveats: one game and restriction; n = 8; the replacement policy is a frozen clean actor (a learning or self-collected replacement was not tested); E − D shows the replacement data is not neutral.

## 2026-10-12 — Phase 6b: targeted vs sham replay purge — PRE-REGISTRATION (committed before any real run)

**Question (user):** does replacing *restriction-era* transitions matter beyond refreshing the buffer? SpaceInvaders-MinAtar, movement restriction, PQN-R, 1M buffer, late window [250, 350), removal u350, recovery 650, ε = 0.05 eval primary (greedy secondary), **12 paired seeds (0–11)**, one per process. Script `scripts/pqn_targeted_purge.py`; analysis `scripts/analyze_pqn_targeted.py`.

**Common start state:** every arm starts from the restricted fork's saved removal state (params, RAdam state, env state, current obs, RNG, update counter, buffer). Training schedule unchanged.
- **A_retain:** no change.
- **B_targeted:** the restriction-era rows (exactly the newest K = 3,200 rows × 128 envs = 409,600 transitions = 41%; asserted from write-time tags) are overwritten with the replacement pool; all other rows preserved.
- **C_sham:** the same number of non-restriction-era rows, chosen as the closest in age — the adjacent older block (rows written u150–249) — overwritten with the same pool; restriction-era rows intact.
- **D_cleannet:** B's exact buffer and start state, but params + optimizer state from the clean fork.

**Replacement pool:** frozen clean actor (clean fork params at u350), ε = 0.05, full action set, 128 fresh unrestricted envs, key `fold_in(PRNGKey(seed), 2026)`, 3,200 time-ordered steps per env (verified identical to the first 3,200 steps of the Phase 6 purge pool). Collected once per seed and frozen before any learner arm branches; identical in B, C, D. Buffer size, pointer, size and schema unchanged; replaced slots tagged 32000 (diagnostic only).

**Successor continuity:** s' is the next row of the same env, so each spliced block creates 2 invalid successor links per env (row before the block; the block's last row). New per-slot flag `valid_next` (`SEAM_MASK`, used by all four arms); flagged transitions (2 × 128 = 256 per buffer, 0.026%) get zero loss weight until overwritten. Verified: flag off is bit-identical to HEAD; flag on with no flags is bit-identical to flag off (so A must reproduce Phase 6 A_retain for seeds 0–7); a unit test shows flagged slots do not affect learning.

**Age-matching limit (stated before the run):** the restriction-era rows are the newest rows, so no non-restriction-era row has the same age. The sham uses the nearest feasible block (rows aged 100–199 updates at removal vs 0–99). Under FIFO, the sham's replacement data is overwritten ~100 updates earlier (+44..+144 vs +144..+244), so B carries replacement data for longer. Mean replacement share over the AUC window is reported per arm. A positive B − C therefore bundles "removing restricted data" with "replacement data persisting ~100 updates longer"; B − C ≈ 0 would be informative despite the mismatch.

**Primary contrasts** (AUC +10..+650; paired; 95% t-CI; **Holm across the three**; seed-level differences reported; exact sign-flip p as a robustness check):
1. **B − C** targeted vs sham (key test). 2. **B − A** targeted purge vs retain. 3. **B − D** residual network impairment under matched data exposure.
**Secondary endpoints:** early +10..+150, middle +160..+240, late +250..+650. Secondary contrast: C − A. Descriptive: (B − C)/(B − A).

### Phase 6b — addendum (user note), committed before any Phase 6b result was viewed

- **Interpretation constraint (user):** B and C replace equal numbers of slots with the same pool, but the sham's replacement data is overwritten ~100 updates earlier. A positive B − C may reflect (1) removal of restriction-era transitions, (2) longer residence of the replacement data, or (3) both, and will not be attributed entirely to the identity of the removed transitions. B − C ≈ 0 would weaken the case that targeted removal matters beyond refreshing the buffer, but because of the residence mismatch it would not establish equivalence.
- **Pre-registered primary contrasts unchanged.** Added descriptive reporting to `analyze_pqn_targeted.py`: replacement residence curves and the expected sampled replacement share per checkpoint for B, C and D; cumulative sampled replacement transitions (B vs C ratio); and B − C over the exposure-matched period. Until the sham's block starts being overwritten (~+44), both arms have had identical replacement exposure.
- **If B − C > 0:** the next step is an exposure-matched follow-up (controlling the replacement pool's residence time or sampling contribution) before any strong data-identity claim.
- **Process:** the sandbox restarted during the queue. Seeds 0–4 completed (all checks True; summaries committed here unviewed); seed 5 was interrupted after 3 of 4 arms. The partial seed-5 summary was lost while being set aside (same filename as its manifest), so seed 5 is rerun from scratch at the same commit; the deterministic rerun is still checked against the Phase 6 purge A_retain and the Phase 6 surgery forks. Seeds 5–11 relaunched.

## 2026-10-12 — Phase 6b targeted vs sham purge: RESULTS (n = 12; pre-registered analysis + pre-results addendum)

Summaries `results/summary/20261012-p6b-tpurge-s{0..11}-pqnr-1000k-…json` (seeds 0–4 at 250cec4, 5–11 at 1c06f18; experiment code identical); analysis `results/summary/p6b_targeted-analysis.md`; figure `figures/p6b_targeted.png`.
- Checks: A_retain bit-identical to the Phase 6 purge A_retain and both forks bit-identical to the Phase 6 surgery for seeds 0–7 (no earlier runs exist for seeds 8–11). Pool collection mean return 24.7–30.1. Seam flags 256 per spliced buffer.

| arm | AUC | early +10..150 | middle +160..240 | late +250..650 |
|---|---|---|---|---|
| A_retain | 33.1 | 24.2 | 27.6 | 37.6 |
| B_targeted | 36.7 | 25.2 | 29.7 | 42.4 |
| C_sham | 34.1 | 23.6 | 27.9 | 39.2 |
| D_cleannet | 39.5 | 28.3 | 32.5 | 45.2 |

**Primary (AUC; Holm across three; exact sign-flip p in brackets):**
- **Targeted vs sham B − C: +2.6 [+0.9, +4.3]*, Holm p = 0.006** (sign-flip 0.009; 9/12 seeds > 0; SD 2.7). Early +1.5*, middle +1.9*, late +3.2*.
- **Targeted vs retain B − A: +3.5 [+1.9, +5.2]*, Holm p = 0.001** (10/12). Early +1.0*, middle +2.2*, late +4.8*.
- **Residual network impairment B − D: −2.9 [−4.0, −1.8]*, Holm p = 0.0004** (11/12 negative). Uniform across windows (−3.2*, −2.8*, −2.8*).
- Secondary: sham refresh vs retain C − A +0.9 [−0.8, +2.7] (n.s.). Descriptive: (B − C)/(B − A) = 73%.

**Exposure (addendum, committed before results were viewed):**
- Replacement share: B (= D) 0.41 until +140, drained by +250; C 0.41 until +40, drained by +150. Mean sampled replacement share over the AUC window B 0.119 vs C 0.056; cumulative sampled replacement transitions 0.65M vs 0.31M (ratio 2.07).
- Restricted-era share in C follows A exactly (0.41 until +140, 0 by +250); B and D 0.
- **Exposure-matched period** (+10..+40, before the sham block starts being overwritten at ~+44; identical replacement exposure so far): **B − C = +2.5 [+0.2, +4.8]*** (9/12). Here the arms differ only in which 41% of the old buffer was retained (B: pre-onset data from u150–249; C: restriction-era data). Descriptive and short (4 checkpoints).

**Reading (under the pre-registered constraint).**
- Targeted removal beats the matched sham, so replacing restriction-era transitions matters beyond refreshing the buffer. But per the user's note, the full B − C is **not** attributed entirely to the identity of the removed data: from +44 on, B also holds the replacement data ~100 updates longer (2.1× the sampled exposure).
- Evidence that identity contributes: the gap already exists while exposure is identical (+10..+40, +2.5*), and the sham refresh alone does not help (C − A +0.9 n.s.).
- Residence could still contribute to the middle/late gap. The size of the two components is not identified by this design.
- The damaged network still trails the clean one under exactly matched data (B − D −2.9*), consistent with Phase 6 (B − E −1.9*) and the 1M surgery (−2.6*). Both carriers exist: restriction-era replay data and the network.

**Planned follow-up (per the user's rule for B − C > 0), not yet run:** an identity × residence 2×2 that matches replacement exposure. Add C′ = remove the u150–249 block but place the replacement pool in the newest rows (moving the restriction-era block into the older position), and B′ = remove the restriction-era block but place the replacement in the older position (moving the u150–249 block into the newest rows). Then B vs C′ differ only in data identity at matched residence, and B vs B′ isolate the residence effect. Seams at moved-block boundaries are handled by the same `valid_next` flags; pre-register before running.

## 2026-10-12 — Phase 6c identity x residence 2x2: PRE-REGISTRATION (before any real run)

**Why.** Phase 6b B − C (+2.6*) mixes data identity with replacement residence: the sham's replacement data sits in older rows and is overwritten ~100 updates earlier (2.07x the sampled exposure in B). Following the user's rule (if B − C > 0, match exposure before a strong identity claim), the design is completed into a 2x2 by repositioning blocks.

**Design.** The removal state (restricted fork D at u350), the frozen replacement pool, evaluation, seeds 0–11 and all training settings are identical to Phase 6b (same code path and keys). Buffer positions at removal, oldest → newest: [rest][P2: rows written u150–249][P1: rows written u250–349]. R = restriction-era data (u250–349), P = pre-onset data (u150–249).

| cell | removed | replacement position (residence) | P2 / P1 contents | source |
|---|---|---|---|---|
| B_targeted | R | P1, newest (long, overwritten +144..+244) | P / pool | Phase 6b (reused) |
| Bp_targeted_short | R | P2 (short, +44..+144) | pool / P | NEW |
| Cp_sham_long | P | P1 (long) | R / pool | NEW |
| C_sham | P | P2 (short) | pool / R | Phase 6b (reused) |

- B vs Cp: same replacement position and same retained-block position. They differ only in whether the retained 41% block is restriction-era or pre-onset data. This is the residence-matched identity test. It is conservative relative to Phase 6b, because in Cp the restriction-era data sits in the older rows and drains by +144 (in C and A it lasts to +244).
- Bp vs C: the same identity test at short replacement residence.
- Moved blocks keep their per-env time order and every field (obs, action, reward, done, tag, valid_next). ptr, size, buffer length, params, optimizer state, env state, RNG and update counter are those of the fork.
- Seams: the new cells have 3 invalid successor links per env, flagged valid_next = False with zero loss weight until overwritten. They are the row before P2, the last row of P2, and the last (newest) row of P1, whose successor is patched to the current env obs. B and C (Phase 6b) have 2 per env. The difference is 128 transitions per buffer (0.013%), disclosed.
- Script: `scripts/pqn_residence_2x2.py`. It runs Bp and Cp per seed. Seed 0 also reruns B_targeted (`--check-b`) as a bit-identity check against Phase 6b.
- Reuse guard (required per seed; the analysis aborts otherwise): fork C/D eval traces, pool collection stats and row blocks must match the Phase 6b summary of the same seed. In-script asserts check that every moved row equals its source rows (tags and packed obs), that the rest rows are unchanged, and that the seam count is 3 × 128.
- Smoke-tested on a 100k buffer (layout asserts, determinism, consistency code). No real run had been made at the time of this entry.

**Analysis** (`scripts/analyze_pqn_2x2.py`; metric eval_eps05_return_mean; AUC over +10..+650 primary; windows early +10..150, middle +160..240, late +250..650 secondary; paired 95% t-CIs; n = 12).
- **Primary (Holm across three AUC tests):**
  1. Identity at matched long residence: B − Cp (key test).
  2. Identity at matched short residence: Bp − C.
  3. Residence main effect: ((B − Bp) + (Cp − C)) / 2.
- **Secondary (descriptive):** identity main effect ((B − Cp) + (Bp − C)) / 2; interaction (B − Cp) − (Bp − C), underpowered at n = 12; simple residence effects B − Bp and Cp − C; the decomposition B − C = (B − Cp) + (Cp − C); exact sign-flip p; seed-level differences. Exposure: restricted-era and replacement shares per checkpoint; cumulative expected sampled restricted-era and replacement transitions per cell; a check that the residence-matched pairs have identical replacement curves.
- **Interpretation rules, fixed now:**
  - B − Cp > 0 (Holm) → removing restriction-era data helps beyond replacement exposure, which supports data identity.
  - B − Cp ≈ 0 with residence main effect > 0 → the Phase 6b gap is better explained by residence. This weakens the identity claim but does not prove no identity effect: Cp's restricted data is resident for less time than in A/C.
  - Residence main effect is reported either way. A, D are drawn for reference only.
- Budget: ~6.5 min per seed (2 arms), seed 0 ~9 min; ~80 min total, sequential, 1 seed per process.

## 2026-10-12 — Phase 6c identity x residence 2x2: RESULTS (n = 12; pre-registered analysis)

Summaries `results/summary/20261012-p6c-res2x2-s{0..11}-pqnr-1000k-…json`. Filenames carry launch commits 2559f2e / eb35453 / 3ddb28f / 5ca0172, because seed summaries were committed while the queue ran (to protect against restarts). Those intermediate commits contain result files only: `git diff 2559f2e 5ca0172 -- src scripts` is empty. Analysis `results/summary/p6c_2x2-analysis.md`; figure `figures/p6c_2x2.png`. The middle panel was changed, before writing this entry, from residence curves (which coincide by design and hid each other) to a cumulative-exposure map. Display only; no numbers changed.
- Reuse guard: for all 12 seeds, the forks, pool stats and row blocks match Phase 6b. Seed 0 reruns B_targeted bit-identically. In-script layout asserts pass. Replacement curves of the residence-matched pairs are identical (max difference 0.0000). Seams: 384 per new-cell buffer.

| cell | removed | replacement | AUC | early | middle | late |
|---|---|---|---|---|---|---|
| B_targeted | R | long | 36.7 | 25.2 | 29.7 | 42.4 |
| Bp_targeted_short | R | short | 35.2 | 25.9 | 28.7 | 40.0 |
| Cp_sham_long | P | long | 35.5 | 24.5 | 29.0 | 41.0 |
| C_sham | P | short | 34.1 | 23.6 | 27.9 | 39.2 |
| (A_retain, ref) | — | — | 33.1 | 24.2 | 27.6 | 37.6 |
| (D_cleannet, ref) | R | long | 39.5 | 28.3 | 32.5 | 45.2 |

**Primary (Holm across three): none significant.**
- Identity at matched long residence, B − Cp: **+1.1 [−0.2, +2.5]**, p = 0.090, Holm 0.180 (sign-flip 0.097; 8/12). Early window +0.7*, middle +0.7, late +1.4.
- Identity at matched short residence, Bp − C: **+1.1 [−0.8, +3.0]**, p = 0.226, Holm 0.226 (8/12). Early +2.2*, middle +0.9, late +0.7.
- Residence main effect: **+1.5 [+0.2, +2.7]**, p = 0.024, Holm 0.073 (sign-flip 0.027; 8/12). Early +0.0, middle +1.1*, late +2.1*.

**Secondary (descriptive):**
- Identity main effect +1.1 [+0.0, +2.2] (sign-flip 0.048; 9/12).
- Interaction +0.0 [−2.3, +2.4] (early −1.5*).
- Residence given R removed (B − Bp) +1.5 [−0.1, +3.1]; given P removed (Cp − C) +1.5 [−0.4, +3.3].
- **Decomposition of Phase 6b B − C (+2.6*) = identity at long residence (+1.1) + residence given P removed (+1.5).**
- Exposure (cumulative expected sampled, M): B 0 restricted / 0.65 replacement; Bp 0 / 0.31; Cp 0.31 / 0.65; C 0.65 / 0.31.

**Reading (against the pre-registered rules).**
- Neither pre-registered outcome is clearly met. B − Cp is not significant, but the residence main effect also misses Holm (0.073).
- The point estimates split the Phase 6b gap about evenly: roughly 40% identity and 60% residence (+1.1 vs +1.5). With 12 seeds, neither component on its own is established.
- **Consequences for the claim:**
  - The Phase 6b result does not support a strong data-identity claim: a good part of B − C is plausibly explained by the replacement data staying in the buffer longer.
  - What survives:
    - In the early window, removing restriction-era data helps at both residence levels (+0.7*, +2.2*). This matches the exposure-matched window in Phase 6b (+2.5*).
    - The pooled identity main effect is borderline (+1.1, CI lower bound 0.0).
  - Supported wording: "the restriction-era data carries a modest, mainly early cost; how long the replacement data stays contributes comparably to later recovery".
  - Not supported: "the damage lives in the restricted data" as the explanation for the full targeted-purge benefit.
- Unchanged: the network carrier. B − D −2.9* (Phase 6b) is not affected by this design.
- Power note (descriptive): with SD ≈ 2.1 for B − Cp, detecting +1.1 at 80% power (two-sided, α = 0.05) would need about 30 seeds.

## 2026-10-12 — POST-HOC (user request): paired 2x2 factorial model of Phase 6c

Script `scripts/posthoc_2x2_factorial.py`; output `results/summary/p6c_2x2-factorial-posthoc.md`; figure `figures/p6c_2x2_factorial.png`. This is not pre-registered: only the interaction was pre-registered, as secondary and descriptive. The p-values are unadjusted and descriptive, and the Phase 6c pre-registered conclusions are unchanged.

- **Model 1:** two-way repeated-measures ANOVA, seeds as blocks, one error term per effect; F(1,11) = t² of the per-seed contrast; effect coding ±½. Seed-block SS 101.6 of 257.9 total.
  - Identity main effect: **+1.11 [+0.01, +2.21]**, F = 4.91, p = 0.049, partial η² = 0.31.
  - Residence main effect: **+1.49 [+0.23, +2.74]**, F = 6.82, p = 0.024, partial η² = 0.38.
  - **Interaction (B − Bp) − (Cp − C) = +0.04 [−2.35, +2.43]**, F = 0.00, p = 0.97, partial η² = 0.00.
- **Model 2 (pooled residual, df 33; equals a random-intercept mixed model here):**
  - identity +1.11 [+0.01, +2.21], p = 0.047;
  - residence +1.49 [+0.39, +2.58], p = 0.009;
  - interaction +0.04 [−2.15, +2.23], p = 0.97.
  - Per-effect error MS 3.02 / 3.90 / 3.53, so the pooling assumption is reasonable.
- **Windows:**
  - Identity is concentrated early: +1.48 [+0.79, +2.16], flip p 0.001.
  - Residence acts mid (+1.08*) and late (+2.11*), not early (+0.05).
  - The interaction is significant only early, at −1.53 [−2.50, −0.56]: early on, the identity effect is larger when the replacement is short-resident, i.e. when the swapped blocks sit in the newest rows.
- **Exact sign-flip (4096 assignments):**
  - identity p 0.048; residence p 0.027; interaction p 0.975 (3992/4096 assignments at least as extreme).
  - The studentized-t version gives the same p-values.
  - Exact Wilcoxon: identity 0.064, residence 0.043, interaction 0.52.
  - **Leave-one-seed-out:** identity +0.82..+1.31 (flip p 0.023–0.096; dropping seed 1 gives 0.096); residence +1.20..+1.81 (0.009–0.054); interaction −0.40..+0.94 (dropping seed 3, at −9.88, gives +0.94, p 0.18).
  - **Conclusion:** the main effects are borderline-to-modest and sensitive to single seeds. No interaction at any LOSO cut.
- **Exposure (M sampled transitions, +1..650):** sampled restriction-era exposure is realised (logged); replacement and pre-onset exposure are expected from exact per-update buffer shares.
  - B: R 0 / pool 0.650 / P 0.314.
  - Bp: R 0 / pool 0.314 / P 0.650.
  - Cp: R 0.314 (per-seed 0.313–0.315) / pool 0.650 / P 0.
  - C: R 0.650 / pool 0.314 / P 0.
  - A: R 0.650 / pool 0 / P 0.314.
  - The matched pairs have identical replacement exposure.
  - The identity contrast swaps 0.314M R ↔ P at long residence but 0.650M at short residence. The residence contrast also swaps which retained block is long-resident (pool ±0.336M against P or R).
- **Exposure-dose check (descriptive):**
  - bR = −2.04 per M restriction-era samples [−4.45, +0.36]; bpool = +3.41 per M [−0.43, +7.25] (relative to P data).
  - Lack of fit +0.61 [−1.07, +2.28], p 0.44.
  - The observed Bp − C ≈ B − Cp (+1.09 vs +1.13), despite 2.1× more R exposure swapped, gives no sign of a dose-response in restriction-era exposure, but the test is weak.
- **Reading:** the data fit an additive model, with small separate contributions from which data was removed (early) and how long the replacement stays (mid/late), and no evidence of interaction. A mild correction to the Phase 6c entry: the post-hoc main effects reach unadjusted p < 0.05, whereas the pre-registered simple contrasts (Holm) did not. They are still borderline and LOSO-sensitive, and they do not upgrade the claim.

## 2026-10-12 — Phase 6d matched-identity purge: PRE-REGISTRATION (before any real run)

**Why.** The earlier identity contrasts compared restriction-era data R with a weak control:
- Phase 6b/6c used the pre-onset block P (written u150–249). P differs from R in epoch, exploration (ε ≈ 0.42 → 0.05 vs 0.05) and policy competence, and in Phase 6c in its slot position.
- The user asked for a comparison where the arms differ only in the identity of the removed transitions. Slot count and age, replacement-pool identity and sampled exposure, retained residence, seams and learner state must all be matched.

**Matched counterpart.** U = the clean fork C's own transitions in the same rows (P1, the newest K = 3,200 rows written u250–349).
- Same update window, same ε (0.05), same 128 env instances with identical env state at u250, same on-line collection.
- U differs from R only in the absence of the restriction, and in the clean learner that the restriction-free history produced.
- Every row outside P1 is identical in C and D (shared history; asserted).

**Design: column-split purge from one common buffer.**
- **Common buffer:** per seed, the common pre-purge buffer Z = D's buffer with the P1 rows of one half of the env columns (H_U) taken from C.
  - The other half (H_R) holds R.
  - Counterbalanced by seed parity: even seeds H_R = columns 0–63, odd seeds 64–127.
  - Each column's time series is its genuine history, so Z has no internal seams.
- **Arms** (both start from the restricted fork D at removal: params, RAdam state, env state, current obs, RNG, update counter; ptr and size unchanged):
  - **purge_R:** P1 rows of H_R ← pool; U retained.
  - **purge_U:** P1 rows of H_U ← the same pool trajectories; R retained.
- **Matching:**
  - Slots: replaced slots = K × 64 = 204,800 (20.5%) in the same rows, so ages and FIFO residence are identical. They differ only in env column, which uniform replay sampling ignores.
  - Pool: the Phase 6b pool (frozen clean actor = fork C params at removal, ε 0.05, full action set, key fold_in(PRNGKey(seed), 2026)); env columns 0–63 in both arms. Identical trajectories and identical expected sampled exposure (~0.325M samples).
  - Retained transitions: identical positions, so the residence distribution is identical. The swapped content (U vs R) sits in the same rows.
  - Seams, identical in both arms: rows P2-end and P1-end are flagged valid_next = False in all 128 columns (256 slots). This covers every invalid link in either arm and discards the same valid links in both.
  - Common random numbers: the same replay (row, column) draws in both arms.
- **Diagnostics:** R keeps tags 250–349, so the per-update sampled R fraction is logged by the learner. U is tagged 20000 + update; the pool 32000. R/U/pool shares are logged at every checkpoint.
- **Asserts:** C and D are identical outside P1; P1 is exactly the restriction-era rows; Z equals D except U cells, which equal C; each arm's purged cells equal the pool and its retained cells equal Z; seam count 256; ptr unchanged.
- **Consistency:** for seeds 0–11, the fork eval traces and pool stats must match Phase 6b, or the analysis aborts.
- **Environment:** the venv was rebuilt after a sandbox restart with the same pinned versions (jax 0.11.2, flax 0.12.10, optax 0.2.8, gymnax 0.0.9, numpy 2.5.4; distrax re-added for PPO tests). 22/22 tests pass. Smoke-tested on a 100k buffer for both column parities.
- **Script and run:** `scripts/pqn_identity_swap.py`; queue `scripts/queue_p6d_idswap.sh` (resumable; one seed per process).
- **Blinding:** the run log prints no post-removal returns. The primary contrast is not computed before all seeds finish.

**Sample size, fixed in advance: n = 32 (seeds 0–31).** No interim analysis and no optional stopping. If the sandbox interrupts a seed, that seed is rerun from scratch. Planning SD of the paired difference = 2.0 (Phase 6c paired SDs were 2.1–2.9 for 41% manipulations; this one is 20.5% with common random numbers). At n = 32:
- power to detect +1.0 = 0.78 (SD 1.5: 0.95; SD 2.5: 0.59);
- TOST power at a true 0 = 0.73;
- expected 95% CI half-width ≈ 0.72.
- Budget ≈ 7.5 min per seed, about 4 h.

**PRIMARY (only confirmatory test).**
- **Contrast:** Δ = AUC(purge_R) − AUC(purge_U), paired by seed; AUC = mean ε = 0.05 eval return over checkpoints +10..+650. Positive = removing restriction-era transitions is better than removing matched same-epoch unrestricted transitions.
- **Test:** two-sided paired t, α = 0.05.
- **SESOI δ = 1.0 AUC point.** It is ≈3% of the retain-arm AUC (33.1) and ≈30% of the full targeted-purge benefit (Phase 6b B − A = +3.5). It is also about the Phase 6c identity estimates (+1.1) at a comparable swapped restricted exposure (0.31M there vs ~0.33M here). A smaller difference would not justify calling restriction-era data a damage carrier in the paper.
- **Equivalence:** TOST vs ±1.0 (90% CI).

| outcome | rule | interpretation |
|---|---|---|
| A | CI95 lower > 0 and CI90 upper > +1.0 | identity effect detected, not shown to be below the SESOI (also report whether CI95 lower > +1.0) |
| B | CI95 lower > 0 and CI90 within (−1, +1) | detected but smaller than the SESOI: negligible |
| C | CI95 includes 0 and CI90 within (−1, +1) | no meaningful identity effect (equivalence) |
| D | CI95 includes 0, CI90 not within (−1, +1) | inconclusive |
| E | CI95 upper < 0 | removing restricted data is worse |

**Secondary (descriptive only):**
- windows early +10..150, middle +160..240, late +250..650;
- greedy-eval AUC;
- Monte Carlo sign-flip (10^6 draws) and Wilcoxon;
- leave-one-seed-out range;
- even vs odd seed (column-parity) check;
- per-seed table;
- exposure (realised sampled R, expected U and pool, between-arm pool-share check);
- TD loss after removal.

Analysis script: `scripts/analyze_pqn_identity_swap.py`.

**Scope of the claim.** Δ compares R with the clean counterfactual U, so it measures their relative value as retained data: "restricted data is worse to keep than matched clean data". It does not split this into "R is harmful" vs "U is helpful". A retain-Z reference arm would do that; it is not included, in order to spend the budget on seeds.

## 2026-10-12 — Phase 6d run status (blind; no results viewed)

- 12/32 seeds complete (0–11). Each has both arms; all Phase 6b consistency checks are True (forks and pool stats bit-identical), and the construction asserts passed. Summaries are committed as they finish. Their filenames carry intermediate commit hashes; those commits contain result files only (`git diff a873dc0 HEAD -- src scripts` is empty).
- Seed 12 is in progress. The queue (`scripts/queue_p6d_idswap.sh`) is resumable: it skips complete seeds and deletes and reruns any partial seed from scratch.
- Operational note: during long in-turn waits the sandbox clock stopped advancing (the VM appears to be suspended), so the run only progresses while the sandbox is active. The primary contrast has not been computed. The analysis runs only at n = 32, as pre-registered.

## 2026-10-13 — Phase 6d matched-identity purge: RESULTS (n = 32, as pre-registered)

**Correction to the 2026-10-12 status note.** The "sandbox clock stopped advancing / VM suspended" explanation was wrong. I had issued several wait commands in one tool block, and those run in parallel, so five 10-minute waits took 10 minutes in total. No run was affected; only my monitoring was misread.

**Run.**
- Seeds 0–11 ran on 2026-10-12. Seeds 12–31 ran on 2026-10-13, after another sandbox restart: the venv was rebuilt with identical pins, and the interrupted seed 12 was rerun from scratch; its partial manifest is kept outside the repo.
- Before resuming, seed 0 was rerun under a scratch tag. All logged arm traces were bit-identical to the committed seed 0, so the rebuilt environment reproduces the run exactly. That check summary was accidentally overwritten when moved (summary and manifest share a filename), so the check was rerun after the analysis, again bit-identical: `results/checks/p6d-rebuild-check-s0.{summary,manifest}.json`.
- All 32 seeds have both arms. Seeds 0–11 pass the fork and pool consistency checks against Phase 6b, and all construction asserts passed.
- The code is unchanged since the pre-registration commit a873dc0 (`git diff a873dc0 HEAD -- src scripts` is empty). Summary filenames carry intermediate result-only commit hashes.
- The primary contrast was computed once, after n = 32. Analysis `results/summary/p6d_idswap-analysis.md`; figure `figures/p6d_idswap.png`.

**Primary.**
- Δ = AUC(purge_R) − AUC(purge_U) = **+0.46**, 95% CI **[−0.19, +1.11]**, 90% CI **[−0.08, +0.998]**. t(31) = 1.44, p = 0.16; 19/32 seeds > 0; SD 1.80 (the planning SD was 2.0); dz = 0.25.
- TOST vs ±1.0: p_lower < 0.0001, p_upper = 0.0495.
- **Pre-registered decision C: no meaningful identity effect (equivalence within ±1.0 AUC).**
- The equivalence call is marginal. The 90% upper bound sits 0.0017 below the SESOI, which is robust to the 4-decimal storage rounding (≤ 0.00005) but would flip with a slightly different SESOI or a few more unlucky seeds. The point estimate is positive, in the direction of the earlier contrasts. The defensible statement is: an identity effect of 1 AUC point or more is rejected at α = 0.05; smaller effects (≲ 1 point) remain compatible.

**Secondary (descriptive).**
- Windows: early +0.20 [−0.28, +0.68]; middle +0.48 [−0.19, +1.15]; late +0.55 [−0.41, +1.50].
- Greedy-eval AUC: +0.72 [−1.68, +3.13].
- Column parity: even seeds +0.44, odd seeds +0.48. No column artifact.
- Sign-flip p (Monte Carlo) 0.159; Wilcoxon p 0.115; leave-one-seed-out Δ +0.35..+0.58.
- Exposure:
  - purge_R: realised sampled R 0; expected U 0.325M; pool 0.325M.
  - purge_U: realised sampled R 0.325M (per seed 0.324–0.326); U 0; pool 0.325M.
  - The pool-share difference between arms is exactly 0 at every checkpoint.
- TD loss over the first 30 updates: purge_R 0.770, purge_U 0.850.

**Reading.**
- With slots, ages, pool, exposure, retained residence, seams and learner state all matched, keeping restriction-era transitions instead of matched same-epoch clean transitions costs less than 1 AUC point, and the estimate is about +0.5.
- Swapped exposure here is ~0.33M sampled transitions (20.5% of the buffer). That is comparable to Phase 6c's B − Cp (0.31M, +1.1, n.s.) and to the post-hoc identity main effect (+1.1, borderline).
- Taken together, the identity of the restriction-era data is at most a small carrier of damage at this dose.
- **What remains:**
  - The large purge benefits of Phases 6/6b (B − A +3.5–4.6) are mostly not explained by the restricted data's identity. The candidates are replacement residence (Phase 6c main effect +1.5) and refreshed or extra replacement data in general.
  - The network carrier is the robust finding: B − D −2.9* with identical buffers, plus the surgery results.
- **Scope:** Δ compares R with U (relative value as retained data). It cannot separate "R harms" from "U helps", which would need a retain-both arm. The result covers one dose (20.5% of the buffer, ~0.33M samples), one env and one restriction.

---

## 2026-10-13 — Cross-algorithm causal narrative (synthesis; no new runs)

User request: finish the cross-algorithm causal narrative, keeping PPO actor-state surgery and PQN-R network-versus-buffer surgery central and replay composition a qualified secondary finding.

- **Deliverables:**
  - `docs/causal_narrative.md`: draft Results/Discussion spine. It covers the claim hierarchy, the shared phenomenon table, PPO actor surgery, PQN-R network vs buffer, the cross-algorithm synthesis table, the replay-composition ladder with qualifications, non-claims, threats to validity, the pre-submission plan and an evidence table.
  - `figures/causal_summary.png` via `scripts/make_causal_summary.py`: a 3-panel forest plot (A: PPO, B: PQN-R, C: replay composition with the SESOI band). The script asserts each of its 18 "est [lo, hi]" strings verbatim against the cited `results/summary/*.md` file and exits on any mismatch.
- **Synthesis decisions (no new numbers):**
  - **Central:**
    - PPO actor −12.7 (89%); transplant +10.9* move / +12.9* fire; reset −27.2* ≈ scratch; optimizer null by two tests.
    - PQN-R network −5.6* (100k) / −3.2* (1M); B−D −2.9* on identical buffers.
  - **Buffer carrier is a mediator, not a store.** Newly highlighted from the existing `p6_surgery_pqnr1m_n8-phases.md`: the 1M buffer main effect is −1.4* while the data is present and −2.9 [−4.1, −1.8]* after it is fully overwritten.
  - **Secondary:** purge +4.6*, targeted vs sham +2.6* (residence-confounded), 6c identity +1.1 n.s. / residence +1.5 (Holm 0.0726), 6d +0.46 equivalence (marginal).
  - **Non-claims:** no early critical period; no direct plasticity measurement; no PQN-R optimizer split; one environment.
  - **Recommendations:** reframe the title away from "early damage"; extend the PPO surgery to ≥12 seeds first.
- **README:** added a top-level "Causal narrative" section with the figure. The stale "Status and results so far (2026-10-12)" header became "Results by phase (last updated 2026-10-13)", and the Status line was updated through 6d.
- **Environment note:** the sandbox restarted again. `.venv` was rebuilt with the pinned versions and the git remote re-added (token-free URL).

---

## 2026-10-13 — Phase 7: point of no return (causal rescue curve) — plan and Stage 0 gate, written BEFORE any Phase 7 data

**User request.** Find the point during a damaging intervention at which the damage becomes encoded in the network, and test whether a measurable state variable at removal predicts which runs remain rescuable. A strong result needs two things: a threshold that predicts persistent recovery impairment across seeds and interventions, and an intervention showing that preventing the threshold crossing preserves recovery. User decisions: **PPO**; **staged, with a pilot gate**; **8 seeds** for Stage 1.

**Priors from our own data (stated now, so they can't be rationalised later):**
- At d = 50 the delay-adjusted residual is only −2.0 (Phase 2, Q4). There may be no persistent impairment to predict.
- Masked-logit drift is not causal for recovery: Phase 4 F/A prevented it without changing AUC. The critic swap was null.
- Leading candidate: **representation retention**, i.e. whether the onset policy is still linearly decodable from the damaged actor's features.

**Design (`scripts/ponr_sweep.py`).** All windows start at onset w0 = 100, so they are nested: one masked trajectory D passes through the exact removal state for every duration d, and one clean trajectory C serves as the reference.
- Recovery arms at each removal w1 = w0 + d run for a fixed budget of R = 250 updates (2.05M steps, the Phase 2 Q4 horizon). Env state, RNG and update counter come from C at w1, as in Phase 4.
  - `D_online`: compose(C, D, "DDD"), the agent's own recovery.
  - `D_transplant`: compose(C, D, "CDD"), a clean actor from C at w1 plus the damaged critic and Adam state.
- Eval: deficit-free, every 10 updates, 64 envs, eval key 10000. Bank: 2000 onset-policy observations per seed, bank seed 20000 (identical to Phase 4).
- Predictor probes (`src/rled/probes.py`; the candidate set below was fixed before any data) are computed at every checkpoint for C and D through the mask and for every recovery arm:
  - policy: KL(onset‖t), conditional KL among masked actions, masked-logit state correlation, p(masked), entropy;
  - representation: decode R² h1/h0, 1−CKA a0/a1, effective rank a1, dead units a1;
  - critic: value correlation, value nMSE;
  - weights: relative change of actor and critic weights.
- Params are saved at the removal points only.

**Outcomes (per seed and duration):**
- **L_d (gate outcome):** late delay-adjusted deficit = mean over k ∈ {210, …, 250} of D_online(w0 + d + k) − C(w0 + k). Negative = impairment beyond pause-equivalent lost time at the end of the budget.
- A_d: the same difference averaged over k = 10…250 (delay-adjusted AUC).
- Raw: D_online(w1 + k) − C(w1 + k).
- Transplant rescue: D_transplant − D_online, and D_transplant − C(w1 + k).

**Stage 0 pilot:** movement restriction (allowed [0, 3]), seeds 0–3, d ∈ {50, 100, 200, 300}. Exploratory.
- Reproducibility anchor: at d = 50 the removal evals must equal Phase 4 exactly (D [21.875, 28.75, 19.2344, 4.2969], C [64.0469, 53.1406, 71.7344, 45.7344]).
- **Seeds 0–3 are excluded from every Stage 1 confirmatory test.** Stage 1 will use fresh seeds 100–107.

**Gate (fixed now):**
- **PROCEED** to the Stage 1 pre-registration if, for at least one d, mean L_d ≤ −5 return points and the 95% paired-t CI (n = 4) has an upper bound < 0.
- **EXTEND** (a Stage 0b pilot at d ∈ {400, 500}) if seed-mean L_d decreases monotonically in d and L_300 ≤ −5, but no CI excludes 0.
- **STOP** otherwise. Report: no point of no return up to 300 updates (2.46M steps) of movement restriction within a 250-update budget; damage remains lost time. Predictor trajectories would be reported descriptively only.

**What the pilot may set** (and nothing else): the Stage 1 duration grid, the SESOI and "rescuable" cut-off, and a predictor shortlist of at most 3 (plus acute deficit and duration as baseline covariates). Stage 1 will test the shortlist on fresh seeds and on the held-out fire intervention. Stage 2 (preventing the crossing, against an active control matched on acute collapse) will be pre-registered only after Stage 1.

### Phase 7 Stage 0 pilot — results (2026-10-13)

**Run:** `runs/20261013-p7-ponr-pilot-move-ppo-SpaceInvaders-MinAtar-ba1802f.json` (commit ba1802f, clean tree, 2683 s). Analysis: `scripts/analyze_ponr.py` → `results/summary/p7_ponr_pilot-analysis.md`, `figures/p7_ponr_pilot.png`. Analysis run once, with the gate rule fixed beforehand.

**Reproducibility:** at d = 50, the removal evals of C and D equal Phase 4 exactly for all four seeds (the nested design reproduces the Phase 4 forks).

| d (updates) | eval at removal D vs C | **L_d (late, delay-adj.)** | A_d (delay-adj. AUC) | raw AUC vs C(w1+k) | transplant − D_online | transplant vs C(w1+k) |
|---|---|---|---|---|---|---|
| 50 | 18.5 vs 58.7 | **−0.4 [−9.5, +8.7]** | −0.3 [−3.7, +3.1] | −11.5* | +8.3* | −3.2 [−6.8, +0.5] |
| 100 | 15.3 vs 70.4 | **+4.8 [−4.3, +13.8]** | −1.5 [−7.0, +4.0] | −22.6* | +18.2* | −4.4 [−9.5, +0.7] |
| 200 | 13.4 vs 99.3 | **+2.3 [−7.2, +11.7]** | −0.8 [−3.0, +1.3] | −33.9* | +26.7* | −7.1 [−14.2, −0.1]* |
| 300 | 7.3 vs 111.3 | **−1.4 [−8.2, +5.5]** | **−7.6 [−10.6, −4.6]*** | −48.8* | +42.8* | −6.1 [−14.2, +2.1] |

**Gate decision: STOP**, by the pre-written rule:
- no d has L_d ≤ −5 with a CI below 0;
- the seed means (−0.4, +4.8, +2.3, −1.4) do not decrease monotonically.

As pre-specified, Stage 1 is **not** run. Predictor results are descriptive only.

**What the pilot shows:**
1. **No point of no return within 300 updates (2.46M steps) of movement restriction**, three times the agent's whole pre-onset history. Lengthening the mask lowers competence at removal (unmasked eval 45.4 at onset → 18.5, 15.3, 13.4, 7.3). Raw damage grows almost linearly with d (−11.5 → −48.8), but by the end of the 250-update budget every duration is back at the pause-equivalent level. The damage stays lost time.
2. **The only duration-dependent cost beyond lost time is transient.** At d = 300 the delay-adjusted AUC is −7.6 [−10.6, −4.6]*: recovery starts more slowly, but the gap closes by the end of the budget (L_300 = −1.4).
3. **The competent policy stays latent in the damaged features.** At d = 300, onset logits remain linearly decodable from D's a1 features (R² 0.905), more so than from the clean C at the same update (D − C +0.13), because C has moved past its onset policy. The masked-action logit structure also survives (corr 0.78).
4. **The critic degrades fast and then saturates.** D's value correlation with onset falls to ≈ 0.19 within ~30 updates and stays flat, so it cannot track duration. A damaged critic briefly disrupts a transplanted clean actor (the dips in the dotted curves). Transplant vs C has a small residual of −3 to −7, significant only at d = 200.
5. **No probe explains seed-to-seed differences** at a fixed duration (within-d |ρ| ≤ 0.45, n = 16 nested units). The across-d associations are confounded with duration. There is no basis for a predictor shortlist.

**Limits:**
- With n = 4, the L_d CIs are about ±9, so an impairment of roughly −8 at d = 300 is not excluded. "STOP" means none was detected, not that the effect is zero (no equivalence test was pre-registered for the pilot).
- One intervention (movement), one budget (250 updates), one onset.

**Implication for the narrative:** this strengthens "lost competence, not lost plasticity" in PPO. Even a mask six times longer leaves a fully recoverable agent whose features still encode the competent policy.

---

## 2026-10-13 — Phase 7b: can a removal-time measurement predict slow recovery? Pre-specification, written BEFORE any of these metrics are computed

**User request.** Stop extending the duration sweep. Use existing checkpoints for a cheap diagnostic: does a pre-removal measurement predict which agents recover slowly? Choose a small, motivated predictor set; do not pick whichever metric correlates best after inspecting everything; check consistency across seeds and durations; pre-register the strongest candidate for a separate causal test. A null is informative.

**Disclosure of what has already been seen.**
- The pilot analysis (`p7_ponr_pilot-analysis.md`) already printed Spearman correlations of every *onset-relative* probe (KL from onset, CKA vs onset, decode R², value correlation, etc.) with L_d and A_d.
- Those probes are therefore **excluded from selection**. They will be reported only as context.
- The outcomes are known: seed-mean A_d = −0.3, −1.5, −0.8, −7.6.
- The four predictors below are **clean-fork-relative** and have not been computed for any unit.

**Units.** The 16 pilot units: seeds 0–3 × d ∈ {50, 100, 200, 300}, movement restriction. Data: the saved removal params `results/<pilot stem>/{C,D}_u{w1}.npz` and the pilot bank (2000 onset-policy observations per seed).

**Outcome (recovery cost).**
- **Primary:** A_d, the delay-adjusted AUC = mean over k = 10…250 of D_online(w1 + k) − C(w0 + k). More negative = slower recovery beyond lost time.
- Secondary: A_early = the same over k = 10…100 (where the d = 300 excess is concentrated).

**Predictors.** Exactly four, measured at removal on the bank. Each compares D(w1) with the clean fork C(w1) at the same update and uses unmasked policies.

| # | Predictor | Definition | Motivation | Expected sign vs A_d |
|---|---|---|---|---|
| P1 | `kl_clean` | KL(π_C ‖ π_D), full action set | The actor carries the damage (surgery), so the distance the policy must travel back should set the recovery cost | − |
| P2 | `cka_clean` | 1 − linear CKA(h1_D, h1_C), actor a1 features | If the features have diverged from where clean learning took them, relearning needs more than a readout change | − |
| P3 | `move_gap_rel` | On movement-relevant states (the half of the bank with the highest π_C(left) + π_C(right), per seed): mean of p_C(move) − p_D(move) | Under-use of the restricted actions exactly where the clean agent needs them | − |
| P4 | `value_err_clean` | mean (V_D − V_C)² / var(V_C) on the bank | A damaged critic gives wrong advantages to a relearning actor (cf. the transplant dip) | − |

Baselines (not candidates): duration d, and acute deficit = eval(D, w1) − eval(C, w1).

**Criteria.** A predictor "robustly tracks recovery cost" only if all of R0–R3 hold:
- **R0, reliability:** split-half bank reliability (metric on bank half 1 vs half 2, Spearman across the 16 units) ≥ 0.8. This is the "is the probe measurement reliable" check, applied to every predictor, critic included.
- **R1, pooled:** Spearman(P, A_d) ≤ −0.5 across the 16 units.
- **R2, within-duration:** Spearman of duration-centred P vs duration-centred A_d ≤ −0.3, **and** the per-duration ρ (4 seeds each) is negative in ≥ 3 of the 4 durations.
- **R3, incremental over duration:** leave-one-seed-out (4 folds). The model M1 = duration-group mean of A plus β · (duration-centred P), fitted on the training seeds, must lower out-of-fold MSE by ≥ 10% relative to M0 = duration-group mean only.
- Reported, not required: whether the predictor's within-duration |ρ| exceeds that of the acute deficit.

**Selection.**
- Among predictors passing R0–R3, the candidate is the one with the largest within-duration |ρ|. At most **one** candidate goes forward.
- If none passes, the conclusion is: *no robust removal-time predictor among policy divergence, representation divergence, relevant-state action change and critic error*. Then no causal test is pre-registered from this analysis.
- Caveat fixed in advance: there are 16 nested units from 4 seeds, so a passing predictor is a *candidate* only. Its confirmatory test must use fresh seeds.

**Secondary, exploratory, not used for selection:**
1. The same four metrics at early recovery (+10, +20, +30 updates; D_online vs C at the same update). These come from a deterministic re-run that must reproduce the pilot's removal evals bit-for-bit. They are related to A_d and A_early.
2. Phase 4 F/A context: drift suppression changed onset-relative KL without changing recovery AUC.

**Causal-test mapping, fixed now** (so the intervention is not chosen after the result). The test runs at d = 300 (the only duration with cost beyond lost time) on fresh seeds, against an active control matched on acute collapse:
- **P1:** during the mask, add a KL penalty pulling D's unmasked policy toward its concurrently trained clean twin. Active control: the same penalty toward the onset policy (the Phase 4 A arm).
- **P2:** feature distillation of D's a1 toward the twin's a1 on the bank, with the actor head free. Active control: matched-strength logit anchoring.
- **P3:** a KL penalty only on movement-relevant states, toward the twin's movement probabilities. Active control: the same penalty on the irrelevant half.
- **P4:** at removal, transplant the clean twin's critic into D (keeping D's actor and Adam state), vs D_online. A second arm runs critic distillation during the mask.

### Phase 7b results — removal-time predictors of recovery cost (2026-10-13)

**Provenance:** pre-specification aabd1ae; analysis script committed before running (39c3049); run once. Outputs: `results/summary/p7b_predictors-analysis.md` (plus `-units.json`) and `figures/p7b_predictors.png`. No training; the saved pilot removal checkpoints were used. Self-checks: identical params give 0 on all four metrics, and the numpy forward pass matches the JAX probe forward.

| Predictor | R0 reliability | R1 pooled ρ | R2 within-d ρ (per-d signs) | R3 LOSO gain | Result |
|---|---|---|---|---|---|
| P1 kl_clean | +0.98 ✓ | −0.57 ✓ | +0.05 (2/4 negative) ✗ | −25% ✗ | fail |
| P2 cka_clean | +0.99 ✓ | −0.68 ✓ | −0.11 (1/4) ✗ | −8% ✗ | fail |
| P3 move_gap_rel | +0.97 ✓ | −0.15 ✗ | +0.13 (1/4) ✗ | −8% ✗ | fail |
| P4 value_err_clean | +0.87 ✓ | −0.54 ✓ | −0.35 (3/4) ✓ | −83% ✗ | fail |
| acute deficit (baseline) | – | +0.64 | +0.17 | −16% | – |

**Verdict (pre-specified rule): no predictor passes R0–R3.**
- Conclusion: there is no robust removal-time predictor of recovery cost among policy divergence, representation divergence, relevant-state action change and critic error.
- **No causal test is pre-registered from this analysis.**
- Nearest miss: critic error (P4) meets R0–R2 but makes out-of-fold prediction much worse, driven by one extreme unit (seed 3, d = 300, nMSE 30.6). It is not a candidate under the rule.
- All four predictors are reliable measurements and all rise steadily with duration. Their pooled correlations with recovery cost come from the shared duration trend, and none adds anything once duration is known.
- The secondary outcome (A_early) gives the same picture: within-duration |ρ| ≤ 0.09 for all four.

**Post-hoc (labelled): the outcome's own reliability** (`scripts/posthoc_ponr_outcome_reliability.py`).
- Within duration, odd- vs even-checkpoint A_d agree at ρ +0.19 (Spearman-Brown reliability 0.30).
- Early vs late recovery cost agree at ρ −0.01.
- The ceiling on any predictor's within-duration correlation is therefore ≈ 0.54.
- With 4 seeds and 64-episode evals, whether a given agent "recovers slowly" at a fixed duration is mostly noise. The reliable variation in recovery cost is the duration effect, concentrated at d = 300.

**Deviation from the pre-specification (disclosed).** Secondary item 1 (the four metrics at +10/+20/+30 early recovery, from a deterministic re-run) was **not run**. Its outcome is the same low-reliability A_d/A_early, so it could not change the conclusion. It stays available: the re-run is deterministic and would reproduce the pilot exactly. Secondary item 2 (context) stands as already logged: Phase 4 F/A drift suppression moved KL-from-onset at removal without changing recovery AUC.

**Interpretation.**
1. Simple policy divergence, representation drift, under-use of restricted actions and critic error each grow with restriction length. None of them explains recovery cost beyond what restriction length already explains.
2. As currently measured, the "which agents recover slowly" question has little stable per-agent signal to explain. Answering it would need either much more reliable per-agent outcomes (more eval episodes and seeds) or experimental dissociation: interventions at a fixed duration that move one candidate without the others, as F/A did for policy drift.

## 2026-10-13 — Paper draft (LaTeX, TMLR anonymous submission format)

- `paper/main.tex` + `paper/refs.bib` (TMLR style files from JmlrOrg/tmlr-style-file), compiled with tectonic 0.15.0 to `paper/main.pdf` (21 pages incl. appendix). Figures copied from `figures/` into `paper/figures/`.
- Working title: "Where Does Deficit Damage Live? Actor, Network, and Replay Carriers of Post-Competence Damage in Deep Reinforcement Learning".
- Every number is taken from `results/summary/*.md`, converted to the single convention arm − reference (negative = harm). The PPO recovery-test numbers (previously printed only to stdout) were regenerated with the corrected t-interval script and saved as `results/summary/p2_recovery_all-analysis.md` so the paper's Table 1 has a committed source.
- The repository URL is anonymized in the PDF (`[anonymized repository]`); switch to `\usepackage[preprint]{tmlr}` / `[accepted]` and fill in authors for a non-anonymous version.
- Sandbox restart during this step again rolled `.git` back (to 02293d4); re-synced with `git reset --mixed origin/main` (479af27) before committing; working files were intact.

## 2026-10-13 — Phase 8: reviewer-response experiments — PRE-REGISTRATION (written and committed before any Phase 8 data)

**Context.** A review of the paper draft listed ten weaknesses. Six can be addressed with new runs within an 8-hour CPU budget. The others (non-additive attribution percentages, borderline replay equivalence, delay-adjusted baseline, predictor reliability) need re-analysis and narrower claims, which do not change any pre-registered contrast. All runs use 2 CPU cores and execute sequentially through `scripts/queue_p8.sh` (resumable; one job per process).

**Code changes (committed with this entry; 24/24 tests pass; tiny-config smoke runs deleted):**
- `ppo.py`: new runner flag `behaviour_ratio_learner`, default False. When a behaviour swap is active and the flag is set, the learner's own log-prob of the behaviour action is stored, so PPO's ratio starts at 1. There is no importance weighting; the data content still comes from π_b. With the flag off, the computation is bit-identical: `jnp.where(False, ·, log_prob)`.
- `surgery_ppo.py`: `--arms` subset; new arms `CCC+bDr` and `DDD+bCr` (ratio-neutral versions of the Phase 3 data arms); `--deficit {action_mask, reward_flip}` with `--flip-prob`; `--eval-fork`, which evaluates C and D during [w0, w1) for delay-adjusted readouts; per-arm swap diagnostics (clip fraction, |ratio − 1|, approx. KL, behaviour KL) for the first K and next K updates.
- `pqn_surgery.py`: 3-letter arm codes (params, optimizer state, buffer). The existing 2-letter path is unchanged. In a smoke run, `DDD` reproduced `DD` exactly.

**J1 + J2 — PPO component surgery to n = 12 and the de-confounded data arms (SpaceInvaders, movement, window [100, 150), recovery 150, K = 10).**
- J1: seeds 4–11 (fresh; two processes of 4 seeds), all Phase 3 arms (2^3 factorial, `CCC+bD`, `DDD+bC`) plus `CCC+bDr` and `DDD+bCr`.
- J2: seeds 0–3, arms `DDD+bC` (bit-identity check against Phase 3), `DDD+bCr` and `CCC+bDr`. Seeds 0–3 factorial arms come from the Phase 3 summary (365cf2b). The forks are verified against the Phase 2 recovery test via `--reference`.
- **Primary (n = 12, AUC over +10..+150, paired t, Holm across the three):** actor, critic and optimizer main effects. **Equivalence:** TOST for the critic and optimizer main effects with SESOI ±3.0 AUC (≈ 20% of the n = 4 total damage of −14.3).
- **Attribution (fixes the non-additivity point):** an exact Shapley decomposition of total damage DDD − CCC over the three carriers. It sums to the total by construction, and each carrier's share is reported with a seed-paired t-CI. The factorial main-effect "shares" are no longer described as a partition.
- **Replication check:** the actor main effect on the fresh seeds 4–11 alone (n = 8).
- **Data arms (primary, n = 12, Holm across the two):** `DDD+bCr − DDD` (clean data content without importance weighting) and `CCC+bDr − CCC`. Descriptive: the IS versions, the IS − ratio-neutral differences, and the swap diagnostics.
- Reading: a positive `DDD+bCr − DDD` with a CI excluding 0 means clean data content helps a damaged learner without off-policy importance weights. Any clipping that remains then comes from the learner's own policy change, i.e. from the data.

**J3 — Asterix replication (PPO, allowed {noop, up, down} = [0, 2, 4], meaning-matched onset u390 from the long control, window [390, 440), recovery 200 updates, seeds 0–7, 2^3 factorial, eval-fork).**
- Primary: total damage DDD − CCC (AUC +10..+200), and the actor, critic and optimizer main effects (Holm across the three); Shapley shares as above.
- **Replication criterion (fixed now):** total damage CI < 0, AND actor main effect CI < 0, AND the actor's Shapley share is the largest of the three.
- If the total damage CI includes 0, we report "no damage to localize", not a failure to replicate.
- Secondary: unlearning at removal, D(w1) − D(w0); delay-adjusted AUC, D(w1 + k) − C(w1 + k − 50).

**J4 — PQN-R parameters vs optimizer state (100k buffer, late onset 250, duration 100, recovery 650, seeds 0–7 in two processes; arms DDD, CDD, DCD).**
- 2 × 2 params × optimizer at the damaged buffer: the cells are DDD and CDD/DCD from this run, plus CCD = the earlier CD arm (clean network with damaged buffer, same seeds; bit-identical by construction).
- DDD must equal the earlier DD arm. A mismatch is reported and blocks the analysis.
- Primary (AUC +10..+650, ε = 0.05 eval, Holm across the two): parameter rescue CDD − DDD and optimizer rescue DCD − DDD. Secondary: interaction, and the per-window (+10..+150) versions.

**J5 — point-of-no-return confirmation (PPO, movement, onset 100, d = 300 only, recovery 250, fresh seeds 100–107 in two processes, 64-episode eval as in the pilot; arms D_online, D_transplant).**
- Primary: L_300 (delay-adjusted, mean over k = 210..250), n = 8, 95% t-CI, and **TOST against SESOI ±5** (the pilot gate value).
- Decisions:
  - **equivalent** if both one-sided p < 0.05;
  - **persistent impairment** if mean ≤ −5 and the upper CI bound < 0;
  - otherwise **inconclusive**.
- Secondary: A_300, and pooling with the pilot seeds 0–3 (n = 12, descriptive). Predictors are not re-tested (8 units at one duration is too few); the outcome's split-half reliability is reported descriptively.

**J6 — a different deficit family: reward sign flip (PPO, SpaceInvaders, reward_flip with prob 1.0, window [100, 150), recovery 150, seeds 0–7, 2^3 factorial, eval-fork).**
- Primary and replication criterion as for J3.

**J7 (only if time remains) — architecture:** PPO hidden 256 (2 × 256), movement restriction, otherwise as J1; seeds 0–7, 2^3 factorial. Same criterion.

**Analysis scripts** are written after the runs, but they only implement the contrasts above. Any additional analysis will be labelled post-hoc.

## 2026-10-13 — Phase 8 results: J1 + J2 (PPO surgery to n = 12, de-confounded data arms)

Runs: J1a seeds 4–7 (8c62fe0, clean, 2131 s); J1b seeds 8–11 (8a2de07, clean); J2 seeds 0–3 data arms (3f952c0, clean, ~730 s). Analysis: `scripts/analyze_p8_surgery.py` → `results/summary/p8_ppo_surgery_n12-analysis.md`. Bit-identity: J2 `DDD+bC` equals the Phase 3 arm for seeds 0–3 (loader assertion passed). The `--reference` reproduction check in the J1 summaries compares seeds 4–11 with the reference's seeds 0–3, so it is meaningless there. For J2 (seeds 0–3) the forks match exactly.

The queue failed once: after J1b, every remaining job hit the dirty-tree guard because I had edited `paper/main.tex` while the queue was running. No data were affected. I committed, then restarted the queue, which resumed at J2. From now on, paper edits are made outside the repo until the queue finishes.

**Pre-registered primary (n = 12, AUC +10..+150):**
- Total DDD − CCC −15.1 [−17.2, −13.0]*.
- Main effects (Holm across 3):
  - actor −13.7 [−14.9, −12.5]* (Holm p < 1e-4);
  - **critic −1.9 [−3.5, −0.4]* (Holm p = 0.038)**;
  - optimizer +0.2 [−0.9, +1.3] (Holm p = 0.74).
- TOST ±3: optimizer p = 0.0001 (**equivalent**); critic p = 0.081 (**not equivalent**: a small critic effect is detected; it is early, −4.4 at +10 and −0.2 at +150).
- Shapley (exact, sums to total): actor −13.6 [−15.0, −12.2]* (90%), critic −1.8 [−3.2, −0.4]* (12%), optimizer +0.3 [−0.9, +1.5] (−2%).
- Fresh seeds 4–11 only (n = 8): actor −14.2 [−16.0, −12.4]* → the actor result replicates.
- Correction to the paper: "critic not detectably involved" (n = 4) is revised to "small, transient critic contribution (~12%)".

**Data arms (primary, Holm across 2):**
- `DDD+bCr − DDD` (clean data, no importance weighting) **−5.4 [−7.0, −3.7]*** (Holm p < 1e-4): it *hurts*.
- `CCC+bDr − CCC` −2.5 [−6.6, +1.7] (n.s.).
- Descriptive:
  - IS arms replicate Phase 3: `DDD+bC − DDD` +8.3 [+6.1, +10.5]*; `CCC+bD − CCC` −9.4 [−14.7, −4.2]*.
  - IS − ratio-neutral: +13.6 [+11.5, +15.7]* (clean behaviour) and −7.0 [−11.1, −2.8]* (damaged behaviour).
- Diagnostics over the first K = 10 updates:
  - clip fraction 0.34 (IS) vs 0.20–0.23 (ratio-neutral) vs 0.07–0.09 (no swap);
  - KL(π_b‖π_learner) 0.13–0.18 (IS) vs 0.76–1.07 (ratio-neutral).
- **Reading (per the pre-registration):** the pre-registered positive reading is not supported. Clean data content does not help a damaged learner without importance weights. The Phase 3 data-swap benefit runs through the importance-weighted update, which pulls the learner toward the behaviour actor (behaviour KL is 5× lower under IS): off-policy distillation of the clean actor. This is consistent with the actor being the carrier.
- **Caveat:** the ratio-neutral arm is not a "pure data" arm either. It treats off-policy actions as on-policy, which biases the gradient. No arm cleanly isolates data content from the update rule.

## 2026-10-13 — Phase 8 results: J4 (PQN-R parameters vs optimizer state, 100k buffer)

Runs: J4a seeds 0–3 (3f952c0), J4b seeds 4–7 (bfca538); both clean, ~1130 s each. Old cells come from 77afd57 and c75175b (CD = CCD, DD, CC). **Bit-identity: Phase 8 DDD == earlier DD for all 8 seeds (PASS).** Analysis: `scripts/analyze_p8_pqnr_split.py` → `results/summary/p8_pqnr_split-analysis.md`.

- **Primary (Holm across 2, AUC +10..+650, ε = 0.05):**
  - parameter rescue CDD − DDD **+6.4 [+4.8, +8.0]*** (Holm p = 0.0001);
  - optimizer rescue DCD − DDD +1.3 [−0.4, +3.1] (Holm p = 0.12, n.s.).
- **Secondary:**
  - 2×2 at the damaged buffer: params main effect +6.2 [+5.1, +7.3]*; optimizer main effect +1.1 [+0.4, +1.9]* (small); interaction −0.4 (AUC).
  - Clean parameters alone recover 87% of the network rescue (CCD − DDD +7.3); clean RAdam state alone recovers 18%.
  - Transient: clean parameters with stale RAdam moments crash at +10 (CDD 11.1 vs CCD 29.9; interaction at +10 +17.1*). This is gone by +150.
- **Reading:** in PQN-R the network carrier is the parameters. Optimizer state contributes at most a small, early part.

## 2026-10-13 — Phase 8 results: J3 (Asterix replication, PPO)

Runs: J3a seeds 0–3 (bfca538), J3b seeds 4–7 (a8e2e91); clean, ~2030 s / 1850 s. Analysis: `results/summary/p8_asterix_surgery-analysis.md`.

- **Fork:** C at onset u390 5.36; D at removal 4.38 vs C 5.41. Unlearning during the window D(w1) − D(w0) −1.0 [−1.6, −0.3]* (≈ 18% of onset competence).
- **Primary:** total DDD − CCC (AUC +10..+200) −0.1 [−0.5, +0.4]. Main effects: actor −0.2, critic −0.0, optimizer +0.0 (all n.s.; all TOST ±3 p < 0.001; ±3 is a loose margin at this return scale).
- **Pre-registered criterion: NO DAMAGE TO LOCALIZE** (total CI includes 0). This is not a failure to replicate. Shapley shares are undefined.
- **Secondary:**
  - Delay-adjusted AUC +0.5 [+0.0, +1.0]*; end +1.0 [+0.2, +1.7]*. The restricted agent recovers to *ahead of* a paused control.
  - The acute deficit sits in the actor (actor main effect −1.0 [−1.3, −0.6]* at +10; clean actor +1.0* at +0) and is gone by +50.
- **Reading:** the timing pattern replicates on a second game (late deficit → unlearning → full recovery within lost time). The carrier claim cannot be tested there because no persistent damage exists to localize. Caveat: PPO on Asterix-MinAtar is a weak, slowly improving agent (returns ~5–8 over the studied range).

## 2026-10-13 — Phase 8 results: J5 (confirmatory point-of-no-return test, d = 300)

Runs: J5a seeds 100–103 (975ad08), J5b seeds 104–107 (b634055); clean, ~1310 s each. Analysis: `scripts/analyze_p8_ponr_confirm.py` → `results/summary/p8_ponr_confirm-analysis.md`.

- Unmasked eval: C at onset 46.3; at removal D 11.6 vs C 106.2.
- **Primary: L_300 +1.1 [−6.6, +8.7]** (n = 8); TOST ±5: p_lower 0.051, p_upper 0.130. **Pre-registered decision: INCONCLUSIVE.** Neither equivalence nor persistent impairment is shown; the mean is positive. Per-seed values: +8.3, −17.6, +2.2, −5.1, +9.6, +8.9, −1.0, +3.1. Seed 101 is an outlier.
- **Secondary:**
  - A_300 −4.6 [−12.3, +3.0] (pilot −7.6*).
  - Raw AUC −45.9*. The transplant rescues +39.4*, and late transplant vs pause reference is +26.6*.
  - Pooled with the pilot (n = 12, descriptive): L_300 +0.2 [−4.7, +5.1] (TOST p = 0.028); A_300 −5.6 [−10.4, −0.8]*.
- **Reliability:** split-half (odd/even checkpoints) reliability of A_300 across the 8 confirmatory seeds is Spearman +0.90 (Spearman–Brown 0.95; Pearson-based 0.97). The pilot's d = 300 seeds give Spearman 0.80.
  - **This contradicts the Phase 7b post-hoc reliability of 0.30.** That estimate pooled within-duration ranks over n = 4 seeds per duration, mostly at short durations where between-seed spread is small. The paper's "ceiling ≈ 0.54 on any predictor" statement must be withdrawn for long restrictions.
  - Early-recovery cost (+10..+120) and the late deficit L_300 remain uncorrelated (Spearman −0.05; pilot −0.01). Slow early recoverers do not end worse.
- Post-hoc note: the Phase 7b predictors (needing D-vs-C network comparisons) are not available from these summaries and were not re-tested, as pre-registered.

## 2026-10-13 — Phase 8 results: J6 (reward sign flip, PPO SpaceInvaders)

Runs: J6a seeds 0–3 (b634055), J6b seeds 4–7 (d072d3c); clean, ~1290 s / 1420 s. Analysis: `results/summary/p8_rflip_surgery-analysis.md`.

- **Fork:** C at onset 45.6; D at removal 0.00 (the agent learns to avoid reward) vs C 58.2. Unlearning −45.6 [−50.3, −40.9]*.
- **Primary:** total DDD − CCC −32.3 [−40.7, −23.9]*.
  - Main effects (Holm): actor −32.8 [−36.9, −28.7]* (Holm p < 1e-4); critic +1.2 [−2.5, +4.9]; optimizer −1.9 [−4.0, +0.2] (Holm p = 0.14).
  - Shapley: actor −32.4 [−36.5, −28.3]* (≈100%); critic +1.6; optimizer −1.5.
- **Pre-registered criterion: REPLICATES.**
- Clean actor into the damaged agent +31.9 [+24.2, +39.6]*. The hybrid vs clean is −0.4 [−3.4, +2.5]. A damaged actor alone gives −31.4*.
- **Secondary:** delay-adjusted AUC −18.6 [−27.2, −9.9]*; end −10.7 [−22.0, +0.6]. Unlike the action mask, the reward flip causes damage beyond lost time within the 150-update budget. It is again carried entirely by the actor.

## 2026-10-13 — Phase 8 results: J7 (PPO 2×256 architecture)

Runs: J7a seeds 0–3 (d072d3c), J7b seeds 4–7 (8abfef6); clean, ~2390 s each. Analysis: `results/summary/p8_h256_surgery-analysis.md`.

- **Fork:** C at onset 46.9; D at removal 18.3 vs C 68.2. Unlearning −28.7 [−36.4, −21.0]*.
- **Primary:**
  - total −19.7 [−23.7, −15.6]*.
  - Main effects (Holm): actor −15.7 [−17.6, −13.8]* (p < 1e-4); critic −2.5 [−4.5, −0.5]* (Holm p = 0.040); optimizer −0.4 [−2.6, +1.7]. Optimizer TOST ±3 p = 0.013 (equivalent); critic not equivalent (p = 0.30).
  - Shapley: actor −16.1 [−18.0, −14.1]* (82%), critic −2.9 [−5.4, −0.4]* (15%), optimizer −0.7 (4%).
- **Pre-registered criterion: REPLICATES.**
- Clean-actor rescue +15.7 [+11.7, +19.7]*. Delay-adjusted AUC −5.1 [−8.3, −1.8]*; end −2.4 [−7.8, +2.9].
- **Reading:** the 2×128 pattern (actor ≈ 90%, small transient critic, no optimizer effect) holds at double width.

## 2026-10-13 — Phase 8 wrap-up: paper revision

All Phase 8 jobs are complete. The queue ran 16:48–22:49 UTC (~6 h, inside the 8 h budget), with two short dirty-tree stops (see J1/J2 entry and the J4 commit).

**Post-hoc analyses (labelled as such in the paper):**
- `posthoc_alt_baselines.py`: same-update / pause / competence-matched baselines. Late-restricted agents relearn faster than clean agents from equal competence (savings): PPO move +18.0*, fire +14.4*, PQN +17.5*, PQN-R +7.2*.
- `posthoc_6d_sensitivity.py`: rung-4 TOST holds for ±1.00 and wider, fails at ±0.75, and needs ±1.24 at α = 0.01; MDE 0.92.
- `posthoc_power.py`: MDEs of the remaining n = 4 nulls.

**Paper changes, mapped to the ten reviewer weaknesses:**
1. Generalizability: new Section "Does the localization generalize? (pre-registered)": Asterix (no damage to localize; timing replicates), reward flip (replicates), 2×256 (replicates).
2. n = 4 power: PPO surgery to n = 12; "Power" paragraph; MDE table.
3. PONR null: confirmatory d = 300 test (inconclusive; large deficits excluded). Claims reworded to "not detected".
4. Non-additive 89%: replaced by exact Shapley values (90% at n = 12). Noted that 2×2 PQN-R main effects equal Shapley values.
5. PQN-R bundling: parameters carry 87% of the network rescue; RAdam state is n.s. apart from a transient.
6. PPO data-swap confound: ratio-neutral arms show the benefit runs through importance weighting (off-policy distillation of the clean actor).
7. 6d borderline: claim narrowed to "effects of ≥ 1 AUC are excluded", with a sensitivity analysis.
8. Delay baseline: three-baseline table and paragraph.
9. Predictor reliability: the claim is narrowed. The 0.30 reliability is contradicted at d = 300 (0.95), and the ceiling interpretation is withdrawn.
10. Asterix: replication completed.

**Other:**
- Abstract, contributions, discussion, limitations, conclusion, provenance table and deviations updated.
- `make_causal_summary.py` updated (20 verified numbers).
- Paper recompiled (25 pages).

## 2026-10-14 — Phase 9: second reviewer round — PRE-REGISTRATION (committed before any Phase 9 data)

**Context.** The second review lists six remaining weaknesses:
1. baseline-dependent conclusions with no declared primary estimand;
2. null claims (point of no return, predictors) still underpowered or inconclusive;
3. post-hoc sensitivity analyses not clearly separated from confirmatory tests;
4. inconsistent treatment of the significant critic effect;
5. the scope of generalization, given that Asterix showed no lasting damage;
6. "no predictor found in this sample" vs "measurements cannot predict".

Points 3 and 4 are fixed in the text. Points 1, 2, 5 and 6 also get new data. Budget: ~6 h on 2 CPU cores. Queue: `scripts/queue_p9.sh` (sequential, resumable; logs in /home/user/p9logs). The analysis scripts were written and checked before this entry, on existing data only:
- `scripts/analyze_p9_estimands.py` reproduces the Phase 8 post-hoc values on the Phase 2 seeds: E_damage +1.1, E_plast +18.0.
- `scripts/analyze_p9_ponr_predictors.py` reproduces the Phase 7b d = 300 correlations on the pilot seeds: +0.80, +0.60, +0.40, −0.20.
No Phase 8 J5 predictor values were computed.

**Declared estimands (paper-wide, fixed now).** Each question gets one estimand. The three baselines are not alternatives for the same question.
- **Q-damage**, "damage beyond the training time the deficit consumes": **primary estimand E_damage = D(w1 + k) − C(w1 + k − L)**, the pause counterfactual.
  - Justification: it is the only one of the three that (a) charges the deficit exactly for the training time it occupies, and (b) is defined without conditioning on the restricted agent's post-treatment state.
- **Q-plasticity**, "does the restricted agent learn more slowly than a clean agent from equal competence?": **E_plast = D(w1 + k) − C(t* + k)**, competence-matched.
  - It conditions on the competence reached at removal (a mediator), so it answers only the rate question. It is never used to claim absence of damage.
- **E_total = D(w1 + k) − C(w1 + k)**, same-update: the total cost including lost time. Descriptive only.
- Carrier analyses (surgery) compare damaged and clean components at the same update and do not depend on this choice.

**E1 — confirmatory test of E_damage and E_plast (PPO, movement, late [100, 150), fresh seeds 200–211, n = 12, 440 updates; configs control + move_late; AUC over k = 10..250).**
- **E_damage:** TOST against SESOI ±3. Decisions:
  - **equivalent** if both one-sided p < 0.05;
  - **damage beyond lost time** if the mean ≤ −3 and the CI upper bound < 0;
  - otherwise **inconclusive**.
- **E_plast:** two-sided paired t. Decisions:
  - **savings** if the CI lower bound > 0;
  - **slower relearning** if the CI upper bound < 0;
  - otherwise inconclusive.
- The two answer different questions, so there is no multiplicity correction between them. E_total is descriptive.
- t* is defined exactly as in `posthoc_alt_baselines.py`: the first checkpoint at which C's 3-checkpoint moving average ≥ D(w1).

**E3 — independent confirmatory point-of-no-return study and predictor test (PPO, movement, onset 100, d = 300, budget 250, fresh seeds 300–331, n = 32, arm D_online only, 64-episode eval).**
- It is independent of Phase 8 J5 (seeds 100–107, inconclusive), which stays reported as is. J5 is pooled with E3 only descriptively.
- **Primary 1:** L_300 (k = 210..250), TOST ±5. Decisions as in J5 (equivalent / persistent impairment / inconclusive).
  - Power: with SD ≈ 9 (J5), the probability of declaring equivalence when the true mean is 0 is ≈ 0.84.
- **Primary 2:** the four Phase 7b predictors at removal (kl_clean, cka_clean, move_gap_rel, value_err_clean; definitions unchanged, imported from `ponr_predictors.py`) vs A_300.
  - Test: Spearman ρ, one-sided H1 ρ < 0, Holm across the four. A predictor is **supported** if Holm p < 0.05.
  - Power: 80% for ρ ≈ −0.45 at the uncorrected level and ≈ −0.53 at the Holm worst case.
  - For every predictor we report a 95% CI for ρ (Fisher z, Bonett–Wright SE). If none is supported, the conclusion is stated as bounds: associations stronger than the CI lower bounds are excluded. It is not stated as "unpredictable".
- **Descriptive:** predictor split-half reliability; A_300 split-half reliability; leave-one-out R² gain; correlations with L_300; pooling with J5.
- **Fallback:** if the budget ends before all 8 chunks finish, the completed chunks are reported descriptively and no decision is made.

**E2 — Asterix reward-flip surgery (PPO, Asterix-MinAtar, reward sign flipped with prob 1.0 during [390, 440), recovery 200, seeds 0–7, 2^3 factorial, eval-fork).**
- Rationale: Phase 8 J3 (Asterix, movement) produced only a small competence gap at removal (−1.0, ~18%) and no lasting damage. The lost-competence account predicts lasting damage only when the deficit removes enough competence (and the clean agent gains enough) relative to the relearning speed. A harsher deficit on the same game tests whether Asterix's null is game-specific or deficit-specific.
- **Primary:** the same replication criterion as Phase 8 (total CI < 0 AND actor main-effect CI < 0 AND the actor has the largest Shapley share); "no damage to localize" if the total CI includes 0.
- **Secondary:** gap at removal, unlearning, delay-adjusted AUC.
- **Post-hoc, labelled:** across all PPO surgery settings, the relation between gap at removal and total damage.
- Runs last; if it does not finish within the budget, it is reported as not run.

**Order:** E1 → E3 → E2 (~42 + 145 + 68 min).

### 2026-10-14 — Phase 9 E1 result (confirmatory recovery estimands; seeds 200–211, n = 12)
Source: `results/summary/p9_estimands-analysis.md`. D at removal 24.4; C at onset 41.3, at removal 60.7. t* mean 52 (range 30–90).
- **E_damage (primary, pause): +1.9 [−0.7, +4.4]**; TOST ±3: p_lower 0.0007, p_upper 0.171 → pre-registered decision **INCONCLUSIVE**.
  - The harmful side is rejected decisively: mean > −3 at p = 0.0007, and the CI excludes harm beyond −0.7.
  - Equivalence fails only because a *benefit* larger than +3 cannot be excluded.
  - It replicates the Phase 2 value (+1.1 [−2.9, +5.2], n = 4).
- **E_plast (competence-matched): +14.2 [+10.9, +17.5]***, p < 10⁻⁴ → **SAVINGS** (12/12 seeds positive). This replicates the post-hoc Phase 8 value (+18.0, n = 4).
- E_total (descriptive): −8.9 [−11.1, −6.6]*.

### 2026-10-14 ~07:00 UTC — Phase 9 deviation (budget; decided before any E2/E3 outcome was inspected)
- E3 chunks take ~27.5 min, not the estimated 18. The full queue would end at the budget limit, leaving no time for analysis.
- With the user's approval, E2 (Asterix reward flip) runs **seeds 0–3 only** (n = 4 instead of 8). E2-s4 was skipped with a done-marker; `queue_p9.sh` is unchanged.
- The E2 replication criterion is applied unchanged but is reported as an **underpowered, reduced-n confirmatory test**.
- E3 is unaffected (n = 32 as pre-registered). No E3 chunk outcome had been analysed; only the run log's one-line progress output for chunk 1 was visible.

### 2026-10-14 ~09:30 UTC — Phase 9 deviation 2 (sandbox restart; decided before inspecting any E3 outcome)
- A sandbox restart (~07:20 UTC) killed the queue during E3 chunk 2. About 2 h of the 6 h budget were lost; ~1 h 50 min remain.
- The partial chunk-2 output (shared history only) was deleted.
- With the user's choice, a resume queue `scripts/queue_p9b.sh` (same commands) runs:
  1. **E2 seeds 0–3** (as in deviation 1);
  2. **E3 chunks in pre-registered order until the budget ends**.
- E3 will therefore not reach n = 32. Per the pre-registered fallback, **E3 is reported descriptively with no decision** (no TOST decision, no predictor support/no-support verdict). Interval estimates and Fisher-z CIs for ρ are still reported as bounds.
- E1 is complete and unaffected.

### 2026-10-14 — Phase 9 E2 result (Asterix reward flip, seeds 0–3, n = 4 per deviation 1)
Source: `results/summary/p9_asterix_rflip_surgery-analysis.md`.
- **Pre-registered criterion: REPLICATES.**
  - Total damage −2.4 [−3.6, −1.2]*.
  - Actor main effect −2.2 [−3.3, −1.1]* (Holm 0.023).
  - Actor Shapley −2.3 [−3.2, −1.3]* = 93% (4/4 seeds).
  - Critic −0.1 and optimizer +0.0, both TOST ±3 p ≤ 0.0001.
- Clean-actor transplant +2.5 [+1.9, +3.0]*; hybrid vs. clean +0.1 [−0.9, +1.0].
- Fork readouts:
  - C at onset 6.01, at removal 5.98: the clean agent made no progress during the window.
  - D at removal 1.76; gap −4.2 [−6.0, −2.4]* (70% of C(w1); 18% under the movement restriction).
  - Delay-adjusted AUC −1.8 [−2.6, −1.0]*; delay-adjusted end −1.2 [−2.2, −0.1]*.
- Reading: the Asterix movement null was deficit-specific, not game-specific. When the deficit removes most competence, Asterix shows damage beyond lost time, and the actor carries it.
- Post-hoc cross-setting table: `results/summary/p9_gap_vs_damage-posthoc.md` (descriptive).

### 2026-10-14 ~11:10 UTC — Phase 9 E3 result (descriptive per fallback; seeds 300–315, n = 16 of 32)
Source: `results/summary/p9_ponr_predictors-analysis.md` (header notes that script decision labels are not applied). The queue was stopped at the budget limit; the partial chunk s316–319 was deleted unanalysed.
- L_300 +6.0 [+1.9, +10.2]: harm worse than −5 is rejected (p_lower < 10⁻⁴), but a benefit above +5 is not excluded. Pooled with J5 (n = 24): +4.4 [+0.8, +7.9].
- A_300 +1.3 [−0.7, +3.3] (split-half SB 0.78). Pooled with J5: −0.7 [−3.4, +2.1]. The pilot's early-recovery cost does **not** replicate.
- Predictors vs A_300 (ρ [95% CI]; LOO gain):
  - KL −0.24 [−0.66, +0.30], −18%;
  - 1−CKA −0.13 [−0.59, +0.39], −21%;
  - move_gap −0.15 [−0.60, +0.38], −14%;
  - value_err +0.18 [−0.35, +0.62], −8%.
  - Predictor split-half reliabilities 0.65–0.96.
  - Reported as "no predictor found in this sample"; moderate correlations (down to about −0.6) remain compatible.

### 2026-10-14 — Phase 9 paper revision (all six second-round weaknesses)
1. New §3.4 "Declared estimands": E_damage (pause, primary) with its justification; E_plast (competence-matched); E_total descriptive. Confirmatory Table 4 (E1). The B1–B3 appendix table is relabelled post-hoc.
2. "Not detected"/bounds wording throughout (abstract, contributions, §4, §5, §7 title and text, conclusion, captions).
3. New "Evidence status" table (confirmatory / exploratory / post-hoc). The rung-4 sensitivity analysis is labelled as not part of the test.
4. Critic made consistent everywhere (main effect −1.9*, Shapley 12%, transient time course, not equivalent to 0), with a paragraph on how the main effect relates to the Shapley value. §5 title is now "mainly in the actor".
5. Asterix scope paragraph, E2 row in Table 9, post-hoc gap-vs-damage table.
6. Predictor table with ρ CIs, and a "No predictor found ≠ unpredictable" paragraph. The early-recovery-cost claim was withdrawn after non-replication.
Appendix: provenance rows and the Phase 9 deviation item.

## 2026-10-14 ~13:10 UTC — Phase 10: third reviewer round — PRE-REGISTRATION (before any Phase 10 data)

**Context.** The third review lists ten points. Main concern: component surgery decomposes the *same-update* deficit (DDD − CCC), while the primary estimand is damage *beyond lost time* (pause counterfactual). So "damage lives in the actor" may overstate what the primary estimand supports. The other nine points get text revisions (scope, hedging, normalized effect sizes). Budget: 2.5 h.

**K: onset-reference surgery (pause counterfactual at the state level).**
- C and D share every bit of state up to w0, so the shared onset state is the state a *paused* agent would carry at removal.
- New flag `surgery_ppo.py --clean-source onset`: the "C" source of each component (actor, critic, Adam) is the onset state at w0 instead of the clean fork at w1.
- As in all surgery arms, env state, observations, RNG and the update counter come from the clean fork at w1.
- Notation: P = onset (paused) component.
- **Total PPP − DDD contrast:** DDD − PPP = damage beyond lost time, measured from a common post-removal environment state.
- **The 2³ factorial** (P vs D per component) decomposes it.
- Runs (8 arms each, recovery 150, same seeds as Phase 8 so DDD must reproduce bit for bit):
  - **K1:** SpaceInvaders reward flip, seeds 0–7 (Phase 8 delay-adjusted AUC −18.6*);
  - **K2:** SpaceInvaders movement 2×128, seeds 0–3 (E_damage ≈ 0, so we predict little or nothing beyond lost time to localize).
- **Primary (K1):** DDD − PPP AUC; actor/critic/Adam main effects (Holm across the three); exact Shapley.
  - Criterion "beyond-lost-time damage localizes to the actor": total CI < 0 AND actor main-effect CI < 0 AND the actor has the largest Shapley share.
  - "No beyond-lost-time damage to localize" if the total CI includes 0.
  - TOST ±3 for critic and Adam.
- **K2:** the same analysis, descriptive (n = 4). Prediction: the total CI includes 0 or the total is small.
- **Secondary/descriptive:**
  - (a) DDD reproduces the Phase 8 DDD cells (bit-identity check);
  - (b) PPP vs. the pause curve C(w1 + k − L) from Phase 8 fork evaluation (how well the state-level pause matches the time-shifted control);
  - (c) interaction size = DDD − PPP minus the sum of main effects; two- and three-way interaction contrasts.
- **Post-hoc, labelled:** interaction contrasts for the existing n = 12 fork-reference surgery; AUC normalized by the clean agent's AUC for the PPO and PQN-R carrier effects.
- **Order:** K1a, K1b, K2. Whatever does not finish within the budget is reported as not run.

### 2026-10-14 ~13:52 UTC — Phase 10 K1 result (onset-reference surgery, SpaceInvaders reward flip, seeds 0–7)
Sources: `p10_onsetref_rflip-analysis.md` (the script's "C" labels denote the onset component P), `p10_onsetref_rflip-checks.md`, `p10_interactions_onsetref_rflip-posthoc.md`.
- **Pre-registered criterion: beyond-lost-time damage localizes to the actor (REPLICATES).**
  - DDD − PPP −19.8 [−27.8, −11.8]*.
  - Actor main effect −22.0 [−25.2, −18.8]* (Holm < 10⁻⁴).
  - Actor Shapley −21.6 [−24.8, −18.4]* = 109% (8/8 seeds).
  - Critic +2.8 [−2.8, +8.4] (Shapley −16%; TOST ±3 p 0.47, not equivalent).
  - Adam −1.6 [−4.3, +1.0] (TOST p 0.13, not equivalent).
- Onset actor into damaged agent: PDD − DDD +21.2 [+14.9, +27.5]*; PDD 68.8 vs PPP 67.4.
- Checks:
  - (a) DDD bit-identical to Phase 8 (max |diff| 0).
  - (b) State-level pause PPP vs time-shifted pause C(w1 + k − L): +1.2 [−0.6, +3.0]. DDD − PPP −19.8 vs delay-adjusted −18.6: the state-level decomposition matches the primary estimand.
- Interactions (post-hoc): only actor × Adam −2.1 [−4.1, −0.2]* (3% of PPP AUC); non-additivity +1.2 [−7.2, +9.6].

### 2026-10-14 — Phase 10 K2 result (onset-reference surgery, SpaceInvaders movement, seeds 0–3; descriptive)
Source: `p10_onsetref_move-analysis.md`.
- DDD − PPP −2.3 [−4.8, +0.2] → **NO DAMAGE TO LOCALIZE**, as predicted from E_damage ≈ 0.
- Descriptively: actor main effect −3.9 [−5.7, −2.1]* (Shapley −3.6, 4/4 seeds); critic −0.2; Adam +1.0; PDD − DDD +2.7.
- DDD bit-identical to Phase 3 (max |diff| 0).
- Reading: in the movement setting, the fork-reference actor effect (−13.7) is mostly lost progress.

### 2026-10-14 — Phase 10 paper revision (ten third-round points)
1. Surgery vs. estimand: new §5 paragraph and Table `tab:onsetref` (surgery against the pause counterfactual); abstract, contributions, conclusion and §3.4 now separate the same-update deficit from damage beyond lost time.
2. Hybrid validity: new paragraph and Table `tab:interactions` (post-hoc). The claim is restricted to causal sufficiency within the intervention family; the critic's context dependence is reported.
3. Plasticity: §9 retitled "not a relearning deficit"; the claims are limited to same-task relearning; limitation added.
4. Replay: abstract, ladder and limitation now say "bounded, not zero".
5. PONR: already bounded (Phase 9); consistency checked.
6–10. New limitations (interventions; mechanistic readings; effect scales with Table `tab:relative`; incomplete confirmatory evidence). The PPO behaviour-swap conclusion is narrowed to "depends on importance weighting; no conclusion about data content alone".

## 2026-10-14 ~14:40 UTC — Phase 11: completing the unfinished pre-registered Phase 9 runs
- The user asked to continue. We resume the two Phase 9 experiments cut short by sandbox restarts, with **identical commands** (`scripts/queue_p11.sh`):
  1. **E2 seeds 4–7**: Asterix reward flip, bringing n to the pre-registered 8;
  2. **E3 seeds 316–331**: PONR + predictor replication, bringing n to the pre-registered 32.
- **Disclosure (optional-stopping risk):** the interim results are already public.
  - E2 at n = 4: REPLICATES.
  - E3 at n = 16: L_300 +6.0 [+1.9, +10.2]; no predictor supported.
- Completing to the pre-registered n was the plan from the start and does not depend on those results. However, it is not blind.
- Reporting rule (fixed now):
  - The pre-registered decision rules are applied at the final n and labelled "pre-registered n, completed after an unblinded interim look".
  - The n = 16 / n = 4 interim values stay in the paper.
  - If only part of the runs finishes, the completed n is reported descriptively, as before.

### 2026-10-14 ~15:12 UTC — Phase 11: E2 completed (Asterix reward flip, n = 8, pre-registered n; completed after the unblinded n = 4 interim)
Source: `p9_asterix_rflip_surgery_n8-analysis.md`.
- **REPLICATES.**
  - Total −2.3 [−2.9, −1.7]*.
  - Actor main effect −2.1 [−2.6, −1.7]* (Holm < 10⁻⁴).
  - Actor Shapley 94% (8/8 seeds).
  - Critic −0.1 and Adam +0.0, both TOST ±3 p < 10⁻⁴.
- Fork readouts: C 5.36 → 5.41 (no clean progress); D at removal 1.70; gap −3.7 [−4.6, −2.8]* (68% of C(w1)).
- Delay-adjusted AUC −1.7 [−2.1, −1.3]*; end −1.2 [−1.7, −0.6]*.
- The interim n = 4 result (−2.4, 93%) is unchanged in substance. The post-hoc gap table was regenerated with the n = 8 row.

### 2026-10-14 ~16:40 UTC — Phase 11: E3 completed (n = 32, pre-registered n; completed after the unblinded n = 16 interim)
Source: `p9_ponr_predictors_n32-analysis.md`.

**Primary 1, L_300: +2.3 [−4.6, +9.2] → INCONCLUSIVE** (TOST ±5 p = 0.21).
- Harm worse than −5 is rejected (p_lower 0.019); a benefit above +5 is not excluded.
- Median L +5.7. Pooled with J5 (n = 40, descriptive): +2.1 [−3.5, +7.6].
- The interim n = 16 estimate (+6.0) moved mainly because of one seed.

**Seed 330: the restricted agent never recovered** (L −92.0, A −75.2). Its eval was exactly 3.0 at every checkpoint for 250 updates.
- Diagnosis (post-hoc): no NaNs, no dead units.
- Its actor at removal had collapsed: entropy on the evaluation bank 0.19 (other D agents 0.43–0.53, clean agents 0.55–0.59), with 85% probability on fire.
- Its clean fork is normal (114 at +250).
- So 1 of 44 agents restricted for d = 300 (pilot 4 + J5 8 + E3 32) failed to recover within budget.
- Post-hoc, excluding 330: L +5.3 [+2.3, +8.4]; A −1.0 [−3.1, +1.2]. Only one other seed had L < −10 (316: −11.2).

**A_300: −3.3 [−8.4, +1.9]** (split-half SB 0.86). Pooled with J5: −3.6 [−7.8, +0.7].

**Primary 2, predictors (one-sided, Holm): kl_clean SUPPORTED**, ρ −0.48 [−0.72, −0.14], one-sided p 0.0026, Holm 0.010.
- The other three are not supported: 1−CKA −0.19, move_gap −0.24, value_err +0.03.
- LOO R² gain for kl_clean: +5% (small).
- Post-hoc robustness without seed 330: ρ −0.43 [−0.69, −0.07], one-sided p 0.008. Seed 330 had the largest KL of all 32.

### 2026-10-14 ~17:10 UTC: Phase 11 paper revision
`paper/main.tex` and `main.pdf` (33 pp.) were updated.

Asterix reward flip at n = 8:
- `tab:general`, the Asterix paragraph, `tab:gap` and `tab:relative`;
- status, provenance and deviation rows.

E3 at n = 32:
- the abstract (iv) and the contributions;
- the PONR section, rewritten with an inconclusive verdict, the non-recovering seed 330 (post-hoc diagnosis: policy collapse) and the exclusion sensitivity analysis;
- `tab:predictors`, now with a Holm column; KL is supported but framed as association only;
- limitations, conclusion, provenance, and Appendix deviations (interim-look disclosure, with interim values reported).

## 2026-10-14 ~17:50: Phase 12 PRE-REGISTRATION (generalization panel). Committed before any deficit run.

**Context.** Fourth-round review: the main concern is generalization (environments, deficits, algorithms, budgets), plus claim calibration (C1–C6). We had 8 hours. A fifth sandbox restart happened at about 16:40 (recovered to 7138fe2; venv rebuilt).

**Code change (no outcome seen).** `ppo.py` gains two deficits, both implemented as static Python branches:
- `action_noise`: an actuator fault. With probability p the *executed* action is replaced by a uniform random action, while the agent stores its own sampled action and log-prob. It uses a key split from the existing `k_flip` key, so no existing RNG stream changes.
- `reward_scale`: r → s·r inside the window; s = 0 is extinction.

Existing deficits trace identical code, so earlier runs are unaffected. `surgery_ppo.py` exposes `obs_noise`, `action_noise` and `reward_scale` (`--sigma`, `--noise-prob`, `--scale`).

**Onset rule (meaning-matched, Phase 5), generalized.** Late onset = the first 10-update multiple at which the long control's mean eval reaches **40% of the way from its u0 value to its final value** (final = mean over checkpoints in the last 50 updates).
- This equals the original 40%-of-final rule when the u0 return is about 0 (SpaceInvaders, Asterix, Freeway).
- It is needed for Acrobot, whose returns are negative.
- Long controls (`p12-longctl`, seeds 0–3, controls only, no deficit run):
  - CartPole-v1: u0 21.7, final 477.2 → **onset u10**.
  - Acrobot-v1: u0 −500, final −105.2 → **onset u60**.
  - Freeway-MinAtar: u0 0.5, final 52.1 → **onset u20** (window [20, 70)). The control is still rising slowly at u400 (+6 over the last 150 updates); disclosed.
  - Per-seed crossings: CartPole 10/10/20/30; Acrobot 60/50/50/150; Freeway 10/20/20/30.
- Breakout was dropped as a fallback because Freeway passed the competence gate (control learns: final ≫ u0) and Breakout PPO is bimodal (Phase 1).

**Tests (PPO 2^3 actor×critic×Adam surgery at removal, fork reference, eval-fork, unmasked/undeficited eval with CRN, window 50 updates, recovery 150 updates, seeds 0–7 in two chunks of four, factorial arms only).** Listed in priority order; tests not completed within the budget are reported as not run.

| id | env | deficit | window | why |
|---|---|---|---|---|
| G1 | CartPole-v1 | reward flip p = 1 | [10, 60) | new domain (low-dimensional state, no pixels) |
| G2 | Acrobot-v1 | reward flip p = 1 | [60, 110) | new domain, negative-reward task |
| G3 | SpaceInvaders | **actuator noise** p = 0.5 | [100, 150) | new deficit family: corrupts dynamics, not reward or action set |
| G4 | Freeway-MinAtar | reward flip p = 1 | [20, 70) | third MinAtar game |
| G5 | SpaceInvaders | **observation noise** σ = 1 | [100, 150) | new deficit family: sensory degradation |
| G6 | SpaceInvaders | **reward extinction** (scale 0) | [100, 150) | new deficit family: passive, no adversarial signal |

**Decision rules** (identical to Phase 8 / `tab:general`; `analyze_p8_surgery.py`; paired 95% t-intervals over seeds; AUC over recovery):
1. *Damage to localize* iff total DDD − CCC AUC has its 95% CI below 0. Otherwise the verdict is "no damage to localize" (not a failure).
2. If there is damage: the *localization criterion* is met iff the actor main effect has a CI below 0 **and** the actor has the largest Shapley share.
3. Critic and Adam main effects get a TOST with margin **±10% of the CCC (clean-continuation) mean AUC in that setting**. Return scales differ across environments; Phase 8 used ±3 points, which is ~3% of the SpaceInvaders scale.
4. Holm correction across the three main effects within each test.
5. Secondary (descriptive): gap at removal, delay-adjusted AUC and end difference (requires eval-fork), unlearning during the window, relative effects (% of CCC AUC).

**Pre-registered cross-setting summary.** Over all settings with damage to localize (earlier plus new): the number in which the actor is the largest carrier, and the range of actor Shapley shares.

**Ceiling caveat (stated in advance).** CartPole's clean agents are near the 500 ceiling by the window. Damage there can show only as slower relearning, so a null in CartPole is weak evidence.

**Claim-calibration work (text only; no new data):**
- C1: claims ledger with evidence status, n and fresh-seed status in the abstract, results and conclusion.
- C2: "carrier" defined operationally and the text audited.
- C4: SESOI motivation, the smallest margin that passes, and the one-sided harm bound.
- C5: what the interim looks did and did not influence, plus a conservative two-look adjustment.
- C6: exact (Clopper–Pearson) interval for 1/44 non-recoveries; restrained wording in the abstract and conclusion.

**Estimated compute:** about 4.7 h sequential (`scripts/queue_p12.sh`, logs in `/home/user/p12logs`). Analysis: `analyze_p8_surgery.py --fork-eval --sesoi <10% of CCC AUC>` per test, then a cross-setting table.

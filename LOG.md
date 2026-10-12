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

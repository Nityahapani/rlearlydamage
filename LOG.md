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

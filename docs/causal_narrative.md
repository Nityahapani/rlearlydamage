# Cross-algorithm causal narrative

**Status:** synthesis of Phases 1–6d (2026-10-13). No new experiments. Every number below is copied from a committed analysis file in `results/summary/` (sources in the evidence table, §9). Effects are paired-seed differences in post-removal return AUC with 95% paired-t CIs. `*` means the CI excludes 0. Summary figure: `figures/causal_summary.png`, built by `scripts/make_causal_summary.py`, which asserts each plotted value verbatim against its source file.

This document is written as the draft Results/Discussion spine of the paper. It sets the claim hierarchy: two central claims, one qualified secondary finding, and an explicit list of things we do **not** claim.

---

## 0. Claim hierarchy (one-paragraph version)

A temporary action restriction imposed **after** an agent has become competent causes it to unlearn during the restriction. The cost that remains after the restriction is lifted is mostly lost time. In both an on-policy and an off-policy algorithm, the part of the damage that survives state transplantation lives in the **function approximator**:

- in PPO, in the **actor parameters**, not the critic or the Adam state;
- in PQN-R, in the **network** (params + optimizer state, bundled by prespecified rule), with a replay-buffer contribution that appears only when the buffer is long-lived.

Damaged state is not toxic: resetting it is worse than keeping it. Experience is the channel through which damage is made and repaired, not a separate store of it.

- In PPO, a clean behaviour policy repairs the damaged learner almost as well as a clean actor transplant. A fixed stale dataset does not.
- In PQN-R, replay composition changes how quickly the network is retrained. Replacing restriction-era data helps.
- Once exposure is matched, however, the *identity* of the removed transitions accounts for less than 1 AUC unit (equivalence at a pre-registered SESOI of ±1.0, marginal).

---

## 1. The shared phenomenon: restriction timing changes the type of harm

Same environment (SpaceInvaders-MinAtar, gymnax), same deficit (movement actions removed; noop+fire kept), same allowed-action set for every seed, every agent evaluated **without** the mask. Windows are matched across algorithms by meaning ("before competence" vs "after competence"), not by update count.

| Algorithm | Early restriction | Late restriction (during mask) | Residual after removal |
|---|---|---|---|
| PPO (n=4) | Pure delay: at removal, scores like an untrained agent (delay-adjusted ≈ −1) | Collapse from 58.7 (control) to 18.5 at removal | +2.05M steps: raw +7.2, **delay-adjusted −2.0** (lost time) |
| PQN (n=4) | Delay; catches up faster than a delayed control (delay-adj. AUC −6.4*) | 3.6 vs 61.8 at removal | AUC +4.6* raw, **delay-adjusted +1.4 [−0.7, +3.5]** (lost time) |
| PQN-R (n=4) | Delay (delay-adj. AUC −2.7 n.s.) | 3.2 vs 26.7 at removal | AUC +8.6* raw, **delay-adjusted +4.3 [+1.6, +7.0]*** (small excess beyond lost time, concentrated in the first ~100 updates) |

**Reading.** Early deficits behave like a delay in all three agents, so we find **no early critical period** (pre-registered decision point 2 not met). Late deficits cause *active unlearning*: the agent collapses to roughly the score of a noop/fire-only policy. Recovery is then mostly re-acquisition. The only excess beyond lost time is in the replay agent (PQN-R), and that is what motivated asking *where* the post-removal deficit is carried.

Surgery logic (both algorithms): at the moment of removal, fork a matched control **C** (no restriction) and a damaged agent **D** from a shared history. Then cross their state components factorially and measure post-removal AUC. Each component's contribution to the deficit is the change in post-removal AUC when that component is swapped from C to D.

---

## 2. Central claim 1: in PPO, the surviving damage is in the actor

Source: `20261011-p3-surgery-move-late-analysis.md`, Phase 4 files. Movement-late, n=4. Total damage D − C = **−14.3 [−17.5, −11.2]***.

**2a. 2×2×2 actor × critic × optimizer factorial (main effects, AUC):**

| Carrier | Effect | Share of total |
|---|---|---|
| Actor params | **−12.7 [−13.8, −11.6]*** | 89% |
| Critic params | −2.3 [−6.2, +1.7] | 16% |
| Adam state | −0.1 [−2.8, +2.6] | 1% |

**2b. Transplant and reset (Phase 4; movement and the distinct fire intervention):**

| Contrast | Movement | Fire |
|---|---|---|
| Clean actor transplanted into D | **+10.9 [+7.6, +14.2]*** | **+12.9 [+1.3, +24.5]*** |
| Transplant vs control C | −3.4 [−7.5, +0.7] | −4.5 [−6.9, −2.1]* |
| D's actor reset to fresh init | **−27.2 [−39.4, −15.1]*** | **−23.4 [−35.6, −11.1]*** |
| Actor+Adam reset − actor reset | +2.8 [−7.6, +13.1] | +0.5 [−7.9, +8.9] |

- **The actor carries the damage, and the effect replicates across the two interventions.** A clean actor recovers most of the deficit: under movement, the transplant is statistically indistinguishable from the control. Under fire restriction a small residual remains (−4.5*), so the actor is the main carrier there, not the only one.
- **The damaged state is not toxic.** Resetting the damaged actor is much *worse* than keeping it, and the reset agent tracks learning from scratch (26.9 vs 28.2 at +50). The damaged actor still contains useful competence. The damage is competence that was lost or overwritten, not a scar that blocks learning. Under this protocol, resets do not help.
- **The optimizer is not a carrier in PPO**, by both tests: the factorial main effect (−0.1) and actor+Adam reset vs actor reset (+2.8 n.s., +0.5 n.s.).
- **Suppressing drift does not prevent the harm.** Freezing or anchoring the actor's left/right logits during the mask raises eval at removal by about +20, but post-removal AUC does not change (move: +1.1 [−3.4, +5.7] and −0.0 [−1.8, +1.7]). The harm is not simply the masked logits drifting.

**2c. The data channel in PPO is guidance, not a data scar.** In these Phase 4 behaviour-swap arms, the damaged learner trains on data collected by a different policy (movement, n=4):

| Behaviour policy | vs D_online |
|---|---|
| Clean actor of the same seed | **+8.3 [+5.6, +11.0]*** |
| Clean behaviour − actor transplant | −2.6 [−6.2, +0.9] (n.s.: guidance ≈ transplant) |
| Frozen self, pre-intervention data | −1.9 [−6.3, +2.4] |
| Clean actor of another seed | +2.4 [−5.6, +10.4] |
| Clean − tempered self, matched KL | **+13.4 [+3.4, +23.4]*** (fire: +13.7 [+3.8, +23.7]*) |

Collecting with a competent policy repairs the damaged learner almost as well as transplanting that policy. Replaying the damaged agent's own fixed data does not help, and off-policyness at matched KL is not the active ingredient: the *quality* of the behaviour is. On-policy data is therefore a route for repair, not a place where damage is stored.

---

## 3. Central claim 2: in PQN-R, the network carries the damage robustly; the buffer contributes only when it is long-lived

PQN-R = PQN with a uniform FIFO replay buffer (MLP 2×128 + LayerNorm, RAdam, 1-step targets). **Prespecified assignment rule:** "network" = params + RAdam state (moved together), and the target/optimizer assignment never varies with the buffer treatment. The 2×2 network × buffer surgery happens at removal (u350).

| | 100k buffer (n=12) | 1M buffer (n=8) |
|---|---|---|
| Total DD − CC | −6.7 [−8.0, −5.3]* | −5.6 [−7.0, −4.2]* |
| Parameter transfer DC − CC | **−4.6 [−6.1, −3.0]*** | **−2.6 [−3.6, −1.6]*** |
| Data transfer CD − CC | **+0.0 [−1.0, +1.0]** | −1.7 [−3.3, −0.2]* |
| **Network main effect** | **−5.6 [−6.3, −4.9]*** (84%) | **−3.2 [−4.2, −2.2]*** (57%) |
| Buffer main effect | −1.0 [−2.0, −0.1]* | −2.4 [−3.4, −1.4]* (43%) |

Notes on the table:
- **100k buffer:** turnover is fast; the interaction is −2.1 [−4.5, +0.2]. A clean network rescues more than a clean buffer: CD − DD +6.7 [+5.5, +7.8]* vs DC − DD +2.1 [+0.2, +4.0]*.
- **1M buffer:** turnover takes 244 updates, and the restriction-era share of the D buffer is 0.410.

**The damaged network trails even on an identical buffer.** Two later experiments hold the replay buffer *bit-identical* across arms and vary only the network:
- Phase 6b B − D: **−2.9 [−4.0, −1.8]*** (n=12, pre-registered). Early −3.2, mid −2.8, late −2.8; only 1/12 seeds > 0.
- Purge B − E: −1.9 [−3.0, −0.8]* (n=8, post-hoc).

**Reading.**
1. The network carrier is the robust one. It appears with both buffer sizes and in every design that isolates it. It is still present in the late window, long after any restriction-era data has been overwritten.
2. A buffer carrier appears **only when the buffer outlives the deficit**. With 100k (fast turnover), transferring the damaged buffer into a clean network costs nothing (+0.0 [−1.0, +1.0]). With 1M it costs −1.7*.
3. The 1M buffer effect does not behave like stored content. Per overwrite phase (`p6_surgery_pqnr1m_n8-phases.md`), the buffer main effect is −1.4* while the data is present (share 0.408) and **−2.9 [−4.1, −1.8]* after it is completely gone** (share 0.000). An effect that persists, and grows, after its supposed carrier has been deleted must have been written into the network while the network trained on that data. The buffer is therefore a **mediator** of network damage, not an independent store.
4. Severity confound (disclosed): with 1M, the deficit at removal is milder (−50% vs −87%). So the 57/43 split at 1M mixes buffer size with damage severity.

---

## 4. Cross-algorithm synthesis

| Question | PPO (on-policy) | PQN-R (off-policy) |
|---|---|---|
| Where does the surviving deficit live? | Actor params (89%) | Network (84% at 100k; 57% at 1M) |
| Is the critic / value side a carrier? | No (−2.3 n.s.) | Not separable (single Q-network) |
| Is optimizer state a carrier? | No (−0.1; reset +2.8 n.s.) | **Not tested separately** (bundled with params by rule) |
| Is damaged state toxic (reset better)? | No: reset −27.2* ≈ scratch | Not tested (no PQN-R reset arm) |
| Does the data channel matter? | Yes, as *guidance*: clean behaviour +8.3* ≈ transplant | Yes, through *retraining*: buffer effect only at 1M, persists after the data is gone |
| Is the damage beyond lost time? | Delay-adjusted −2.0 (≈ lost time) | Small excess, +4.3* delay-adjusted, early post-removal |

**Unifying statement.** The damage lives in the learned function. Experience is how it got there and how it is undone. In PPO, the experience channel is the behaviour policy; in PQN-R, it is the replay distribution the network is retrained on. Neither algorithm gives evidence that experience *stores* damage independently of the network. The removed data's identity, isolated at matched exposure, contributes less than 1 AUC unit (§5).

What is matched across algorithms, by design:
- deficit meaning;
- onset relative to competence;
- recovery horizon in env steps;
- eval protocol (unmasked, ε 0.05 for PQN-R);
- explicitly recorded update:env-step ratio, replay ratio and exploration schedule.

What is not matched:
- the number of gradient steps;
- the network architecture (CNN actor-critic vs MLP Q-network);
- optimizer separability.

The cross-algorithm claim is therefore qualitative (which component carries the damage, and its sign), not quantitative (the AUC scales differ).

---

## 5. Secondary, qualified finding: replay composition

All PQN-R with a 1M buffer, all with fresh unrestricted replacement transitions collected under a fixed, prespecified protocol. Params, optimizer, env state, RNG and update counter are held at the fork.

| Step | Design | Result | What it licenses |
|---|---|---|---|
| Purge (n=8) | Replace restriction-era slots vs retain | B − A **+4.6 [+2.2, +6.9]*** | Replay composition matters for repair |
| 6b (n=12) | Targeted purge vs sham matched on number + age | B − C **+2.6 [+0.9, +4.3]*** | Targeted > sham, **but** the sham's replacements are overwritten ~100 updates earlier, so this mixes identity with residence |
| 6c (n=12) | Identity × residence 2×2 | Identity at matched long residence +1.1 [−0.2, +2.5]; at short +1.1 [−0.8, +3.0]; residence +1.5 [+0.2, +2.7] (Holm p = 0.0726). Neither outcome met | Identity and residence both plausibly small and positive. Post-hoc factorial (identity p = 0.049, residence p = 0.024) is fragile |
| **6d (n=32)** | Pre-registered single primary; arms differ only in removed-transition identity (number/age of slots, replacement pool, sampled exposure, retained residence, seams all matched) | Δ = **+0.46 [−0.19, +1.11]**, 90% CI [−0.08, +1.00], TOST vs ±1.0 p = 0.0495 → **decision C (equivalence, marginal)** | Identity effect at matched exposure is < 1 AUC |

**Statement for the paper (qualified).** Changing what a damaged PQN-R network is retrained on changes how fast it repairs (+4.6* for a purge). Once replacement exposure and residence are controlled, however, which transitions were removed matters at most modestly: the 6d equivalence bound is 1 AUC, against a network carrier of 2–6 AUC.

**Qualifications to state alongside it:**
1. **Equivalence is marginal.** The 90% upper bound is 0.998 against a margin of 1.0, a slack of 0.0017. The point estimate is positive (+0.46, 19/32 seeds > 0, dz 0.25). We cannot exclude a small positive identity effect, and we do not claim the effect is zero.
2. **The SESOI of ±1.0 AUC was pre-registered** (commit a873dc0), not chosen after seeing the results.
3. **Buffer size matters.** The data channel is visible only with a 1M buffer, the condition where it outlives the deficit. With 100k it is ≈ 0.
4. **6d's dose is half the 1M surgery's.** 6d swaps 20.5% of slots on a *damaged* network; the 1M surgery's CD − CC (−1.7*) swaps the full 41% on a *clean* network, *and* there the alternative data came from a competent on-policy agent rather than a fixed collector. The two are not the same contrast and should not be pooled.
5. **Sampled exposure is matched only on average** (pool-share difference 0; restriction-era sampled ≈ 0.325M in R vs 0 in U by construction). Per-seed variation exists.
6. **Relative, not absolute.** Δ compares restriction-era data (R) with unrestricted data (U) as retained content. It cannot separate "R harms" from "U helps"; that would need a retain-both arm.

---

## 6. What we do not claim

- **No early critical period.** Early restriction is pure delay in all three agents, so the original framing ("early damage") is not supported. The damage is caused by **late deficits after competence**.
- **No claim of plasticity loss.** The evidence that recovery is mostly re-acquisition is behavioural: delay adjustment, reset ≈ scratch, transplant recovery. We did not measure plasticity directly (no dormant-unit, feature-rank or fit-new-targets probes). The PQN-R residual network gap (B − D late −2.8*) cannot yet be split into reduced plasticity vs slow re-acquisition.
- **No optimizer-state claim for PQN-R.** RAdam state moved with the params by prespecified rule. Only PPO separates the optimizer, and there it is null.
- **No generality claim across environments.** Everything is on SpaceInvaders-MinAtar (gymnax), whose dynamics differ from the original MinAtar (gymnax issue #120; disclosed). The Asterix arm stopped after onset localisation, so there is no Asterix surgery and no Asterix replication.
- **No dose-response or deficit-family generality.** There is one restriction family (action masks: movement and fire). Fire was analysed as a distinct intervention, not as a matched-severity replicate. LR was constant for PPO.

---

## 7. Threats to validity (paper "Limitations" draft)

1. **PPO sample size.** The PPO surgery and Phase 4 arms use **n=4**. The actor effect is large and tight (−12.7 [−13.8, −11.6]), but the n=4 t-intervals are fragile for the smaller contrasts (critic, optimizer, behaviour arms).
2. **Single environment, single deficit family** (see §6).
3. **Severity is not held fixed across buffer sizes** in PQN-R (−50% vs −87% at removal).
4. **Bundled network carrier** in PQN-R (params + RAdam).
5. **Greedy eval brittleness** in PQN: we use ε = 0.05 eval as primary and greedy as secondary (6d greedy +0.72 [−1.68, +3.13]).
6. **Analytic flexibility** was controlled by pre-registration (purge, 6b, 6c, 6d, each with commit hashes in LOG.md). Every post-hoc analysis is labelled (purge B − E, the 6c factorial).

---

## 8. Recommended before submission (in priority order)

1. **Extend the PPO actor/critic/optimizer surgery and the transplant/reset arms to ≥ 12 seeds.** This is the weakest link in a central claim.
2. **Retitle and reframe.** Suggested title: *"Where Does Deficit Damage Live? Actor, Network, and Replay Carriers of Post-Competence Damage in Deep RL"*, with the thesis "late deficits cause unlearning carried by the function approximator; data mediates but does not store it."
3. **Split the PQN-R optimizer from the params** (a 2-arm add-on to the 100k surgery) so that the optimizer row in §4 is filled in for both algorithms.
4. **A direct plasticity probe at removal** (e.g. fitting fresh random targets from C vs D network states), to support or drop the "not plasticity" reading.
5. **Second environment** (PQN-R Asterix replication, with onset scheduled from the control's competence threshold), if compute allows. Name it a PQN replication, not a replication of PPO Phase 4.

---

## 9. Evidence table (source of every number)

| Claim | Number(s) | Source file (`results/summary/`) |
|---|---|---|
| PPO recovery, delay-adjusted | +7.2 raw; −2.0 delay-adj. | LOG.md Phase 2/4 recovery tables |
| PQN / PQN-R recovery | +1.4 [−0.7, +3.5]; +4.3 [+1.6, +7.0]* | `p5_recovery_move-analysis.md` |
| PPO factorial | −14.3; actor −12.7; critic −2.3; opt −0.1 | `20261011-p3-surgery-move-late-analysis.md` |
| PPO transplant / reset (move) | +10.9; −27.2; −3.4; +2.8 | `20261011-p4-phase4-move-late-ppo-SpaceInvaders-MinAtar-9a789ef-analysis.md` |
| PPO transplant / reset (fire) | +12.9; −23.4; −4.5; +0.5 | `20261011-p4-phase4-fire-late-ppo-SpaceInvaders-MinAtar-8456a9b-analysis.md` |
| PPO behaviour / drift | +8.3; −2.6; −1.9; +2.4; +13.4; +1.1; −0.0 | Phase 4 move file (fire: +13.7) |
| PQN-R surgery 100k | −5.6; −1.0; −4.6; +0.0 | `p6_surgery_pqnr_n12-analysis.md` |
| PQN-R surgery 1M | −3.2; −2.4; −2.6; −1.7 | `p6_surgery_pqnr1m_n8-analysis.md` |
| 1M overwrite phases | −1.4 carried; −2.9 gone | `p6_surgery_pqnr1m_n8-phases.md` |
| Purge | +4.6; B − E −1.9 | `p6_purge-analysis.md`, `p6_purge-posthoc.md` |
| Targeted vs sham | +2.6; B − D −2.9 | `p6b_targeted-analysis.md` |
| Identity × residence | +1.1; +1.1; +1.5 (Holm 0.0726) | `p6c_2x2-analysis.md`, `p6c_2x2-factorial-posthoc.md` |
| Matched-identity equivalence | +0.46 [−0.19, +1.11]; TOST 0.0495 | `p6d_idswap-analysis.md` |

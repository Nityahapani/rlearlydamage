# Literature Review: Critical Periods in Deep RL ("Data Scars vs. Weight Scars")
*Search done: 11 Oct 2026. Covers arXiv, OpenReview, conference proceedings, and recent TMLR acceptances (27 Sep – 11 Oct 2026).*

---

## TL;DR verdict
| Claim | Status |
|---|---|
| "We are the first to show critical periods exist in RL agents" | ❌ **Already claimed** (Park et al. 2021; de Kleijn et al. 2022; Park et al. 2022) |
| "A fresh network trained on an old agent's buffer recovers quickly" | ⚠️ **Partly done** (Nikishin et al. 2022, primacy bias, *without* a temporary deficit) |
| "Early non-stationarity leaves a permanent mark on representations" | ⚠️ **Partly done** (Igl et al. 2021, ITER) |
| Systematic onset × duration deficit sweep in deep RL, many seeds, several deficit types | ✅ **Open** |
| Full factorial "state surgery": {weights, optimizer state, data/buffer} × {damaged, clean} | ✅ **Open** |
| Data scars in **on-policy** RL (no buffer), separated by swapping the behaviour policy | ✅ **Open** |
| RL-specific deficits (actions, reward, exploration) vs. perceptual deficits (Achille-style) | ✅ **Open** |

**Bottom line:** the project is still novel if you reframe it. The contribution is a **mechanistic decomposition** of where early damage is stored, not "finding" that critical periods exist. Drop any "first critical period in RL" claim.

---

## 1. Closest prior work (you must cite and differentiate all of these)

### A. Critical periods in RL agents (closest by name)
| Paper | What they did | Gap you fill |
|---|---|---|
| **Park et al. 2021**, *Toddler-Guidance Learning: Impacts of Critical Period on Multimodal AI Agents* (ICMI '21) — [arXiv 2201.04990](https://arxiv.org/pdf/2201.04990) | Timing of reward "guidance" in ViZDoom; one guidance onset worked best | Curriculum timing only. No deficit-window sweep, no mechanism, few seeds |
| **de Kleijn, Sen & Kachergis 2022**, *A Critical Period for Robust Curriculum-Based Deep RL…* (Topics in Cog. Sci.) — [link](https://onlinelibrary.wiley.com/doi/10.1111/tops.12595) | PPO robot arm: an energy penalty introduced too early or too late hurts ("Goldilocks" onset) | Cognitive-science framing, one task, no data/weight separation |
| **Park et al. 2022**, *On the Importance of Critical Period in Multi-stage RL* — [arXiv 2208.04832](https://arxiv.org/pdf/2208.04832) | Formalizes critical period as the stage-transition time that minimizes convergence time; ViZDoom | Defined in terms of guidance timing. No mechanism, no data/weight split |

➡️ **How to differentiate:** these papers study *when to add helpful guidance*. You study *where harm from a temporary deficit is stored*. Use the Achille-style deficit protocol plus causal state surgery.

### B. Primacy bias / overfitting to early data (closest by mechanism)
| Paper | Key finding | Relation |
|---|---|---|
| **Nikishin et al. 2022**, *The Primacy Bias in Deep RL* (ICML) — [PDF](https://evgenii-nikishin.github.io/data/RL_primacy_bias_RLDM2022.pdf) | A **fresh agent trained on the old agent's replay buffer** recovers fast. So primacy bias is "a failure to learn from data, not to collect it." Resetting the buffer prevents recovery | ⚠️ **This is one cell of your factorial**, done in a no-deficit, high-replay-ratio setting. Their result predicts "weight scar dominates" for off-policy agents. Your job is to test whether that holds under temporary deficits and in on-policy RL |
| **Igl et al. 2021**, *Transient Non-stationarity and Generalisation in Deep RL* (ICLR) — [OpenReview](https://openreview.net/forum?id=Qun8fv4qSby) | Early non-stationarity leaves a permanent "memory effect" in representations that hurts generalisation; ITER distils into fresh networks | Weight-scar evidence for *generalisation*. No timing curve, no data-scar arm |
| **Lyle et al. 2025**, *What Can Grokking Teach Us About Learning Under Nonstationarity?* — [arXiv 2507.20057](https://arxiv.org/html/2507.20057v1) | Re-warming the effective learning rate overwrites early features; helps with primacy bias in RL | A possible "rescue" intervention: test whether it closes the critical period |
| **Ash & Adams 2020**, *On Warm-Starting Neural Network Training* (NeurIPS) — [PDF](https://papers.neurips.cc/paper_files/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf) | Warm-starting hurts generalisation even after a few epochs; shrink-and-perturb helps | Supervised analogue of weight scars |

### C. Critical periods in supervised / theory work (foundation)
- **Achille, Rovere & Soatto 2019**, *Critical Learning Periods in Deep Networks* (ICLR) — [arXiv 1711.08856](https://arxiv.org/abs/1711.08856). The protocol you'll copy: sliding deficit windows, Fisher information. Key result: **low-level deficits (blur) are permanent, while high-level ones (flips, label permutation) are recoverable.** Asking whether this pattern holds or reverses in RL is a strong hook.
- **Kleinman et al. 2024**, *Critical Learning Periods Emerge Even in Deep Linear Networks* (ICLR) — [arXiv 2308.12221](https://arxiv.org/html/2308.12221v2). Analytical model, with depth dependence. Lets you predict that deeper agents have stronger critical periods.
- Kleinman et al. 2023: critical periods for multisensory integration.

### D. Early training & seed variance in RL
- **Bjorck, Gomes & Weinberger 2022**, *Is High Variance Unavoidable in RL?* (ICLR) — [arXiv 2110.11222](https://arxiv.org/abs/2110.11222). Variance arises **early**, from outlier runs that never recover. **Init and initial exploration are *not* to blame**; numerical instability and saturation are. Supports the "early fate" story and gives you a baseline explanation to rule out.
- **Xu et al. 2024**, *DrM* (ICLR) — [arXiv 2310.19668](https://arxiv.org/abs/2310.19668). Early motor *inactivity* correlates with dormant neurons. This is a link between early behaviour (data) and weights that your factorial can test causally.
- *Predicting DNN Training Outcomes from Early Telemetry* (2026, supervised) — [arXiv 2608.03709](https://arxiv.org/html/2608.03709). Your "predict failing seeds early" analysis is the RL analogue. Cite it and differentiate.

### E. Optimizer state as a hidden carrier (new angle, 2026)
- **Xu 2026**, *Stored in Optimizer State, Valued by Later Training* — [arXiv 2608.20442](https://arxiv.org/abs/2608.20442). Adam's **first moment carries a perturbation after its source is removed**, shown with "state surgery" (transplanting moments).
- *Delayed Optimizer-State Transport…* (2026) — [arXiv 2608.24593](https://arxiv.org/html/2608.24593).
- **Dohare et al. 2023**, *Overcoming Policy Collapse in Deep RL* — [OpenReview](https://openreview.net/forum?id=m9Jfdz4ymO). Adam defaults cause collapse under non-stationarity.

➡️ **Use:** add **optimizer state as a third carrier** (weights / optimizer moments / data). Nobody has done this in RL. Nikishin found resetting the optimizer had "essentially no impact," but that was without a deficit.

### F. Methodological precedent for decoupling data from learner
- **1000-Layer Networks for Self-Supervised RL** (2025/26) — [arXiv 2503.14858](https://arxiv.org/html/2503.14858v5). A "collector" network and separate "learners" that train only on the collector's buffer separate exploration from expressivity. **Cite it as precedent for your design**, and note it addresses scaling, not timing.

### G. Plasticity-loss background (cite briefly)
- Klein et al. 2024, *Plasticity Loss in Deep RL: A Survey* — [arXiv 2411.04832](https://arxiv.org/abs/2411.04832). Lists "early overfitting" as a cause. Its call to "experimentally isolate specific contributions" is exactly what you do.
- Sokar et al. 2023 (dormant neurons); Lyle et al. 2022/2023 (capacity loss); Dohare et al. 2024 (Nature, loss of plasticity); *A Study of Plasticity Loss in On-Policy Deep RL* 2024 — [arXiv 2405.19153](https://arxiv.org/html/2405.19153v2) (mentions critical periods only in related work).

---

## 2. Reviewer attacks to anticipate
1. **"This is just primacy bias."** Primacy bias is overfitting to *normal* early data at high replay ratios, with no timing curve. You study **temporary deficits** with an **onset × duration sensitivity curve** and a causal decomposition. Show the sensitivity curve peaks and then falls; primacy bias predicts no such window.
2. **"Nikishin already showed a fresh net on the old buffer recovers."** True for one setting. You test the full factorial, add optimizer state, add on-policy RL, and look for **regimes where the data scar dominates** (e.g., exploration-limiting deficits in sparse-reward tasks). If the weight scar always dominates, that's still a clean confirmatory result across settings.
3. **"Park et al. already found critical periods in RL."** They studied guidance timing in 1–2 tasks. Yours is a systematic, mechanistic, many-seed study.
4. **"It's just learning-rate decay."** Achille ruled this out with a constant LR. Do the same: run constant-LR controls.
5. **"Too few seeds / cherry-picked games."** Use JAX (PureJaxRL, gymnax/MinAtar, Craftax-Classic) with 30+ seeds and rliable statistics.

---

## 3. Refined contribution statement (draft)
> We study critical learning periods in deep RL using a sliding-window deficit protocol across perceptual (observation noise/blur), behavioural (action masking), and reward (delay/corruption) deficits. Unlike supervised learning, an RL agent's early experience shapes both its parameters *and* its future data. Using **state surgery** at the moment the deficit is removed, transplanting network weights, optimizer moments, and replay data (or, for on-policy agents, the behaviour policy) independently, we decompose lasting damage into **weight, optimizer, and data scars**. We find [regime-dependent results], show that early plasticity metrics (dormant ratio, feature rank, Fisher trace) predict which windows are critical, and test whether known plasticity interventions (resets, shrink-and-perturb, LR re-warming) close the critical period.

**Title options**
- *Scars in the Buffer or in the Weights? Decomposing Critical Learning Periods in Deep RL*
- *Where Does Early Damage Live? Weight, Optimizer, and Data Scars in Reinforcement Learning*

---

## 4. Remaining checks before committing
- [ ] Google Scholar "Cited by" for **Achille et al. 2019** filtered by "reinforcement", 2025–2026.
- [ ] Google Scholar "Cited by" for **Nikishin et al. 2022**, looking for any buffer/weight factorial with deficits.
- [ ] OpenReview TMLR **Under Review** tab, plus ICLR 2027 submissions (public after the deadline): search "critical period", "early training", "primacy".
- [ ] arXiv search: `"critical period" AND ("reinforcement learning" OR PPO OR DQN)` from 2025 onward.
- [ ] Set an arXiv alert on these keywords while you run experiments.

# Response to the clarity and major-weakness review

This is a **presentation and claim-scope revision, not a new experiment**. The primary registration, core runner/PPO/composition source, raw data, statistical analysis and numerical conclusions are unchanged. The substantive generalizability and mechanism limitations remain limitations; the revision does not claim to solve them with wording.

## Main changes

| Review concern | Change and location |
|---|---|
| Long, experiment-heavy narrative | Replaced the chronological 49-page manuscript with an **8-page main paper including references** and a separate **45-page supplement**. The main paper follows reference framework → one registered study → results → evidence hierarchy → reproduction → limitations. Every earlier scientific table/figure and its label remains in the supplement; redundant motivation/discussion was removed. |
| Broad overall claim | Retitled to **Actor Checkpoint Restoration in Asterix PPO: A Reference-Aware Study of Recovery Cost**. The abstract, opening question and conclusion all identify the named intervention, setting and finite horizon. The research record is not presented as a general RL recovery law. |
| Undefined terminology | **Table 1 / Section 2.1** defines Tier I, Tier II, pause reference, native recovery cost, state surgery and Shapley allocation. Section 2.3 separates background-specific swaps, factorial effects, Shapley allocation and the ratio of mean repair effects. |
| Unclear AUC scale | **Section 2.2 / equations 1–3** gives the exact arithmetic mean of 20 recovery evaluations at +10…200, excluding removal. Units are evaluation episode-return points—not an unnormalised integral, training reward lost or recovery time. The −1.72 example and prior-score calibration of the .50/.30 practical margins explain scale. Different reward scales, evaluation policies and environment-step grids prevent naive cross-task/algorithm comparisons. |
| Heterogeneous evidence strength | **Table 4 / Section 5** distinguishes strongest registered support, context-specific support, exploratory/mixed/descriptive evidence, narrow practical bounds, unresolved mechanisms and unvalidated prospective utility. Four-seed studies, pooled pilot+new IDs, exploratory PQN-R results and incomplete Freeway are explicit. Registration is not equated with a successful outcome. |
| No paper-wide multiplicity control | Section 5 states exactly that: the focused 32-seed compound is one separate alpha=.01 claim; only five Phase-13 primary claims share that phase's Holm family. Older families, exploratory analyses and interim looks retain their own scopes. There is no retrospective paper-wide guarantee or selected-task prevalence estimate. |
| Context-sensitive causal interpretation | Sections 2–4 distinguish historical clean pause H, the matched onset learner under natural D context, and extra-progress clean C. Equality at construction is not future-context neutrality. The supplement also replaces overly strong legacy onset-reference and PQN-R wording with explicit procedure/background-specific interpretations. |
| Mechanisms not established | **Table 4** retains the narrow replay-identity bound (±1 only, ±.75 fails), inconclusive label-exposure primary/equivalence, failed CartPole stronger compound and failed frozen prospective KL utility. They are not converted into absence claims or rescued by secondary contrasts. |
| Overinterpreting 91.8% | **Equation4 / Sections 2.3 and4** defines `mean(actor rescue) / −mean(native historical-pause cost)`. It is not a Shapley value, mean seedwise percentage, fraction of damage stored in actor parameters, or universal restoration rate. **Table 3** retains the small negative residual and the failed matched-state practical-equivalence test. |
| Reproducibility entry points | **Section 6 / Table 5** directly links protocol, frozen runner/composer, donor construction, state-identity assertions, registered analysis, independent verifier, raw data and new mechanically verified 32-row CSV/index. The detailed [primary reproduction guide](primary_asterix_reproduction.md) links all eight raw groups and manifests, describes exact field orientations, and distinguishes full recomputation from cell resumption. |
| Conclusion limitations | **Section 7**, including its final paragraph, explicitly denies proof of irreversible damage, preserved plasticity on new tasks, or a unique internal representation of impairment. It also states the single game/algorithm/impairment/horizon, context dependence, limited evaluation pools, lack of paper-wide control and lack of a validated automatic deployment rollback policy. |

## Important unresolved artifact limitation

**Full serialized donor/runner checkpoint trees were not retained.** The revision does not pretend to provide missing checkpoint downloads. Available P/C/D hashes cover complete **four-seed batched numerical state trees**, not per-seed checkpoint blobs. The guide links the precise construction code and recorded integrity assertions and explains how the frozen runner regenerates donors in memory. A fresh tag recomputes cells; using the original tag can resume already-complete cells after matching fingerprints.

Thus archived curves support exact statistical reanalysis; a new rollout-level reproduction still requires computation. Retrospective fingerprints cannot recreate omitted arrays or independently recheck unarchived full CDD states. This is now explicit in the main text, supplement, guide, index and README.

## Scientific invariants and validation

- No new training, new cohort, extra setting, endpoint substitution, evaluation-key selection or altered primary analysis.
- Unchanged 32-seed result: compound p=3.319136731810534e−8; 32/32 native-negative and 32/32 rescue-positive; repair ratio 91.8%, paired-bootstrap 95% interval 86.3–97.5%.
- Small historical residual and failed matched-state ±.30 equivalence remain, along with the smaller nonzero fresh-Adam critic estimate.
- The separate raw-row verifier reproduces all four primary effects, historical alignment, shifted p-values, 95/99% intervals, consistency and source/data hashes. Protocol/runner/PPO/composition still match registration 6262b2b byte for byte.
- The new deterministic packaging checks all 352 requested seed/cell curves, requested second-key curves, 32 paired rows, eight group manifests and all existing data hashes. It generates an index/CSV, not a new inferential result.
- Two new packaging tests and two existing synthetic Phase14 analysis tests pass. Both PDFs compile with resolved citations/cross-references and no overfull boxes or oversized floats. Key pages and direct artifact links are inspected.

The response improves clarity and makes limitations auditable; it does not turn one strong intervention result into evidence for a universal mechanism.

> **Historical scientific release (74c6ceb).** This page records the original 49-page presentation and its PDF hash. The current presentation has a concise main paper plus a full supplement; numerical results and registered rules are unchanged. See [presentation response](response_presentation_review.md). The original PDF is [permanently available at 74c6ceb](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/paper/main.pdf).

# One focused result: a past actor checkpoint repairs most native recovery cost

**All 32 planned new seeds complete; one registered practical-effect compound passes.** No new setting/endpoint search, interim primary analysis, N expansion or prior-seed pooling. Asterix PPO, one reward-sign-corruption window, one finite recovery horizon.

## Primary result

Compound p = **3.3191367e-08**, alpha .01. The p-value is the maximum of the four required one-sided paired-t practical-effect tests, not their minimum. Seed consistency is an additional required filter.

| component | mean [95% CI] | mean [99% CI] | required mean | shifted p |
|---|---|---|---|---|
| native_AUC_harm | -1.722 [-1.940, -1.504] | -1.722 [-2.015, -1.429] | <−.50 | 6.0214385e-13 |
| native_late_harm | -1.187 [-1.444, -0.930] | -1.187 [-1.533, -0.841] | <−.30 | 3.3191367e-08 |
| onset_actor_rescue | +1.580 [+1.371, +1.790] | +1.580 [+1.298, +1.862] | >+.50 | 4.7890384e-12 |
| reciprocal_actor_transfer | -1.636 [-1.828, -1.445] | -1.636 [-1.894, -1.379] | <−.50 | 1.3904017e-13 |

All 32 native AUC costs are negative and all 32 onset-actor rescues positive. Reciprocal transfer is negative in 32/32; the late comparison in 31/32. Exact sign and paired-bootstrap sensitivities, all individual seed effects, and simultaneous one-sided 95% Bonferroni bounds are preserved in the analysis JSON. No episode/checkpoint pseudoreplication. Inference pairs training seeds conditional on the registered primary evaluation batch.

The ratio-of-paired-means fraction repaired is **91.8%**, paired-bootstrap 95 interval **86.3–97.5%**. It is not an information share or unique natural decomposition.

## What is controlled

- Donor actor is the shared **pre-impairment** checkpoint, not the clean fork after 50 additional updates. No clean-run progress is imported.
- Main swaps retain the exact natural D removal-time environment, observations, future RNG, counters and non-actor learner fields. All-D equals the full untouched D runner-state tree. After intervention, all arms learn normally; changed experience/critic/moments are downstream effects.
- Native cost is compared with the earlier **unbroken clean trace**. The matched onset-state CCC reference is separately measured and never declared bit-identical to historical pause.
- All 32 unused IDs 1200–1231 and all 11 requested cells were fixed by public preregistration 6262b2b. Earlier B8 is prior evidence only. No low-competence seed is filtered.

## Prespecified secondary checks: all retained, not primary substitutes

- Fresh-Adam actor main effect: -1.594 [-1.767, -1.421]; actor rescue: +1.626 [+1.420, +1.832]. Fresh-Adam critic main effect: -0.129 [-0.229, -0.029]; it is smaller, not absent.
- Reset-world actor rescue: +1.515 [+1.308, +1.722]. Reset-minus-natural rescue: -0.065 [-0.248, +0.118]. Nonsignificance of a context difference is not literal context neutrality.
- Independently frozen evaluation key 54321: native AUC -1.599 [-1.762, -1.436], late -1.070 [-1.315, -0.825], rescue +1.398 [+1.227, +1.569]. Neither key selection nor key pooling enters the primary.
- Repaired-minus-historical-pause residual: -0.141 [-0.246, -0.037]; secondary TOST±.30 p=0.00210879 **passes its practical margin**, not literal zero. A small negative residual remains.
- Repaired-minus-matched-state-pause residual: -0.231 [-0.370, -0.091]; TOST±.30 p=0.15991 **fails**. No full-restoration claim.
- Matched state-pause minus historical-pause reference: +0.089 [-0.035, +0.213]; secondary TOST±.30 p=0.000791269. A practical reference agreement result is not identity.

## Verification and availability

- Independent scalar-row arithmetic audit reproduces every raw paired primary value, historical lag, shifted t-CDF p-value,95/99% interval, consistency count, source/data hash and final verdict (`scripts/verify_p14.py`).
- Preregistered scientific protocol, runner, PPO and surgery files remain byte-identical to 6262b2b. Source/analysis/code history is public. All 597 catalogued data files hash-match;191 surviving NPZ files remain versioned.
- Full software suite: **26 passed**, after scientific training exited (no parallel JAX workload). The 61 warnings concern pinned-library future casts/deprecated APIs, not failed tests; no unrecorded scientific-code patch was made.
- Eight sequential four-seed groups ran 16:33:04–23:07:37 UTC on 2026-10-16: 6h34m33s, two CPU cores, about 2GB, no GPU. The first task clock was 16:22; final deadline 00:18UTC. No further scientific training was launched.
- All 11 seed-level mean score curves, requested second-key curves and donor fingerprints are archived in eight group JSONs. Full transient runner-state trees are not archived; reproduce them from the recorded fixed seed group, source and environment. Availability/reset limitations for older phases remain explicit.

## Interpretive boundary

This is one selected-condition, intervention-specific Tier-II actor-repair result. It does not establish unique naturally encoded damage, literal full restoration, universal actor dominance, an automatic deployment rollback policy, new-task learning capacity or irreversible loss. Phase14 alpha .01 is a separate fixed claim, not retrospective error control over all earlier research. The old eligible-setting count is not enlarged by this repeat condition.

Primary artifact: [`p14_single_result-analysis.json`](../results/summary/p14_single_result-analysis.json). Final paper: [`paper/main.pdf`](../paper/main.pdf). All older failed/inconclusive primary results remain in the manuscript, log and inference map.

## Final manuscript deliverable

- **49 pages**, 4066200 bytes. New central result is Section6, Table8 and Figure4 (pages16–17); the abstract, introduction, inference map, discussion and conclusion are aligned.
- PDF SHA256: `1fbde29d0f30cf41c271490622dc3eabc9b7e4d274ab0816576cd77a6391de24`.
- All33 cited keys exist in the compiled bibliography; no unresolved references/outcome slots, overfull boxes or oversized floats. The new figure and key pages were visually checked at page scale.
- Tectonic0.15 automatic bibliography-rerun detection emitted an internal consistency warning despite unchanged `.bbl`/`.aux` hashes. Three explicit reruns produce a stable, fully resolved final PDF; this command is documented in README. Underfull provenance-table line warnings are typographic, not missing content.
- Full scientific protocol and inference remain unchanged; no post-primary new training or endpoint substitution. All597 catalogued file hashes and all352 requested seed/cell curves were audited.

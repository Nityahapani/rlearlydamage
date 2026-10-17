# Expanded manuscript and exact-state reproduction plan — 2026-10-17

User budget: 7–8 hours, first verified clock05:47:42UTC. Target handoff by13:30UTC, no later than13:45UTC. This is not permission to change the completed scientific primary.

## Manuscript

Replace the eight-page presentation with a fuller argument and an **integrated, self-contained reproducibility appendix**. Keep the full historical supplement separately. Add substantive detail, not padding:

1. Articulate the advance relative to later-clean-actor transplantation, fresh resets, checkpoint restoration and plasticity studies without claiming a universal recovery principle or priority over all transplantation work.
2. Define a single state notation and diagram connecting observed onset/clean/impaired donors, field-level interventions, historical reindexing, matched learner reference and downstream trajectories.
3. Explain that historical pause is a constructed benchmark, not an identified literal paused-runner counterfactual. Actor rescue is a total downstream effect, not a measurement of uniquely stored impairment.
4. State exactly that primary t/paired-bootstrap intervals vary training seeds **conditional on the fixed shared evaluation batch**. They do not integrate over redrawn evaluation worlds/policy RNG, task selection, hyperparameters or horizons. The second key is a fixed secondary check, not doubled N or a pooled evaluation population.
5. Derive and document the **joint paired ratio bootstrap**: resample the same seed indices for rescue numerator and native-cost denominator, preserve covariance and all shared trajectories;20,000 draws, seed140033, untruncated ratio, linear2.5/97.5 percentiles. Primary numerical artifact remains unchanged.
6. Give a result-by-result reproducibility matrix: archived means permit arithmetic reanalysis; newly reconstructed numerical states (if exact checks succeed) permit loading without prehistory; continuations/new evaluation randomness still require simulation. Static functions come from pinned source, not executable checkpoint pickles.

## Fixed exact reconstruction, not an additional empirical confirmation

- Same original32 training IDs1200–1231, same eight groups of four, all11 original cells and both originally specified evaluation keys. No new setting, architecture, impairment, horizon, primary contrast, independent seed or hypothesis.
- Scientific protocol/runner/PPO/composer remain byte-identical to registration6262b2b. Frozen primary JSON and original raw summaries are immutable.
- Instrument the frozen runner through a separate wrapper. Before continuing, P/C/D must match the original batched numerical fingerprints exactly. Serialize all dynamic runner fields with Flax state_dict/msgpack; restore them before the frozen runner continues. All cell curves and original checks must match exactly.
- Reconstruct and retain individual fixed-batch evaluation returns as an audit of the originally archived64-episode means. These are reconstructed deterministic returns, not previously retained episode arrays or new independent evaluation draws.
- Fixed additional validation: first original group1200–1203, native DDD and restored CDD, fresh process/template loading from the written bundle, both evaluation keys and every21 checkpoints. This validates actual saved-file use, not just a serialization hash.
- A donor fingerprint, episode-mean, trajectory or loader mismatch stops the schedule and is reported as a discrepancy. No outcome-driven replacement, source tuning or seed substitution. Technical failures/partial completion remain visible.
- New reconstruction tags are excluded from the original primary loader. Never pool duplicate seeds or call this N=64. Original inference and91.8% interval are unchanged.
- Use an isolated clean worktree and one JAX worker at a time. Paper editing in main must not alter the frozen run source. No concurrent JAX tests or heavy analysis while the worker is at its memory peak.
- Operational no-new-group cutoff12:20UTC; hard stop13:20UTC, preserving final manuscript/QA time. The historical preregistration dates are not edited.

## Archive provenance

Any published states will be labelled **reconstructed2026-10-17**, not falsely described as retained at the original experiment. P/C/D hashes cover four-seed batches. The archive stores complete dynamic numerical state, optimizer/environment/observation/RNG/counters/auxiliaries; static apply/optimizer functions require the pinned source/template. Original/independent reconstructed episode arrays are distinguished from original mean curves.

Large bundles may be published as public repository release assets with SHA256 indexes, keeping the source repository small. A bundle is marked complete only after the predefined checks pass. If reconstruction is incomplete or not exact, the paper retains that limitation rather than asserting that missing states have been recovered.

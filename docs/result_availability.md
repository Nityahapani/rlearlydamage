# Result archive and completion status

This inventory records saved outputs; it does not invent missing experiments. All saved run summaries, manifests, analyses and available NPZ arrays are committed. See `result_inventory.json` for file hashes and requested/observed arms.

## Phase 12

- Complete at planned n=8: G1 CartPole flip, G2 Acrobot flip, G3 actuator noise, G5 observation noise, G6 extinction, G7 PQN-R flip.
- G4 Freeway: four complete factorial seeds (0–3); partial arms for seeds 4–5 are archived but excluded from factorial aggregation. Seeds 6–7 did not run. The four-seed analysis is descriptive, never counted as a confirmatory success.
- Optional independent-seed replications H1 (600–607) and H2 (700–707) did not start.

## Raw data and large generated states

Per-seed score curves and raw derived predictor values are preserved in summary JSON/Markdown files. All NPZ arrays that survived the sandbox reset are included, including removal-parameter arrays and evaluation-state banks. Full runner-state pickles and some redundant onset/short-duration weight dumps had been placed in a temporary snapshot-excluded cache; they did not survive the reset. They are not falsely described as archived. Regenerate them from the saved seed/configuration when needed.

## Source hashes after the sandbox reset

Some original local-only Git objects were lost before they could be pushed. Original hashes in run manifests are retained, not rewritten. The available public pre-run revision `79700c2` has the same four Phase-12 training-file contents as the recovered source; changes in later local commits were analysis, documentation or result artifacts. This is a reconstruction aid, not a claim that the unavailable whole Git trees were recovered.

| training file | matches public 79700c2? |
|---|---|
| `src/rled/agents/ppo.py` | True |
| `src/rled/agents/pqn.py` | True |
| `scripts/surgery_ppo.py` | True |
| `scripts/pqn_surgery.py` | True |

Recorded source commits unavailable in the restored clone: `01b2d55`, `343675e`, `4254b3e`, `685fd17`.

A deterministic rerun must preserve each manifest’s seed group, configuration, checkpoint grid and package versions. The original experiment code and the published recovered source are both identified; no missing outcomes are imputed.

## Phase 13 follow-up (all planned cohorts complete)

The next registered phase completes A/B/C/E at 8 independent seeds each and prospective D at 16. Phase-12 H1/H2 were not retroactively completed: the formerly unused IDs are used in expanded Phase-13 protocols and the cohorts are not double-counted as additional tasks.
The new inference family is fixed at five claims, with registered compound tests and Holm correction. Only B (Asterix pause) and C (actuator-noise gap) meet it. A has clear finite-window native harm but its stronger compound dominance test does not pass; E and D retain their null/inconclusive primary outcomes.
Natural D identity and controlled washout checks, all cell curves, frozen predictor coefficients, all prospective feature rows and surviving checkpoint arrays are archived. No fresh-seed result is inferred from an unrun or partial cohort.

## Phase 14: one fixed independent actor-repair confirmation

All32 unused training seeds1200–1231 complete all11 requested cells. The earlier eight-seed Asterix cohort is prior evidence only and is not pooled into the new primary. This is one selected-condition confirmation, not a new task or an unconditional prevalence estimate.
The single registered practical-effect compound at alpha .01 is met (maximum component p=3.3191367e-08), including the prespecified seedwise consistency rule. This is a separate Phase14 error-control scope, not retrospective paper-wide control.
All native, onset, reciprocal, common-fresh-Adam and reset-world curves, requested second-key evaluations, exact donor fingerprints and source hashes are archived in the group JSONs. Full transient runner-state trees are not archived. Reproduce using the preserved fixed seed/configuration/source and pinned packages.
Source: `prereg/phase14.json`, `scripts/p14_actor_repair.py`, `scripts/analyze_p14.py`, `results/summary/p14_single_result-analysis.json`. No interim primary, seed expansion, endpoint/key selection or secondary-primary rescue.

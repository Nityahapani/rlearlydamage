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

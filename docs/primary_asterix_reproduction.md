# Primary Asterix study: direct reproduction and artifact guide

This guide concerns **one selected Asterix PPO condition**, training seeds 1200–1231, not the legacy generalization panel. It packages the already-completed experiment; no new training, contrast, primary rule or outcome is introduced by this revision.

## 1. Start here

- [32-row per-seed primary CSV](../results/summary/p14_primary_per_seed.csv): every paired effect plus its exact raw file and run manifest.
- [Machine-readable artifact/checkpoint-availability index](p14_primary_artifact_index.json): hashes, eight groups, recorded runtime assertions and batched donor fingerprints.
- [Full frozen primary and secondary analysis](../results/summary/p14_single_result-analysis.json) and [readable report](../results/summary/p14_single_result-analysis.md).
- [Compact main paper](../paper/main.pdf) and [complete technical supplement](../paper/supplement.pdf). The main glossary and AUC definition explain the estimand before the result.

Scientific release commit: `74c6ceb9eb75f3727137a98b42ec7caa685dc325`. Protocol/runner/PPO/composition bytes still match public registration `6262b2b`. This presentation revision does not replace that frozen evidence.

## 2. Exact source entry points

| Reproduction question | Frozen source |
|---|---|
| What was registered? | [Protocol: seeds, cells, practical thresholds, one alpha=.01 compound, secondary checks](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/prereg/phase14.json) |
| Where are donors created? | [Runner lines 61–84: train to 390, fork to 440, capture P/C/D hashes, create onset learner reference](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/scripts/p14_actor_repair.py#L61-L84) |
| Which fields are transplanted? | [PPO composition lines 62–78](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/src/rled/surgery.py#L62-L78), then [natural-context assembly](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/scripts/p14_actor_repair.py#L89-L104) |
| What state identity is actually checked? | [Full-tree equality function](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/scripts/p14_actor_repair.py#L34-L37) and [runtime all-D/native and all-C/clean assertions](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/scripts/p14_actor_repair.py#L96-L104) |
| How is historical pause aligned? | [Analysis lines 70–84](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/scripts/analyze_p14.py#L70-L84): reconstruct clean history, shift50 updates, exclude removal |
| What is the registered decision? | [Shifted one-sided tests and consistency](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/scripts/analyze_p14.py#L17-L29), [four-part maximum-p rule](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/scripts/analyze_p14.py#L81-L91) |
| What does 91.8% mean? | [Ratio of mean rescue to minus mean native cost; paired bootstrap](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/scripts/analyze_p14.py#L101-L113) |
| Can the arithmetic be checked separately? | [Independent raw-row/paired-t/source-hash verifier](https://github.com/Nityahapani/rlearlydamage/blob/74c6ceb9eb75f3727137a98b42ec7caa685dc325/scripts/verify_p14.py) |

## 3. Donor checkpoints: what exists and what does not

**Full serialized donor/runner checkpoint trees were NOT retained. There is no downloadable per-seed P/C/D checkpoint archive.** We link the construction code and available fingerprints rather than imply otherwise. Hashes cannot recover omitted arrays.

- **P:** the shared onset runner at update 390. Its actor parameters are the main repair donor. It contains no progress from the following 50 clean updates.
- **D:** the impaired runner at update 440, after removal. It supplies the natural world/observations/future RNG/counters and all other main-repair learner fields.
- **C:** the unimpaired fork at 440. It helps supply the unbroken clean history, but its extra-progress actor is **not** the primary repair donor.
- Every `state_fingerprints.P/C/D` value hashes the **complete four-seed batched runner tree**, using leaf dtype/shape/bytes in deterministic tree order. These are **not individual seed hashes**. Reproduce the exact four-seed batch shape.
- Raw `checks.DDD_equals_untouched_natural_D` and `checks.CCC_equals_untouched_natural_C` record actual full-tree equality assertions at construction. The generic composer returns a reference-context runner; the Phase14 runner takes its constructed `train_state` and explicitly wraps it in natural D context. The CDD repair is built by this field assignment; there is no separate archived full CDD runner tree for independent retrospective field inspection.
- All-D identity does not prove every possible surgery is context-neutral. The primary repair shares removal-time non-actor fields by construction; subsequent behaviour, data, critic and optimizer state can diverge.

The runner regenerates donor states in memory. A fresh-tag recomputation below can recheck their hashes; it does not itself serialize those states. This missing checkpoint archive is a reproducibility limitation and is stated in both PDFs.

## 4. Eight raw group artifacts

Each JSON has all 11 requested cells, 21 checkpoint means per seed (removal plus 20 recovery checkpoints), fork evaluations, requested second-key evaluations, protocol, fingerprints and runtime checks. Each listed seed occupies its corresponding row of every `cells[...].eval_return_mean` matrix. The `fork_eval` orientation is checkpoint-by-seed instead.

| Training seeds | Raw summary | Original run manifest |
|---|---|---|
| 1200–1203 | [JSON](../results/summary/20261016-p14-actor-repair-s1200to1203-Asterix-MinAtar-a39d1bc.json) | [manifest](../runs/20261016-p14-actor-repair-s1200to1203-Asterix-MinAtar-a39d1bc.json) |
| 1204–1207 | [JSON](../results/summary/20261016-p14-actor-repair-s1204to1207-Asterix-MinAtar-bb164a1.json) | [manifest](../runs/20261016-p14-actor-repair-s1204to1207-Asterix-MinAtar-bb164a1.json) |
| 1208–1211 | [JSON](../results/summary/20261016-p14-actor-repair-s1208to1211-Asterix-MinAtar-7ff5854.json) | [manifest](../runs/20261016-p14-actor-repair-s1208to1211-Asterix-MinAtar-7ff5854.json) |
| 1212–1215 | [JSON](../results/summary/20261016-p14-actor-repair-s1212to1215-Asterix-MinAtar-c0e8164.json) | [manifest](../runs/20261016-p14-actor-repair-s1212to1215-Asterix-MinAtar-c0e8164.json) |
| 1216–1219 | [JSON](../results/summary/20261016-p14-actor-repair-s1216to1219-Asterix-MinAtar-9b5538b.json) | [manifest](../runs/20261016-p14-actor-repair-s1216to1219-Asterix-MinAtar-9b5538b.json) |
| 1220–1223 | [JSON](../results/summary/20261016-p14-actor-repair-s1220to1223-Asterix-MinAtar-1464c4a.json) | [manifest](../runs/20261016-p14-actor-repair-s1220to1223-Asterix-MinAtar-1464c4a.json) |
| 1224–1227 | [JSON](../results/summary/20261016-p14-actor-repair-s1224to1227-Asterix-MinAtar-43d9ba4.json) | [manifest](../runs/20261016-p14-actor-repair-s1224to1227-Asterix-MinAtar-43d9ba4.json) |
| 1228–1231 | [JSON](../results/summary/20261016-p14-actor-repair-s1228to1231-Asterix-MinAtar-8e13b34.json) | [manifest](../runs/20261016-p14-actor-repair-s1228to1231-Asterix-MinAtar-8e13b34.json) |

Do not substitute the seed 99992 software smoke. Earlier Asterix 800–807 are prior selection evidence and are not pooled into these 32 rows. New contrast arms on older IDs are not independent replications.

## 5. The exact four contrasts and AUC scale

For seed i, define `A(X) = mean(X[1:])` across the 20 post-removal checkpoints +10,+20,…,+200. `X[0]` is the removal evaluation and is **excluded**. This is a discrete average, not an unnormalised/trapezoidal integral. Its units are true-reward evaluation episode-return points, not training reward lost or steps to recovery.

Construct historical `H(k)=C(390+k)` from the archived unbroken clean history. At recovery evaluation update u=440+k, use the clean value at **u−50**, not u and not u−40. The analysis and independent verifier reconstruct this explicitly.

| quantity | archive fields / expression | required mean |
|---|---|---|
| native AUC cost | `A(pause/damaged/donor/DDD) − A(H)` | <−.50 |
| native late cost | mean of paired `DDD−H` at +160,+170,+180,+190,+200 | <−.30 |
| actor-only rescue | `A(pause/damaged/donor/CDD) − A(pause/damaged/donor/DDD)` | >+.50 |
| reciprocal transfer | `A(pause/damaged/donor/DCC) − A(pause/damaged/donor/CCC)` | <−.50 |

In these **pause-cell** codes, C selects onset parameters/moments, not the clean fork after extra progress. CCC is a matched onset-*learner* reference under D world/RNG/counters, not the full P runner and not an assumed-identical historical H trace. Fork/clean/CCC is a different arm.

The one-sided paired-t tests are shifted by these practical thresholds. The one primary p is their **maximum**, with alpha=.01 and both ≥24/32 seed-consistency requirements. All four tests must pass. The complete cohort gives p=3.319136731810534e−8, 32/32 negative native AUC and 32/32 positive rescue. Late cost is negative in 31/32; do not claim late harm in every seed. No episode/checkpoint pseudoreplication.

A native AUC effect −1.722 is an average gap of 1.722 Asterix return points at this fixed grid. The prior clean score ≈6.0 motivated the registered .50 margin (about 8%) and .30 late margin (5%). Other tasks have different reward scales, evaluation policies and horizons; their raw magnitudes are not directly comparable. One PPO update=8192 environment steps; the recovery spans 1,638,400 steps. PQN-family update counts are different and must be aligned in environment steps before a horizon comparison.

The 91.8% repaired fraction is `mean(rescue) / -mean(native_cost)`, **not** mean of seedwise fractions, a Shapley allocation, an information share or a universal success rate. Its 95% paired-bootstrap interval is 86.3–97.5%. A small negative historical residual remains; secondary ±.30 equivalence passes versus H but **fails** versus matched onset-state CCC. All controls remain in the full JSON and main secondary table.

## 6. Reanalyse existing data, without training

Use a separate working copy if you want to regenerate outputs, rather than silently overwrite a frozen release:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-repro.txt
pip install -e . --no-deps

python scripts/verify_p14.py
python scripts/build_primary_artifact_index.py  # packaging only; raw rows must match frozen analysis
# Optional: regenerate the fixed, complete-n32 analysis in this working copy.
OPENBLAS_NUM_THREADS=1 python scripts/analyze_p14.py
```

The independent verifier checks the registration/core-source bytes, raw file hashes, every paired primary row, historical lag, all shifted p-values, 95/99% t intervals, seed consistency and the compound decision. The export checks all 352 requested seed/cell curve shapes and finiteness, second-key curves where specified, and the exact CSV rows. It does **not** recreate missing runner states or independently reevaluate policies from saved weights.

## 7. Fresh rollout-level recomputation (not cell resumption)

A training rerun is a reproduction of the same seeds/design, not a new independent cohort. Use the recorded Linux/Python 3.13 numerical environment, exact four-seed batches, and sequential JAX jobs on the small CPU machine. The original eight groups took 6h34m33s on two CPU cores; runtime elsewhere can differ.

Example in an isolated clean source worktree:

```bash
git worktree add ../asterix-fresh-reproduction 74c6ceb9eb75f3727137a98b42ec7caa685dc325
cd ../asterix-fresh-reproduction
# Create the environment from requirements-repro.txt as above.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u scripts/p14_actor_repair.py \
  --tag actor-repair-s1200to1203-fresh-reproduction --seeds 1200 1201 1202 1203
```

The **fresh suffix matters**: the original tag can reuse already complete cells after matching donor/fork fingerprints. A new suffix recomputes all 11 cells rather than quietly resuming archived outcomes. Preserve the old archive, compare hashes/curves, and retain any mismatch as a reproduction discrepancy—not a reason to change the scientific design.

Continue the original groups 1204–1207, 1208–1211, 1212–1215, 1216–1219, 1220–1223, 1224–1227 and 1228–1231 in that same four-seed shape. Each newly written run manifest must be committed/recorded, with its corresponding summary, before the next run to satisfy the clean-tree provenance guard. Do not pool duplicate seeds. The loader checks donor fingerprints and overlapping cell equality before accepting duplicates/resumption. Run analysis only after all 32 have all 11 cells.

The historical queue's dates have passed. Do not alter the registered scientific protocol to make its old wall-clock queue run: document a new execution schedule for reproduction while keeping scientific seeds, groups, cells, horizon, thresholds and decision rules fixed.

## 8. Scope, limitations and versioning

The primary is one setting-specific four-part claim at alpha=.01. Only five Phase-13 claims share that phase's Holm family; no paper-wide multiplicity guarantee is asserted. Nominal secondary intervals and TOST results do not establish generic context neutrality, absent mechanisms or irreversible loss. No new-task plasticity, unique natural encoding, general actor dominance or automatic rollback policy is validated.

Public repository links identify the authors' repository. Replace them with a faithful anonymized mirror if required for a double-blind submission; this revision is a public reproducibility package, not a claim of an actual venue submission.

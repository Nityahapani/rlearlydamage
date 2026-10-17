"""Deterministic packaging of existing Phase14 evidence, not a new statistical analysis.

Uses only the standard library. Reconstructs all four paired primary quantities from the raw
curves, checks the frozen analysis, and emits a 32-row CSV plus a source/checkpoint-availability
index. Donor hashes are explicitly four-seed BATCH hashes, not per-seed checkpoint files.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "74c6ceb9eb75f3727137a98b42ec7caa685dc325"
COLUMNS = ["training_seed", "native_pause_AUC_cost", "native_late_cost",
           "onset_actor_rescue", "reciprocal_actor_transfer",
           "repaired_minus_historical_pause", "raw_summary", "run_manifest"]
PRIMARY_KEYS = ["native_AUC_harm", "native_late_harm", "onset_actor_rescue",
                "reciprocal_actor_transfer"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path = ROOT) -> tuple[str, dict]:
    analysis_path = root / "results/summary/p14_single_result-analysis.json"
    analysis = json.loads(analysis_path.read_text())
    protocol_path = root / "prereg/phase14.json"
    protocol = json.loads(protocol_path.read_text())
    assert analysis["n"] == protocol["n"] == 32
    assert analysis["seeds"] == protocol["training_seeds"]
    assert digest(protocol_path) == analysis["prereg_sha256"]
    rows, groups = [], []
    expected_cells = {"/".join(c[k] for k in ["reference", "context", "opt", "code"])
                      for c in protocol["cells"]}
    for filename in analysis["summaries"]:
        path = root / filename
        assert digest(path) == analysis["source_hashes"][filename]
        data = json.loads(path.read_text())
        assert data["status"] == "complete" and set(data["cells"]) == expected_cells
        assert data["eval_updates"] == list(range(440, 641, 10))
        assert all(data["checks"][key] for key in
                   ["DDD_equals_untouched_natural_D", "CCC_equals_untouched_natural_C"])
        n = len(data["seeds"])
        for key, cell in data["cells"].items():
            assert len(cell["eval_return_mean"]) == n
            for curve in cell["eval_return_mean"]:
                assert len(curve) == 21 and all(math.isfinite(v) for v in curve)
            if key in protocol["second_eval_cells"]:
                alt = cell["eval_alt_return_mean"]
                assert len(alt) == n
                assert all(len(curve) == 21 and all(math.isfinite(v) for v in curve)
                           for curve in alt)
        manifest_name = "runs/" + data["manifest"]
        manifest = root / manifest_name
        assert manifest.is_file()
        groups.append(dict(seeds=data["seeds"], raw_summary=filename,
                           raw_summary_sha256=digest(path), run_manifest=manifest_name,
                           run_manifest_sha256=digest(manifest),
                           donor_fingerprints_four_seed_batch=data["state_fingerprints"],
                           recorded_runtime_checks=data["checks"],
                           requested_cell_count=len(data["cells"])))
        for j, seed in enumerate(data["seeds"]):
            def curve(key):
                return data["cells"][key]["eval_return_mean"][j]
            clean = dict(zip(data["fork_eval"]["updates"],
                             [v[j] for v in data["fork_eval"]["C"]]))
            clean.update(zip(data["eval_updates"], curve("fork/clean/donor/CCC")))
            historical = [clean[u - 50] for u in data["eval_updates"]]
            native = curve("pause/damaged/donor/DDD")
            restored = curve("pause/damaged/donor/CDD")
            impaired_actor = curve("pause/damaged/donor/DCC")
            onset_reference = curve("pause/damaged/donor/CCC")
            values = [statistics.mean(native[1:]) - statistics.mean(historical[1:]),
                      statistics.mean(a-b for a, b in zip(native[-5:], historical[-5:])),
                      statistics.mean(restored[1:]) - statistics.mean(native[1:]),
                      statistics.mean(impaired_actor[1:]) - statistics.mean(onset_reference[1:])]
            position = analysis["seeds"].index(seed)
            for key, value in zip(PRIMARY_KEYS, values):
                assert math.isclose(value, analysis["primary"][key]["per_seed"][position],
                                    rel_tol=0, abs_tol=1e-12)
            rows.append([seed, *values, values[0] + values[2], filename, manifest_name])
    assert [row[0] for row in rows] == list(range(1200, 1232))
    assert len(groups) == 8 and all(len(g["seeds"]) == 4 for g in groups)
    sources = ["prereg/phase14.json", "scripts/p14_actor_repair.py",
               "src/rled/agents/ppo.py", "src/rled/surgery.py"]
    for name in sources:
        frozen = subprocess.check_output(["git", "show", "6262b2b:" + name], cwd=root)
        assert (root / name).read_bytes() == frozen, name
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    csv_text = buffer.getvalue()
    index = dict(
        schema="p14-primary-artifacts-v1", role="deterministic evidence export; no new inference",
        scientific_release_commit=BASELINE, registration_commit="6262b2b",
        n=32, seeds=analysis["seeds"],
        frozen_analysis="results/summary/p14_single_result-analysis.json",
        frozen_analysis_sha256=digest(analysis_path),
        protocol="prereg/phase14.json",
        frozen_scientific_source_sha256={name: digest(root / name) for name in sources},
        checkpoint_availability=dict(
            full_serialized_donor_or_runner_trees_archived=False,
            fingerprint_scope="each P/C/D fingerprint covers one complete FOUR-SEED batched runner tree",
            P="shared onset, update390; donor of actor parameters for the main restoration",
            C="unimpaired fork at update440; used for the historical clean trajectory, not the repair actor",
            D="impaired fork at update440 after removal; source of natural context and other learner fields",
            regeneration="scripts/p14_actor_repair.py recreates donors in memory; a fresh tag avoids cell resumption",
            limitation="fingerprints are not checkpoint objects and cannot restore states without recomputation"),
        state_integrity=dict(
            runtime_assertions="raw checks verify full-tree natural DDD=D and clean CCC=C",
            construction="actor-only CDD is built from D context plus onset actor via compose_ppo",
            downstream="environment/critic/moments/experience can diverge after the intervention",
            verifier_limit="archived means/fingerprints do not permit rechecking omitted full runner trees"),
        csv="results/summary/p14_primary_per_seed.csv",
        csv_sha256=hashlib.sha256(csv_text.encode()).hexdigest(),
        auc_definition="mean of eval_return_mean[:,1:] at +10..200, not an unnormalised integral",
        late_definition="mean paired difference at +160,+170,+180,+190,+200",
        primary_evaluation=dict(key=10000, episodes_per_checkpoint=64,
                                independent_second_key=54321, second_key_not_selected_or_pooled=True),
        groups=groups)
    return csv_text, index


def main():
    text, index = build()
    (ROOT / "results/summary/p14_primary_per_seed.csv").write_text(text)
    (ROOT / "docs/p14_primary_artifact_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print("Verified/exported32 paired rows,8 raw groups,352 complete cell curves; batch-hash and checkpoint limitations explicit")


if __name__ == "__main__":
    main()

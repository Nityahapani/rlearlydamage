"""Presentation exports cannot change or inflate the frozen primary evidence."""
from pathlib import Path
import csv
import importlib.util
import io
import json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("primary_artifacts", ROOT / "scripts/build_primary_artifact_index.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_export_is_exact_and_deterministic():
    csv_text, index = module.build(ROOT)
    assert csv_text == (ROOT / "results/summary/p14_primary_per_seed.csv").read_text()
    assert index == json.loads((ROOT / "docs/p14_primary_artifact_index.json").read_text())
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert [int(row["training_seed"]) for row in rows] == list(range(1200, 1232))
    assert all(float(row["native_pause_AUC_cost"]) < 0 for row in rows)
    assert all(float(row["onset_actor_rescue"]) > 0 for row in rows)
    assert len({row["raw_summary"] for row in rows}) == 8


def test_no_checkpoint_or_inference_inflation():
    _, index = module.build(ROOT)
    assert index["checkpoint_availability"]["full_serialized_donor_or_runner_trees_archived"] is False
    assert "FOUR-SEED" in index["checkpoint_availability"]["fingerprint_scope"]
    for group in index["groups"]:
        assert len(group["seeds"]) == 4 and group["requested_cell_count"] == 11
        assert set(group["donor_fingerprints_four_seed_batch"]) == {"P", "C", "D"}
        assert all(len(v) == 64 for v in group["donor_fingerprints_four_seed_batch"].values())
        assert (ROOT / group["raw_summary"]).is_file()
        assert (ROOT / group["run_manifest"]).is_file()
    # Interval/multiplicity decisions are still provided by the untouched registered analysis,
    # not recomputed or selected by a presentation-export routine.
    assert "primary_p_values" not in index
    assert index["n"] == 32

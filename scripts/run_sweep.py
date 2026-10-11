"""Critical-period sweep entry point (Phase 2). Currently prints the planned grid and writes a manifest.

Usage: python scripts/run_sweep.py --deficit obs_noise --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rled.deficits import window_grid
from rled.provenance import build_manifest, write_manifest

ONSETS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DURATIONS = [0.05, 0.1, 0.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deficit", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
    cfg.update(yaml.safe_load(Path(f"configs/deficits/{a.deficit}.yaml").read_text()))
    grid = window_grid(ONSETS, DURATIONS, cfg["deficit"]["type"], cfg["deficit"]["params"])
    print(f"{len(grid)} windows x {len(cfg['seeds'])} seeds = {len(grid) * len(cfg['seeds'])} runs")
    for w in grid:
        print(f"  onset={w.onset_frac:.2f} duration={w.duration_frac:.2f} removal={w.removal_frac:.2f}")
    if a.dry_run:
        return
    m = build_manifest(f"sweep-{a.deficit}", cfg, cfg["seeds"], allow_dirty=a.allow_dirty)
    print("manifest:", write_manifest(m))
    raise NotImplementedError("training loop arrives in Phase 1/2")


if __name__ == "__main__":
    main()

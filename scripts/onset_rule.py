"""Apply the prespecified meaning-matched onset rule to (pooled) control runs.

Rule (fixed in Phase 5, LOG.md): late onset = the first 10-update multiple at which
  (i)  exploration has reached its floor (PQN: epsilon schedule end; PPO: no schedule, condition vacuous), and
  (ii) the control's mean eval is >= 40% of its final value (40% = PPO SpaceInvaders control at u100: 45.4 / 112.6).
"final" = mean over checkpoints in the last 50 updates (the convention of analyze_pqn_recovery.py).
Metric (PQN): the eval with the lower control checkpoint-to-checkpoint noise after the epsilon floor
(mean |diff| / mean level); PPO: the stochastic-policy eval (eval_return_mean).
Also reports per-seed crossing points and per-100-update gains (plateau check).

Usage: python scripts/onset_rule.py results/summary/<control>.json [...] --tag <tag>
Output: results/summary/<tag>-onset.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load(paths):
    U, evals, seeds, kind, floor, man = None, {}, [], None, 0, []
    for p in paths:
        s = json.loads(Path(p).read_text())
        man.append(s["manifest"])
        if "arms" in s:  # PQN / PQN-R
            kind, d = "pqn", s["arms"]["control"]
            sched = s["protocol"]["exposure"]["pqn"]["eps_schedule"]
            floor = int(re.search(r"\((\d+) updates\)", sched).group(1))
            keys = ("eval_return_mean", "eval_eps05_return_mean")
        else:  # PPO recovery_test.py
            kind, d = "ppo", s["configs"]["control"]
            keys = ("eval_return_mean",)
        assert U is None or U == s["eval_updates"], "eval grids differ"
        U = s["eval_updates"]
        seeds += s["seeds"]
        for k in keys:
            if k in d:
                evals.setdefault(k, []).append(np.array(d[k], float))
    assert len(set(seeds)) == len(seeds), "duplicate seeds"
    return np.array(U), {k: np.concatenate(v) for k, v in evals.items()}, seeds, kind, floor, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frac", type=float, default=0.40)
    a = ap.parse_args()
    U, ev, seeds, kind, floor, man = load(a.summaries)
    E = int(U[1] - U[0])
    rel = {}
    for k, M in ev.items():
        sel = U >= max(floor, U[0])
        rel[k] = float(np.abs(np.diff(M[:, sel], axis=1)).mean() / max(M[:, sel].mean(), 1e-8))
    metric = min(rel, key=rel.get) if kind == "pqn" else "eval_return_mean"
    M = ev[metric]
    m = M.mean(0)
    last = U >= U[-1] - 50
    fin_seed = M[:, last].mean(1)
    fin = fin_seed.mean()
    thr = a.frac * fin
    cross = next((int(u) for u, v in zip(U, m) if v >= thr), None)
    cross_seed = [next((int(u) for u, v in zip(U, r) if v >= a.frac * f), None) for r, f in zip(M, fin_seed)]
    onset = next((int(u) for u, v in zip(U, m) if u >= floor and u % 10 == 0 and v >= thr), None)
    L = [f"# Onset rule — {a.tag}", "", f"Manifests: {', '.join(man)}. Seeds (n = {len(seeds)}): {seeds}.", "",
         f"Control eval reliability (mean |checkpoint diff| / level, after u{floor}): "
         + ", ".join(f"{k} {r:.3f}" for k, r in rel.items()) + f". **Metric: {metric}.**", "",
         f"Final (mean of checkpoints in the last 50 updates): {fin:.1f} (per-seed sd {fin_seed.std(ddof=1):.1f}). "
         f"Threshold {a.frac:.0%} of final = {thr:.1f}.", "",
         f"- Exploration floor: u{floor}" + (" (PPO: none)" if kind == "ppo" else ""),
         f"- Mean control first reaches the threshold at u{cross}; per seed: {cross_seed}",
         f"- **Onset by rule: u{onset}**" + (f" (control there = {m[list(U).index(onset)]:.1f} = "
                                             f"{m[list(U).index(onset)] / fin:.0%} of final)" if onset is not None else " (never reached)"),
         "", "Mean control eval and gain per 100 updates:", "", "| block | mean at end | gain |", "|---|---|---|"]
    for b in range(0, int(U[-1]), 100):
        i0 = list(U).index(b) if b in U else None
        i1 = list(U).index(min(b + 100, int(U[-1]))) if min(b + 100, int(U[-1])) in U else None
        if i0 is not None and i1 is not None:
            L.append(f"| {b}-{U[i1]} | {m[i1]:.2f} | {m[i1] - m[i0]:+.2f} |")
    out = Path("results/summary") / f"{a.tag}-onset.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote", out)


if __name__ == "__main__":
    main()

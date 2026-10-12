"""Time-resolved surgery contrasts by replay-overwrite phase (pooled seed groups, paired 95% t-CI).

Phases are defined from the logged buffer tags (buf_frac_window of the D-buffer arms, mean over seeds):
  carried   : post-removal checkpoints where restricted-era data still fills >= 90% of its share at removal
  draining  : share between 0 and 90% of its value at removal
  gone      : share == 0
Usage: python scripts/surgery_phase_contrasts.py <surgery summaries...> --tag <tag>  -> results/summary/<tag>-phases.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def fmt(x):
    m = x.mean(); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}]" + ("*" if (m - h > 0 or m + h < 0) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--metric", default="eval_eps05_return_mean")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    S = [json.loads(Path(f).read_text()) for f in a.summaries]
    S = [t for t in S if "buf_frac_window" in t["arms"]["DD"]["train"]]
    seeds = sum((t["seeds"] for t in S), [])
    U = np.array(S[0]["eval_updates"]); w1 = U[0]
    ev = {k: np.concatenate([np.array(t["arms"][k][a.metric], float) for t in S]) for k in ("CC", "DC", "CD", "DD")}
    B = np.concatenate([np.array(t["arms"]["DD"]["train"]["buf_frac_window"], float) for t in S]).mean(0)
    P = np.concatenate([np.array(t["arms"]["DD"]["train"]["sample_frac_window"], float) for t in S]).mean(0)
    b0 = B[0]
    share = lambda h: B[h - 1] if h >= 1 else b0  # train arrays index post-removal update h at h-1
    ph = {"carried": [], "draining": [], "gone": []}
    for i, u in enumerate(U):
        h = int(u - w1)
        if h == 0:
            continue
        sh = share(h)
        ph["carried" if sh >= 0.9 * b0 else ("gone" if sh <= 0 else "draining")].append(i)
    con = {"data transfer CD - CC": ev["CD"] - ev["CC"], "parameter transfer DC - CC": ev["DC"] - ev["CC"],
           "buffer main effect": (ev["CD"] + ev["DD"] - ev["CC"] - ev["DC"]) / 2, "total DD - CC": ev["DD"] - ev["CC"]}
    L = [f"# Overwrite-phase contrasts — {a.tag}", "", f"Seeds (n = {len(seeds)}): {seeds}. Metric {a.metric}. "
         f"Restricted-era share of D buffer at removal: {b0:.3f}.", "",
         "| phase | post-removal updates (eval checkpoints) | mean buffer share | mean sampled share | " + " | ".join(con) + " |",
         "|---" * (4 + len(con)) + "|"]
    for name, idx in ph.items():
        if not idx:
            continue
        hs = [int(U[i] - w1) for i in idx]
        bs = np.mean([share(h) for h in hs]); ps = np.mean([P[h - 1] for h in hs])
        L.append(f"| {name} | +{min(hs)}..+{max(hs)} ({len(hs)}) | {bs:.3f} | {ps:.3f} | "
                 + " | ".join(fmt(c[:, idx].mean(1)) for c in con.values()) + " |")
    out = Path("results/summary") / f"{a.tag}-phases.md"
    out.write_text("\n".join(L) + "\n"); print("\n".join(L)); print("wrote", out)


if __name__ == "__main__":
    main()

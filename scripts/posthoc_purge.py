"""POST-HOC (not pre-registered) checks for the Phase 6 replay purge.
  B - E : damaged vs clean network on the identical replacement buffer (network damage, data held fixed)
  purge shock: single-checkpoint B - A, C - A, E - D around each purge (+10..+30 for u350 purges, +60..+80 for C's u400 purge)
Usage: python scripts/posthoc_purge.py results/summary/*p6-purge-s?-pqnr-1000k-*.json > results/summary/p6_purge-posthoc.md
"""
import json, sys
import numpy as np
from scipy import stats

S = [json.load(open(f)) for f in sorted(sys.argv[1:])]
U = np.array(S[0]["eval_updates"]); h = U - U[0]
ev = {k: np.concatenate([np.array(t["arms"][k]["eval_eps05_return_mean"]) for t in S]) for k in S[0]["arms"]}
def f(x):
    m = x.mean(); hw = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:+.1f} [{m - hw:+.1f}, {m + hw:+.1f}]" + ("*" if m - hw > 0 or m + hw < 0 else "")
W = {"AUC": (10, 650), "early": (10, 150), "mid": (160, 240), "late": (250, 650)}
print(f"# Phase 6 replay purge — POST-HOC checks (not pre-registered), n = {sum(len(t['seeds']) for t in S)}\n")
print("| contrast | " + " | ".join(W) + " |\n|---|---|---|---|---|")
print("| B - E (network, identical replacement buffer) | " + " | ".join(f((ev["B_purge"] - ev["E_clean_purge"])[:, (h >= lo) & (h <= hi)].mean(1)) for lo, hi in W.values()) + " |")
print("\nPurge shock (single checkpoints):\n\n| checkpoint | B - A | C - A | E - D |\n|---|---|---|---|")
for c in (10, 20, 30, 60, 70, 80):
    i = list(h).index(c)
    print(f"| +{c} | {f(ev['B_purge'][:, i] - ev['A_retain'][:, i])} | {f(ev['C_delay50'][:, i] - ev['A_retain'][:, i])} | {f(ev['E_clean_purge'][:, i] - ev['D_clean'][:, i])} |")

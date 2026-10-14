"""POST-HOC (labelled as such in LOG Phase 9 pre-registration): across all PPO surgery settings, relate the competence
lost by removal and the clean agent's progress during the window to total and delay-adjusted damage. Descriptive only.

Reads the fork readouts and the total-damage row from each surgery analysis file in results/summary/.
SpaceInvaders 2x128 (n = 12) has no fork readout (no eval during the window for seeds 4-11), so C(w0) and C(w1) for
that row are taken from the Phase 2 recovery test (seeds 0-3, same configuration; p2_recovery_all-analysis.md) and
flagged.
Usage: python scripts/posthoc_p9_gap_vs_damage.py --settings name=file.md ... --out p9_gap_vs_damage
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

NUM = r"([+-]?\d+\.\d+)"
CI = NUM + r" \[" + NUM + r", " + NUM + r"\]"


def grab(txt, pat):
    m = re.search(pat, txt)
    return tuple(float(x) for x in m.groups()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", required=True, help="label=analysis.md")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for item in a.settings:
        lab, f = item.split("=", 1)
        t = Path("results/summary", f).read_text()
        fork = grab(t, r"C at onset u\d+: " + NUM + r"; D at removal u\d+: " + NUM + r"; C at removal: " + NUM)
        flag = ""
        if fork is None:  # SpaceInvaders 2x128 n=12: Phase 2 fork values (seeds 0-3)
            fork, flag = (45.4, 18.5, 58.7), " (C, D from Phase 2 seeds 0–3)"
        gap = grab(t, r"raw gap at removal DDD - CCC \(\+0\): " + CI) or grab(t, r"\| total DDD - CCC \| " + CI + r"\S* \| " + CI)[3:]
        tot = grab(t, r"\| total DDD - CCC \| " + CI)
        dadj = grab(t, r"delay-adjusted AUC DDD\(w1\+k\) - C\(w1\+k-\d+\), k = 10\.\.\d+: " + CI)
        c0, d1, c1 = fork
        rows.append((lab + flag, c0, c1, d1, gap, tot, dadj))
    L = ["# POST-HOC: competence lost at removal vs. damage across PPO surgery settings (descriptive)", "",
         "Gap = DDD − CCC at removal; relative gap = gap / C(w1); clean progress = C(w1) − C(w0) during the window; "
         "total = DDD − CCC recovery AUC; delay-adjusted = DDD(w1+k) − C(w1+k−L) AUC. Negative = harm.", "",
         "| setting | C(w0) | C(w1) | D(w1) | clean progress (rel.) | gap at removal | relative gap | total damage | total / C(w1) | delay-adjusted AUC |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, c0, c1, d1, gap, tot, dadj in rows:
        L.append(f"| {lab} | {c0:.2f} | {c1:.2f} | {d1:.2f} | {c1 - c0:+.2f} ({100 * (c1 - c0) / c0:+.0f}%) | "
                 f"{gap[0]:+.1f} [{gap[1]:+.1f}, {gap[2]:+.1f}] | {100 * gap[0] / c1:+.0f}% | "
                 f"{tot[0]:+.1f} [{tot[1]:+.1f}, {tot[2]:+.1f}] | {100 * tot[0] / c1:+.0f}% | "
                 + (f"{dadj[0]:+.1f} [{dadj[1]:+.1f}, {dadj[2]:+.1f}]" if dadj else "–") + " |")
    out = Path("results/summary") / f"{a.out}-posthoc.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

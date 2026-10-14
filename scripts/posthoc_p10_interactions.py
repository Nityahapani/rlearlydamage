"""POST-HOC (Phase 10; secondary (c) for K runs): factorial interaction contrasts and non-additivity for PPO 2^3 surgery.

For each seed, AUC of every cell (mean eval over checkpoints +E..+R). Reports, with paired t-CIs:
  * single-swap effects from the reference cell (XCC - CCC etc.) and from the damaged cell (DDD - CDD etc.);
  * two-way interactions, e.g. actor x critic = mean over opt of [(D D .) - (C D .)] - [(D C .) - (C C .)], halved
    (i.e. the change in the actor effect when the critic is damaged, averaged over optimizer source);
  * three-way interaction;
  * non-additivity = (DDD - CCC) - sum of the three single swaps from CCC.
Also: relative effects = effect / mean AUC of the reference cell CCC (for cross-algorithm comparison).
Usage: python scripts/posthoc_p10_interactions.py --summaries A.json ... --out NAME [--ref-label C]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

CODES = [a + b + c for a in "CD" for b in "CD" for c in "CD"]


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def fmt(x):
    m, lo, hi = tci(x)
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("*" if lo > 0 or hi < 0 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-label", default="C", help="label of the reference source in the report (C or P)")
    a = ap.parse_args()
    cells = {c: {} for c in CODES}
    for f in a.summaries:
        s = json.loads(Path(f).read_text())
        for arm, v in s["arms"].items():
            if arm in cells:
                ev = np.array(v["eval_return_mean"])
                for sd, row in zip(s["seeds"], ev):
                    cells[arm][sd] = row[1:].mean()
    seeds = sorted(set.intersection(*[set(cells[c]) for c in CODES]))
    Y = {c: np.array([cells[c][s] for s in seeds]) for c in CODES}
    R = a.ref_label
    lab = lambda c: c.replace("C", R)
    y = lambda x, z, o: Y[x + z + o]
    out = {}
    out["total DDD - " + lab("CCC")] = Y["DDD"] - Y["CCC"]
    out["actor swap from reference, " + lab("DCC") + " - " + lab("CCC")] = Y["DCC"] - Y["CCC"]
    out["critic swap from reference, " + lab("CDC") + " - " + lab("CCC")] = Y["CDC"] - Y["CCC"]
    out["Adam swap from reference, " + lab("CCD") + " - " + lab("CCC")] = Y["CCD"] - Y["CCC"]
    out["actor swap into damaged, DDD - " + lab("CDD")] = Y["DDD"] - Y["CDD"]
    out["critic swap into damaged, DDD - " + lab("DCD")] = Y["DDD"] - Y["DCD"]
    out["Adam swap into damaged, DDD - " + lab("DDC")] = Y["DDD"] - Y["DDC"]
    ac = np.mean([(y("D", "D", o) - y("C", "D", o)) - (y("D", "C", o) - y("C", "C", o)) for o in "CD"], 0) / 2
    ao = np.mean([(y("D", z, "D") - y("C", z, "D")) - (y("D", z, "C") - y("C", z, "C")) for z in "CD"], 0) / 2
    co = np.mean([(y(x, "D", "D") - y(x, "C", "D")) - (y(x, "D", "C") - y(x, "C", "C")) for x in "CD"], 0) / 2
    i3 = (((Y["DDD"] - Y["CDD"]) - (Y["DCD"] - Y["CCD"])) - ((Y["DDC"] - Y["CDC"]) - (Y["DCC"] - Y["CCC"]))) / 4
    out["interaction actor x critic"] = ac
    out["interaction actor x Adam"] = ao
    out["interaction critic x Adam"] = co
    out["three-way interaction"] = i3
    out["non-additivity: total - sum of swaps from reference"] = (Y["DDD"] - Y["CCC"]) - (
        (Y["DCC"] - Y["CCC"]) + (Y["CDC"] - Y["CCC"]) + (Y["CCD"] - Y["CCC"]))
    ref = Y["CCC"].mean()
    L = [f"# POST-HOC: factorial interactions and relative effects — {a.out}", "",
         f"Summaries: {', '.join(Path(f).name for f in a.summaries)}; seeds {seeds} (n = {len(seeds)}). "
         f"AUC over +E..+R; paired 95% t-CIs. Reference cell {lab('CCC')} mean AUC {ref:.1f}; "
         "'relative' = mean effect / reference AUC.", "",
         "| contrast | AUC effect | relative |", "|---|---|---|"]
    for k, v in out.items():
        L.append(f"| {k} | {fmt(v)} | {100 * v.mean() / ref:+.1f}% |")
    L += ["", "Cell means: " + ", ".join(f"{lab(c)} {Y[c].mean():.1f}" for c in CODES)]
    p = Path("results/summary") / f"{a.out}-posthoc.md"
    p.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

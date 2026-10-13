"""Phase 8 J4: PQN-R parameters vs optimizer state at the damaged buffer (pre-registered, LOG Phase 8).

Cells (params, opt, buffer): DDD, CDD, DCD from the Phase 8 runs; CCD = the Phase 5/6 'CD' arm (clean network =
clean params + clean optimizer, damaged buffer), same seeds. Check: Phase 8 DDD must equal the earlier 'DD' arm.
Metric: eval_eps05_return_mean; AUC = mean over +10..+650; windows early +10..+150, late +250..+650.
Primary (Holm across the two): parameter rescue CDD - DDD; optimizer rescue DCD - DDD.
Usage: python scripts/analyze_p8_pqnr_split.py --new A.json B.json --old OLD1.json OLD2.json [OLD3.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

MET = "eval_eps05_return_mean"


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def fmt(x):
    m, lo, hi = tci(x)
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]" + ("*" if lo > 0 or hi < 0 else "")


def load(files):
    out, U = {}, None
    for f in files:
        d = json.loads(Path(f).read_text())
        U = d["eval_updates"] if U is None else U
        assert d["eval_updates"] == U
        for arm, v in d["arms"].items():
            M = np.asarray(v[MET], float)
            for j, s in enumerate(d["seeds"]):
                out.setdefault(arm, {})[s] = M[j]
    return out, U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--old", nargs="+", required=True)
    ap.add_argument("--tag", default="p8_pqnr_split")
    a = ap.parse_args()
    new, U = load(a.new)
    old, U2 = load(a.old)
    assert U == U2
    seeds = sorted(set(new["DDD"]) & set(old["DD"]) & set(old["CD"]))
    ident = {s: bool(np.allclose(new["DDD"][s], old["DD"][s], atol=1e-3)) for s in seeds}
    cell = {"DDD": new["DDD"], "CDD": new["CDD"], "DCD": new["DCD"], "CCD": old["CD"], "CCC": old["CC"]}
    E = U[1] - U[0]
    w = {"AUC": (E, U[-1] - U[0]), "early +10..+150": (10, 150), "late +250..+650": (250, 650), "+10": (10, 10)}

    def r(arm, lo, hi):
        return np.array([cell[arm][s][lo // E: hi // E + 1].mean() for s in seeds])

    L = [f"# Phase 8 J4 — PQN-R parameters vs optimizer state (100k buffer, damaged buffer held fixed)", "",
         f"New: {', '.join(Path(f).name for f in a.new)}", f"Old (CD = CCD, DD, CC): {', '.join(Path(f).name for f in a.old)}",
         f"Seeds (n = {len(seeds)}): {seeds}. Metric {MET}. Removal u{U[0]}, recovery {U[-1] - U[0]}.",
         f"Bit-identity check Phase 8 DDD == earlier DD: {ident} -> {'PASS' if all(ident.values()) else 'FAIL'}", "",
         "## Cell means", "", "| cell (params, opt, buffer) | " + " | ".join(w) + " |", "|---|" + "---|" * len(w)]
    for arm in ("CCC", "CCD", "CDD", "DCD", "DDD"):
        L.append(f"| {arm} | " + " | ".join(f"{r(arm, *v).mean():.1f}" for v in w.values()) + " |")
    L += ["", "## Contrasts (paired; positive = rescue)", "", "| contrast | " + " | ".join(w) + " | AUC p | Holm p |", "|---|" + "---|" * (len(w) + 2)]
    cons = [("parameter rescue CDD - DDD (primary)", lambda lo, hi: r("CDD", lo, hi) - r("DDD", lo, hi)),
            ("optimizer rescue DCD - DDD (primary)", lambda lo, hi: r("DCD", lo, hi) - r("DDD", lo, hi)),
            ("network rescue CCD - DDD (= earlier CD - DD)", lambda lo, hi: r("CCD", lo, hi) - r("DDD", lo, hi)),
            ("params main effect (2x2 at D buffer)", lambda lo, hi: (r("CDD", lo, hi) + r("CCD", lo, hi) - r("DDD", lo, hi) - r("DCD", lo, hi)) / 2),
            ("opt main effect (2x2 at D buffer)", lambda lo, hi: (r("DCD", lo, hi) + r("CCD", lo, hi) - r("DDD", lo, hi) - r("CDD", lo, hi)) / 2),
            ("interaction (CCD - CDD) - (DCD - DDD)", lambda lo, hi: (r("CCD", lo, hi) - r("CDD", lo, hi)) - (r("DCD", lo, hi) - r("DDD", lo, hi)))]
    ps = [stats.ttest_1samp(cons[i][1](*w["AUC"]), 0).pvalue for i in range(2)]
    order = np.argsort(ps)
    hp = [None, None]
    hp[order[0]] = min(1, 2 * ps[order[0]])
    hp[order[1]] = max(hp[order[0]], min(1, ps[order[1]]))
    for i, (lab, fn) in enumerate(cons):
        p = stats.ttest_1samp(fn(*w["AUC"]), 0).pvalue
        L.append(f"| {lab} | " + " | ".join(fmt(fn(*v)) for v in w.values()) + f" | {p:.4f} | " + (f"{hp[i]:.4f}" if i < 2 else "") + " |")
    nr = cons[2][1](*w["AUC"]).mean()
    L += ["", f"Share of the network rescue (CCD - DDD, AUC {nr:+.2f}) recovered by clean parameters alone: "
              f"{cons[0][1](*w['AUC']).mean() / nr * 100:.0f}%; by clean optimizer state alone: {cons[1][1](*w['AUC']).mean() / nr * 100:.0f}%."]
    out = Path("results/summary") / f"{a.tag}-analysis.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

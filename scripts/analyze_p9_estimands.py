"""Phase 9 E1: confirmatory test of the two pre-declared recovery estimands (LOG 2026-10-14, Phase 9 pre-registration).

Input: a recovery_test.py summary with configs `control` and `move_late` (fresh seeds).
Estimands (per seed, AUC = mean over checkpoints k = 10..250 after removal w1; L = window length):
  E_damage (primary, question "damage beyond the training time consumed"):
        D(w1 + k) - C(w1 + k - L)                       [pause counterfactual]
        Decision (TOST, SESOI +-3): equivalent if both one-sided p < 0.05; damage if mean <= -3 and CI upper < 0;
        otherwise inconclusive.
  E_plast (primary for the plasticity question "does the restricted agent learn more slowly from equal competence?"):
        D(w1 + k) - C(t* + k), t* = first checkpoint at which C's 3-checkpoint moving average >= D(w1)
        Decision (two-sided paired t): savings if CI lower > 0; slower relearning if CI upper < 0; else inconclusive.
  Descriptive: E_total = D(w1 + k) - C(w1 + k)                 [same-update: total cost incl. lost time]
Usage: python scripts/analyze_p9_estimands.py results/summary/<stem1>.json [<stem2>.json ...] --out p9_estimands
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def fmt(x):
    m, lo, hi = tci(x)
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]" + ("*" if lo > 0 or hi < 0 else "")


def tost(x, d):
    x = np.asarray(x, float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    return float(1 - stats.t.cdf((x.mean() + d) / se, n - 1)), float(stats.t.cdf((x.mean() - d) / se, n - 1))


def estimands(U, C, D, w0, w1, K=250, E=10):
    U = list(U)
    L = w1 - w0
    i1 = U.index(w1)
    ks = list(range(E, K + 1, E))
    Cs = np.array([np.convolve(np.pad(c, 1, mode="edge"), np.ones(3) / 3, mode="valid") for c in C])
    out = {"total": [], "damage": [], "plast": [], "tstar": []}
    for s in range(C.shape[0]):
        hit = np.where(Cs[s, : i1 + 1] >= D[s, i1])[0]
        ts = U[hit[0]] if len(hit) else U[i1]
        dk = np.array([D[s, U.index(w1 + k)] for k in ks])
        out["total"].append((dk - np.array([C[s, U.index(w1 + k)] for k in ks])).mean())
        out["damage"].append((dk - np.array([C[s, U.index(w1 + k - L)] for k in ks])).mean())
        out["plast"].append((dk - np.array([C[s, U.index(ts + k)] for k in ks])).mean())
        out["tstar"].append(ts)
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="move_late")
    ap.add_argument("--sesoi", type=float, default=3.0)
    a = ap.parse_args()
    ss = [json.loads(Path(f).read_text()) for f in a.summaries]
    U = ss[0]["eval_updates"]
    assert all(x["eval_updates"] == U for x in ss)
    C = np.concatenate([np.array(x["configs"]["control"]["eval_return_mean"]) for x in ss])
    D = np.concatenate([np.array(x["configs"][a.arm]["eval_return_mean"]) for x in ss])
    w0, w1 = ss[0]["configs"][a.arm]["window_updates"]
    s = {"seeds": sum((x["seeds"] for x in ss), [])}
    assert len(set(s["seeds"])) == len(s["seeds"])
    r = estimands(U, C, D, w0, w1)
    pl, ph = tost(r["damage"], a.sesoi)
    m, lo, hi = tci(r["damage"])
    if pl < 0.05 and ph < 0.05:
        d1 = f"**EQUIVALENT** within ±{a.sesoi:g} (TOST p = {max(pl, ph):.4f})"
    elif m <= -a.sesoi and hi < 0:
        d1 = "**DAMAGE BEYOND LOST TIME** (mean <= -SESOI, CI upper < 0)"
    else:
        d1 = f"**INCONCLUSIVE** (TOST p = {max(pl, ph):.4f})"
    m2, lo2, hi2 = tci(r["plast"])
    p2 = stats.ttest_1samp(r["plast"], 0).pvalue
    d2 = ("**SAVINGS** (faster relearning from equal competence)" if lo2 > 0 else
          "**SLOWER RELEARNING** (plasticity-loss direction)" if hi2 < 0 else "**INCONCLUSIVE**")
    L = [f"# Phase 9 E1 — confirmatory recovery estimands ({a.out})", "",
         f"Sources: {', '.join(Path(f).name for f in a.summaries)}; arm `{a.arm}` window [{w0}, {w1}); seeds {s['seeds']} (n = {len(s['seeds'])}, fresh). "
         "AUC over k = 10..250 after removal; paired 95% t-CIs; negative = restricted agent worse.", "",
         f"- Eval at removal: D {D[:, U.index(w1)].mean():.1f}; C at onset {C[:, U.index(w0)].mean():.1f}; C at removal {C[:, U.index(w1)].mean():.1f}",
         f"- t* (competence-matched restart point): mean {r['tstar'].mean():.0f}, range {r['tstar'].min()}–{r['tstar'].max()}", "",
         f"**E_damage (primary; pause counterfactual): {fmt(r['damage'])}**; TOST ±{a.sesoi:g}: p_lower {pl:.4f}, p_upper {ph:.4f} → {d1}",
         f"**E_plast (primary for plasticity; competence-matched): {fmt(r['plast'])}** (p = {p2:.4f}) → {d2}",
         f"E_total (descriptive; same-update): {fmt(r['total'])}", "",
         "Per-seed (damage / plast / total): " + "; ".join(f"{sd}: {x:+.1f} / {y:+.1f} / {z:+.1f}" for sd, x, y, z in
                                                          zip(s["seeds"], r["damage"], r["plast"], r["total"]))]
    out = Path("results/summary") / f"{a.out}-analysis.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

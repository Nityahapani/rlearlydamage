"""Phase 8 J5: confirmatory point-of-no-return test at d = 300 (pre-registered, LOG Phase 8).

Fresh seeds 100-107 (two summaries). Outcomes exactly as in scripts/analyze_ponr.py:
  L_300 = mean over the last 5 checkpoints (k = 210..250) of D_online(w1 + k) - C(w0 + k)   [primary]
  A_300 = mean over k = 10..250 of the same difference; raw AUC vs C(w1 + k); transplant contrasts.
Primary decision (fixed in advance): TOST against SESOI +-5 -> equivalent if both one-sided p < 0.05;
persistent impairment if mean <= -5 and upper 95% bound < 0; otherwise inconclusive.
Secondary: pooled with the pilot seeds 0-3 at d = 300 (n = 12, descriptive); split-half (odd/even checkpoint)
reliability of A_300 across seeds (Spearman-Brown).
Usage: python scripts/analyze_p8_ponr_confirm.py --confirm A.json B.json --pilot PILOT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

D = 300


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


def units(f):
    s = json.loads(Path(f).read_text())
    E = s["protocol"]["eval_every"]
    K = s["recovery"] // E
    C = np.array(s["C"]["eval_return_mean"])
    Dm = np.array(s["D_mask"]["eval_return_mean"])
    arm = s["arms"][str(D)]
    on = np.array(arm["D_online"]["eval_return_mean"])
    ref_delay = C[:, 1:K + 1]
    ref_raw = C[:, D // E + 1:D // E + K + 1]
    out = dict(seeds=s["seeds"], L=(on[:, -5:] - ref_delay[:, -5:]).mean(1), A=(on[:, 1:] - ref_delay).mean(1),
               raw=(on[:, 1:] - ref_raw).mean(1), at_rem=Dm[:, D // E], C_rem=C[:, D // E], C_on=C[:, 0],
               diff=on[:, 1:] - ref_delay)
    if "D_transplant" in arm:
        tr = np.array(arm["D_transplant"]["eval_return_mean"])
        out.update(tr_vs_on=(tr[:, 1:] - on[:, 1:]).mean(1), tr_vs_C=(tr[:, 1:] - ref_raw).mean(1),
                   tr_L=(tr[:, -5:] - ref_delay[:, -5:]).mean(1))
    return out


def cat(us, k):
    return np.concatenate([u[k] for u in us])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", nargs="+", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--sesoi", type=float, default=5.0)
    a = ap.parse_args()
    us = [units(f) for f in a.confirm]
    pu = units(a.pilot)
    seeds = sum((u["seeds"] for u in us), [])
    L, A = cat(us, "L"), cat(us, "A")
    p_lo, p_hi = tost(L, a.sesoi)
    m, lo, hi = tci(L)
    if p_lo < 0.05 and p_hi < 0.05:
        dec = f"**EQUIVALENT** within ±{a.sesoi:g} (TOST p = {max(p_lo, p_hi):.4f})"
    elif m <= -a.sesoi and hi < 0:
        dec = "**PERSISTENT IMPAIRMENT** (mean <= -SESOI and CI upper < 0)"
    else:
        dec = f"**INCONCLUSIVE** (TOST p = {max(p_lo, p_hi):.4f}; CI [{lo:+.1f}, {hi:+.1f}])"
    diff = np.concatenate([u["diff"] for u in us])
    odd, even = diff[:, 0::2].mean(1), diff[:, 1::2].mean(1)
    r = stats.spearmanr(odd, even).statistic
    sb = 2 * r / (1 + r) if r > -1 else np.nan
    Lp = np.concatenate([L, pu["L"]])
    Ap = np.concatenate([A, pu["A"]])
    pl, ph = tost(Lp, a.sesoi)
    lines = ["# Phase 8 J5 — confirmatory point-of-no-return test at d = 300 (PPO, movement restriction)", "",
             f"Confirmatory summaries: {', '.join(Path(f).name for f in a.confirm)}; pilot: {Path(a.pilot).name}.",
             f"Fresh seeds (n = {len(seeds)}): {seeds}. Onset 100, removal 400, budget 250 updates. Negative = damage beyond lost time.", "",
             f"- Unmasked eval: C at onset {cat(us, 'C_on').mean():.1f}; at removal D {cat(us, 'at_rem').mean():.1f} vs C {cat(us, 'C_rem').mean():.1f}",
             f"- **L_300 (primary): {fmt(L)}**; TOST ±{a.sesoi:g}: p_lower {p_lo:.4f}, p_upper {p_hi:.4f}",
             f"- Pre-registered decision: {dec}",
             f"- A_300 (delay-adjusted AUC): {fmt(A)}",
             f"- raw AUC vs C(w1 + k): {fmt(cat(us, 'raw'))}"]
    if "tr_vs_on" in us[0]:
        lines += [f"- transplant − D_online (AUC): {fmt(cat(us, 'tr_vs_on'))}; transplant vs C(w1 + k): {fmt(cat(us, 'tr_vs_C'))}; "
                  f"transplant late delay-adj.: {fmt(cat(us, 'tr_L'))}"]
    lines += ["", "Per-seed L_300: " + ", ".join(f"{s}: {v:+.1f}" for s, v in zip(seeds, L)),
              "Per-seed A_300: " + ", ".join(f"{s}: {v:+.1f}" for s, v in zip(seeds, A)), "",
              f"## Secondary (descriptive)", "",
              f"- Pooled with pilot seeds 0–3 (n = {len(Lp)}): L_300 {fmt(Lp)} (TOST ±{a.sesoi:g} p = {max(pl, ph):.4f}); A_300 {fmt(Ap)}",
              f"- Split-half reliability of A_300 across the {len(seeds)} confirmatory seeds: odd/even-checkpoint Spearman {r:+.2f}, "
              f"Spearman-Brown {sb:.2f}"]
    out = Path("results/summary/p8_ponr_confirm-analysis.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

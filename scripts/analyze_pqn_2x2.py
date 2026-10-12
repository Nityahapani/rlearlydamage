"""Analyze the Phase 6c identity x residence 2x2. Pre-registered (LOG.md) before any real Phase 6c run.

Cells (paired by seed). B_targeted and C_sham come from the Phase 6b summaries; Bp_targeted_short and Cp_sham_long
from the Phase 6c summaries (scripts/pqn_residence_2x2.py; same removal state and pool, checked):
                      replacement long (newest rows)   replacement short (older rows)
  R removed           B_targeted                       Bp_targeted_short
  P removed (sham)    Cp_sham_long                     C_sham
Primary metric eval_eps05_return_mean; primary endpoint AUC (+10..+650); secondary windows early +10..150,
middle +160..240, late +250..650. 95% paired t-CIs.
Primary contrasts (Holm across the three AUC tests; positive = first better):
  identity at matched long residence   B - Cp
  identity at matched short residence  Bp - C
  residence main effect                ((B - Bp) + (Cp - C)) / 2
Secondary: identity main effect ((B - Cp) + (Bp - C)) / 2; interaction (B - Cp) - (Bp - C); simple residence effects
B - Bp and Cp - C; exact sign-flip p; seed-level differences; Phase 6b B - C decomposed as
(B - Cp) + (Cp - C) = identity at long residence + residence effect given P removed. Exposure: restricted-era and
replacement shares per checkpoint; cumulative expected sampled restricted-era and replacement transitions
(share x 64 x 128 per update); check that the residence-matched pairs have matching replacement curves.
A_retain and D_cleannet (Phase 6b) are drawn for reference only.
Usage: python scripts/analyze_pqn_2x2.py --p6b <p6b summaries> --p6c <p6c summaries> --tag p6c_2x2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pqn_targeted import fmt, holm, signflip_p, tci

CELLS = ("B_targeted", "Bp_targeted_short", "Cp_sham_long", "C_sham")
REF = ("A_retain", "D_cleannet")
COL = {"B_targeted": "tab:blue", "Bp_targeted_short": "tab:cyan", "Cp_sham_long": "tab:brown", "C_sham": "tab:orange",
       "A_retain": "tab:red", "D_cleannet": "k"}
SPU = 64 * 128  # loss-weighted replay samples per update


def load(files, arms, metric):
    S = [json.loads(Path(f).read_text()) for f in files]
    rows = {}
    for t in S:
        for i, s in enumerate(t["seeds"]):
            assert s not in rows, f"duplicate seed {s}"
            rows[s] = {k: {m: np.array(t["arms"][k][m], float)[i] for m in (metric, "restricted_share", "replacement_share")}
                       for k in arms if k in t["arms"]}
    return S, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p6b", nargs="+", required=True)
    ap.add_argument("--p6c", nargs="+", required=True)
    ap.add_argument("--metric", default="eval_eps05_return_mean")
    ap.add_argument("--tag", default="p6c_2x2")
    a = ap.parse_args()
    Sb, rb = load(a.p6b, ("B_targeted", "C_sham") + REF, a.metric)
    Sc, rc = load(a.p6c, ("Bp_targeted_short", "Cp_sham_long", "B_targeted"), a.metric)
    U = np.array(Sb[0]["eval_updates"])
    for t in Sb + Sc:
        assert t["eval_updates"] == list(U) and t["blocks"] == Sb[0]["blocks"], "protocol mismatch"
    checks = {}
    for t in Sc:
        for k, v in t["consistency_check"].items():
            checks.setdefault(k, {}).update(v)
    seeds = sorted(set(rb) & set(rc)); n = len(seeds)
    for k, v in checks.items():
        assert all(v[str(s)] for s in seeds if str(s) in v), f"consistency check failed: {k} {v}"
    assert all(str(s) in checks.get("forks_match_p6b", {}) and str(s) in checks.get("pool_stats_match_p6b", {}) for s in seeds), \
        "every seed needs a fork + pool consistency check against Phase 6b"
    get = lambda k, m: np.stack([(rb[s] if k in ("B_targeted", "C_sham") + REF else rc[s])[k][m] for s in seeds])
    ev = {k: get(k, a.metric) for k in CELLS + REF}
    sr = {k: get(k, "restricted_share") for k in CELLS + REF}
    sp = {k: get(k, "replacement_share") for k in CELLS + REF}
    w1 = int(U[0]); E = int(U[1] - U[0]); R = int(U[-1] - w1); h = U - w1
    win = {"AUC": (E, R), "early": (10, 150), "middle": (160, 240), "late": (250, R)}
    cols = list(win)
    sc = {k: {w: ev[k][:, (h >= lo) & (h <= hi)].mean(1) for w, (lo, hi) in win.items()} for k in CELLS + REF}
    B, Bp, Cp, C = (sc[k] for k in CELLS)
    comb = lambda f: {w: f(B[w], Bp[w], Cp[w], C[w]) for w in cols}
    prim = {"identity at matched long residence: B - Cp": comb(lambda b, bp, cp, c: b - cp),
            "identity at matched short residence: Bp - C": comb(lambda b, bp, cp, c: bp - c),
            "residence main effect: ((B - Bp) + (Cp - C)) / 2": comb(lambda b, bp, cp, c: ((b - bp) + (cp - c)) / 2)}
    sec = {"identity main effect: ((B - Cp) + (Bp - C)) / 2": comb(lambda b, bp, cp, c: ((b - cp) + (bp - c)) / 2),
           "interaction: (B - Cp) - (Bp - C)": comb(lambda b, bp, cp, c: (b - cp) - (bp - c)),
           "residence | R removed: B - Bp": comb(lambda b, bp, cp, c: b - bp),
           "residence | P removed: Cp - C": comb(lambda b, bp, cp, c: cp - c),
           "Phase 6b reference: B - C": comb(lambda b, bp, cp, c: b - c)}
    L = [f"# Phase 6c identity x residence 2x2 — {a.tag}", "",
         f"Phase 6b manifests: {', '.join(t['manifest'] for t in Sb)}.", f"Phase 6c manifests: {', '.join(t['manifest'] for t in Sc)}.",
         f"Seeds (n = {n}): {seeds}. Metric: {a.metric}. Removal u{w1}; recovery {R}; eval every {E}.",
         f"Consistency checks vs Phase 6b (all must be True): {checks}.", "",
         "Layout (positions P2 = rows written u150-249, P1 = newest rows u250-349): B = [P | pool], Bp = [pool | P], "
         "Cp = [R | pool], C = [pool | R].", "", "## Cell scores (mean over seeds)", "",
         "| cell | removed | replacement | " + " | ".join(cols) + " |", "|---" * (len(cols) + 3) + "|"]
    meta = {"B_targeted": ("R", "long"), "Bp_targeted_short": ("R", "short"), "Cp_sham_long": ("P", "long"), "C_sham": ("P", "short"),
            "A_retain": ("none (ref)", "-"), "D_cleannet": ("R, clean net (ref)", "long")}
    for k in CELLS + REF:
        L.append(f"| {k} | {meta[k][0]} | {meta[k][1]} | " + " | ".join(f"{sc[k][c].mean():.1f}" for c in cols) + " |")
    ps = np.array([tci(prim[nm]["AUC"])[3] for nm in prim]); adj = holm(ps)
    L += ["", "## Primary contrasts (Holm across the three AUC tests)", "",
          "| contrast | " + " | ".join(cols) + " | AUC t p | Holm p | sign-flip p | seeds > 0 |", "|---" * (len(cols) + 5) + "|"]
    for nm, pr, pa in zip(prim, ps, adj):
        x = prim[nm]["AUC"]
        L.append(f"| {nm} | " + " | ".join(fmt(prim[nm][c]) for c in cols) + f" | {pr:.4f} | {pa:.4f} | {signflip_p(x):.4f} | {(x > 0).sum()}/{n} |")
    L += ["", "## Secondary (descriptive)", "", "| contrast | " + " | ".join(cols) + " | sign-flip p | seeds > 0 |", "|---" * (len(cols) + 3) + "|"]
    for nm in sec:
        x = sec[nm]["AUC"]
        L.append(f"| {nm} | " + " | ".join(fmt(sec[nm][c]) for c in cols) + f" | {signflip_p(x):.4f} | {(x > 0).sum()}/{n} |")
    bc, bcp = sec["Phase 6b reference: B - C"]["AUC"].mean(), prim["identity at matched long residence: B - Cp"]["AUC"].mean()
    L.append(f"\nDecomposition of Phase 6b B - C ({bc:+.1f}) = identity at long residence B - Cp ({bcp:+.1f}) "
             f"+ residence given P removed Cp - C ({sec['residence | P removed: Cp - C']['AUC'].mean():+.1f}).")
    allc = {**prim, **sec}
    L += ["", "## Paired seed-level differences (AUC)", "", "| seed | " + " | ".join(nm.split(":")[1].strip() for nm in allc) + " |",
          "|---" * (len(allc) + 1) + "|"]
    for i, s in enumerate(seeds):
        L.append(f"| {s} | " + " | ".join(f"{allc[nm]['AUC'][i]:+.1f}" for nm in allc) + " |")
    L.append("| between-seed SD | " + " | ".join(f"{allc[nm]['AUC'].std(ddof=1):.1f}" for nm in allc) + " |")
    marks = [m for m in (0, 20, 40, 60, 100, 140, 150, 200, 250) if m in set(h)]
    L += ["", "## Exposure: restricted-era share / replacement share at checkpoints (mean over seeds)", "",
          "| cell | " + " | ".join(f"+{m}" for m in marks) + " |", "|---" * (len(marks) + 1) + "|"]
    for k in CELLS + REF:
        L.append(f"| {k} | " + " | ".join(f"{sr[k][:, list(h).index(m)].mean():.2f} / {sp[k][:, list(h).index(m)].mean():.2f}" for m in marks) + " |")
    up = np.arange(0, R + 1)
    cum = lambda arr: np.interp(up, h, arr.mean(0))[1:].sum() * SPU / 1e6
    L += ["", "Cumulative expected sampled transitions over +1..+650 (millions): "
          + "; ".join(f"{k}: restricted-era {cum(sr[k]):.2f}, replacement {cum(sp[k]):.2f}" for k in CELLS) + "."]
    for p, q in (("B_targeted", "Cp_sham_long"), ("Bp_targeted_short", "C_sham")):
        d = np.abs(sp[p].mean(0) - sp[q].mean(0)).max()
        L.append(f"Max |replacement-share difference| {p} vs {q}: {d:.4f} (residence-matched pair).")
    out = Path("results/summary") / f"{a.tag}-analysis.md"
    out.write_text("\n".join(L) + "\n"); print("\n".join(L)); print("wrote", out)

    f, ax = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw=dict(width_ratios=[1.3, 0.9, 1]))
    for k in CELLS + REF:
        mu = ev[k].mean(0); se = ev[k].std(0, ddof=1) / np.sqrt(n); ref = k in REF
        ax[0].plot(h, mu, color=COL[k], lw=0.9 if ref else 1.4, ls=":" if ref else "-", label=k + (" (ref)" if ref else ""))
        if not ref:
            ax[0].fill_between(h, mu - se, mu + se, color=COL[k], alpha=0.10)
    ax[0].set_xlabel(f"updates after removal (u{w1})"); ax[0].set_ylabel(a.metric); ax[0].legend(fontsize=7); ax[0].set_title("Identity x residence 2x2")
    # exposure map (residence curves coincide by design, so plot cumulative exposure per cell instead)
    for k in CELLS + REF:
        x_, y_ = cum(sp[k]), cum(sr[k])
        if k in REF:
            ax[1].scatter(x_, y_, s=260, facecolors="none", edgecolors=COL[k], lw=1.5, zorder=2, marker="s")
        else:
            ax[1].scatter(x_, y_, s=90, color=COL[k], zorder=3)
        ax[1].annotate(f"{k}\nAUC {sc[k]['AUC'].mean():.1f}", (x_, y_), textcoords="offset points", xytext=(10, -4 if k != "D_cleannet" else -20), fontsize=7)
    ax[1].set_xlabel("cumulative sampled replacement transitions (M)"); ax[1].set_ylabel("cumulative sampled restriction-era transitions (M)")
    ax[1].set_xlim(-0.05, 0.95); ax[1].set_ylim(-0.08, 0.78); ax[1].set_title("Exposure map (+1..+650; D = clean net)", fontsize=9)
    names = list(allc)
    for j, nm in enumerate(names):
        x = allc[nm]["AUC"]; m, lo, hi, _ = tci(x)
        ax[2].scatter(x, np.full(n, j) + np.linspace(-0.15, 0.15, n), s=12, color="0.5", zorder=2)
        ax[2].errorbar(m, j, xerr=[[m - lo], [hi - m]], fmt="o", color="k" if j < 3 else "0.35", capsize=4, zorder=3)
    ax[2].set_yticks(range(len(names)), [nm.split(":")[1].strip() for nm in names], fontsize=7); ax[2].invert_yaxis()
    ax[2].axvline(0, color="k", lw=0.8); ax[2].axhline(2.5, color="0.8", lw=0.8)
    ax[2].set_xlabel("paired AUC difference (top 3 = primary; bar = 95% t-CI)")
    f.tight_layout(); f.savefig(Path("figures") / f"{a.tag}.png", dpi=130); print("figure:", f"figures/{a.tag}.png")


if __name__ == "__main__":
    main()

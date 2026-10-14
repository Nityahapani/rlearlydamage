"""Phase 9 E3: independent confirmatory point-of-no-return study at d = 300 and a powered test of the four
Phase 7b removal-time predictors (pre-registered, LOG 2026-10-14 Phase 9).

Inputs: ponr_sweep.py summaries run with --durations 300 (fresh seeds 300-331), plus their results/<stem>/ folders
(C_u400.npz, D_u400.npz, bank.npz saved by the sweep).
Outcomes per seed (exactly as scripts/analyze_ponr.py):
  L_300 = mean over the last 5 checkpoints (k = 210..250) of D_online(w1 + k) - C(w0 + k)
  A_300 = mean over k = 10..250 of the same difference (the Phase 7b predictor outcome)
Primary 1 (PONR): TOST on L_300 against +-5 -> equivalent / persistent impairment (mean <= -5, CI upper < 0) /
  inconclusive.
Primary 2 (predictors): P1 kl_clean, P2 cka_clean, P3 move_gap_rel, P4 value_err_clean (definitions imported from
  scripts/ponr_predictors.py, unchanged). Spearman rho with A_300, one-sided (expected rho < 0: more divergence ->
  slower recovery), Holm across the four. A predictor is supported if Holm p < 0.05.
  For every predictor, a 95% CI for rho (Fisher z with the Bonett-Wright SE sqrt((1 + rho^2 / 2) / (n - 3))) is
  reported; its lower bound is the strongest negative association compatible with the data.
Descriptive: predictor split-half reliability (bank halves), A_300 split-half reliability (odd/even checkpoints),
  leave-one-out R^2 gain of a one-predictor linear model over the mean, the same correlations with L_300, and pooling
  of L_300 with the Phase 8 J5 seeds (100-107).
Usage: python scripts/analyze_p9_ponr_predictors.py --confirm S1.json ... [--j5 A.json B.json] --out p9_ponr_predictors
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

D = 300
spec = importlib.util.spec_from_file_location("pp", Path(__file__).with_name("ponr_predictors.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
PRED = pp.PRED


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


def rho_ci(r, n):
    se = np.sqrt((1 + r ** 2 / 2) / (n - 3))
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r_) * ps[i]))
        adj[i] = run
    return adj


def load(f, with_predictors=True):
    s = json.loads(Path(f).read_text())
    assert D in s["durations"], f
    w0 = s["onset"]
    E = s["protocol"]["eval_every"]
    K = s["recovery"] // E
    C = np.array(s["C"]["eval_return_mean"])
    on = np.array(s["arms"][str(D)]["D_online"]["eval_return_mean"])
    diff = on[:, 1:] - C[:, 1:K + 1]
    out = dict(seeds=list(s["seeds"]), L=diff[:, -5:].mean(1), A=diff.mean(1), diff=diff)
    if with_predictors:
        rdir = Path("results") / Path(f).stem
        bank = np.load(rdir / "bank.npz")["bank"]
        allowed = s["protocol"]["allowed"]
        n_act = pp.load_params(rdir / f"onset_u{w0}.npz")["Dense_0"]["kernel"].shape[-1]
        masked = [i for i in range(n_act) if i not in allowed]
        pC_all, pD_all = pp.load_params(rdir / f"C_u{w0 + D}.npz"), pp.load_params(rdir / f"D_u{w0 + D}.npz")
        rows = []
        for i in range(len(s["seeds"])):
            pC, pD, b = pp.seed_slice(pC_all, i), pp.seed_slice(pD_all, i), bank[i]
            full = pp.predictors(pD, pC, b, masked)
            h1 = pp.predictors(pD, pC, b[: len(b) // 2], masked)
            h2 = pp.predictors(pD, pC, b[len(b) // 2:], masked)
            rows.append((full, h1, h2))
        for k in PRED:
            out[k] = np.array([r[0][k] for r in rows])
            out[k + "_h1"] = np.array([r[1][k] for r in rows])
            out[k + "_h2"] = np.array([r[2][k] for r in rows])
    return out


def cat(us, k):
    return np.concatenate([u[k] for u in us])


def loo_gain(P, Y):
    n = len(Y)
    e0, e1 = [], []
    for i in range(n):
        tr = np.arange(n) != i
        e0.append((Y[i] - Y[tr].mean()) ** 2)
        b = np.polyfit(P[tr], Y[tr], 1)
        e1.append((Y[i] - np.polyval(b, P[i])) ** 2)
    return 1 - np.mean(e1) / np.mean(e0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", nargs="+", required=True)
    ap.add_argument("--j5", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sesoi", type=float, default=5.0)
    a = ap.parse_args()
    us = [load(f) for f in a.confirm]
    seeds = sum((u["seeds"] for u in us), [])
    assert len(set(seeds)) == len(seeds)
    n = len(seeds)
    L, A = cat(us, "L"), cat(us, "A")
    pl, ph = tost(L, a.sesoi)
    m, lo, hi = tci(L)
    if pl < 0.05 and ph < 0.05:
        dec = f"**EQUIVALENT** within ±{a.sesoi:g} (TOST p = {max(pl, ph):.4f})"
    elif m <= -a.sesoi and hi < 0:
        dec = "**PERSISTENT IMPAIRMENT** (mean <= -SESOI, CI upper < 0)"
    else:
        dec = f"**INCONCLUSIVE** (TOST p = {max(pl, ph):.4f})"
    diff = cat(us, "diff")
    r_oe = stats.spearmanr(diff[:, 0::2].mean(1), diff[:, 1::2].mean(1)).statistic
    lines = [f"# Phase 9 E3 — independent confirmatory point-of-no-return study and predictor test ({a.out})", "",
             f"Summaries: {', '.join(Path(f).name for f in a.confirm)}.",
             f"Fresh seeds (n = {n}): {seeds[0]}–{seeds[-1]}. Movement restriction from u100 for d = 300; 250-update budget. "
             "Negative = damage beyond lost time.", "",
             "## Primary 1: point of no return", "",
             f"- **L_300 = {fmt(L)}**; TOST ±{a.sesoi:g}: p_lower {pl:.4f}, p_upper {ph:.4f} → {dec}",
             f"- A_300 = {fmt(A)}; split-half (odd/even checkpoints) Spearman {r_oe:+.2f}, Spearman–Brown {2 * r_oe / (1 + r_oe):.2f}"]
    if a.j5:
        j5 = [load(f, with_predictors=False) for f in a.j5]
        Lp = np.concatenate([L, cat(j5, "L")])
        Ap = np.concatenate([A, cat(j5, "A")])
        p1, p2 = tost(Lp, a.sesoi)
        lines.append(f"- Descriptive pooling with Phase 8 J5 seeds 100–107 (n = {len(Lp)}): L_300 {fmt(Lp)} (TOST p = {max(p1, p2):.4f}); A_300 {fmt(Ap)}")
    lines += ["", "## Primary 2: removal-time predictors vs A_300 (one-sided, expected ρ < 0; Holm across 4)", "",
              "| predictor | split-half rel. | ρ with A_300 | 95% CI for ρ | one-sided p | Holm p | supported? | LOO R² gain | ρ with L_300 |",
              "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for k in PRED:
        P = cat(us, k)
        rel = stats.spearmanr(cat(us, k + "_h1"), cat(us, k + "_h2")).statistic
        r = stats.spearmanr(P, A).statistic
        t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
        p = float(stats.t.cdf(t, n - 2))  # one-sided, H1: rho < 0
        rows.append((k, rel, r, rho_ci(r, n), p, loo_gain(P, A), stats.spearmanr(P, L).statistic))
    hp = holm(np.array([x[4] for x in rows]))
    for (k, rel, r, (clo, chi), p, g, rl), h in zip(rows, hp):
        lines.append(f"| {k} | {rel:+.2f} | {r:+.2f} | [{clo:+.2f}, {chi:+.2f}] | {p:.4f} | {h:.4f} | "
                     f"{'**yes**' if h < 0.05 else 'no'} | {100 * g:+.0f}% | {rl:+.2f} |")
    sup = [rows[i][0] for i in range(4) if hp[i] < 0.05]
    lo_bounds = {x[0]: x[3][0] for x in rows}
    lines += ["", ("Pre-registered conclusion: **supported predictor(s): " + ", ".join(sup) + "**") if sup else
              ("Pre-registered conclusion: **no predictor supported.** Strongest negative associations compatible with the data "
               "(lower 95% bounds): " + ", ".join(f"{k} {v:+.2f}" for k, v in lo_bounds.items()) + ".")]
    lines += ["", "Per-seed (seed: L_300, A_300): " + "; ".join(f"{sd}: {l_:+.1f}, {a_:+.1f}" for sd, l_, a_ in zip(seeds, L, A))]
    out = Path("results/summary") / f"{a.out}-analysis.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

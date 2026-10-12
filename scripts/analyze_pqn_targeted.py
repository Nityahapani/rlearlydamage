"""Analyze Phase 6b targeted vs sham replay purge (scripts/pqn_targeted_purge.py). Pre-registered (LOG.md) before runs.

Primary metric: eval_eps05_return_mean (greedy secondary via --metric). Primary endpoint: AUC = mean eval over +10..+650.
Secondary endpoints: early +10..+150, middle +160..+240, late +250..+650.
Primary contrasts (paired by seed; 95% t-CI; Holm across the three primary AUC tests; seed-level differences shown):
  targeted vs sham        B_targeted - C_sham
  targeted purge vs retain B_targeted - A_retain
  residual network impairment B_targeted - D_cleannet  (identical buffer, start state and pool; only params+optimizer differ)
Secondary (descriptive): C_sham - A_retain (refresh without removing restricted data); share of B - A explained by
targeting = (B - C) / (B - A); sign-flip permutation p for the primary AUC contrasts (exact, 2^n).
Usage: python scripts/analyze_pqn_targeted.py results/summary/<p6b>.json [...] --tag p6b_targeted
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ARMS = ("A_retain", "B_targeted", "C_sham", "D_cleannet")
COL = {"A_retain": "tab:red", "B_targeted": "tab:blue", "C_sham": "tab:orange", "D_cleannet": "k"}


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2 or np.allclose(x, x[0]):
        return m, m, m, 1.0
    se = x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h, float(2 * stats.t.sf(abs(m / se), n - 1))


def fmt(x):
    m, lo, hi, _ = tci(x)
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]" + ("*" if (lo > 0 or hi < 0) else "")


def holm(ps):
    order = np.argsort(ps); k = len(ps); adj = np.empty(k); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (k - r) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def signflip_p(x):
    x = np.asarray(x, float); obs = abs(x.mean())
    if len(x) > 16:
        rng = np.random.default_rng(0); s = rng.choice([-1, 1], size=(200_000, len(x)))
        return float((np.abs((s * x).mean(1)) >= obs - 1e-12).mean())
    sims = [abs((np.array(s) * x).mean()) for s in itertools.product([-1, 1], repeat=len(x))]
    return float(np.mean(np.array(sims) >= obs - 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--metric", default="eval_eps05_return_mean")
    ap.add_argument("--tag", default="p6b_targeted")
    a = ap.parse_args()
    S = [json.loads(Path(f).read_text()) for f in a.summaries]
    for t in S[1:]:
        assert t["eval_updates"] == S[0]["eval_updates"] and t["blocks"]["K_rows"] == S[0]["blocks"]["K_rows"], "protocol mismatch"
    seeds = sum((t["seeds"] for t in S), []); n = len(seeds)
    assert len(set(seeds)) == n, "duplicate seeds"
    U = np.array(S[0]["eval_updates"]); w1 = int(U[0]); E = int(U[1] - U[0]); R = int(U[-1] - w1); h = U - w1
    ev = {k: np.concatenate([np.array(t["arms"][k][a.metric], float) for t in S]) for k in ARMS}
    sr = {k: np.concatenate([np.array(t["arms"][k]["restricted_share"], float) for t in S]) for k in ARMS}
    sp = {k: np.concatenate([np.array(t["arms"][k]["replacement_share"], float) for t in S]) for k in ARMS}
    win = {"AUC": (E, R), "early": (10, 150), "middle": (160, 240), "late": (250, R)}
    sc = {k: {w: ev[k][:, (h >= lo) & (h <= hi)].mean(1) for w, (lo, hi) in win.items()} for k in ARMS}
    cols = list(win)
    prim = {"targeted vs sham: B - C": ("B_targeted", "C_sham"), "targeted purge vs retain: B - A": ("B_targeted", "A_retain"),
            "residual network impairment: B - D": ("B_targeted", "D_cleannet")}
    sec = {"sham refresh vs retain: C - A": ("C_sham", "A_retain")}
    blk = S[0]["blocks"]
    L = [f"# Phase 6b targeted vs sham replay purge — {a.tag}", "", f"Manifests: {', '.join(t['manifest'] for t in S)}.",
         f"Seeds (n = {n}): {seeds}. Metric: {a.metric}. Removal u{w1}; recovery {R}; eval every {E}.",
         f"Replaced rows: K = {blk['K_rows']} rows x 128 envs; B = rows written u{blk['B_rows_written']}, C (sham) = rows written u{blk['C_rows_written']}.",
         f"Consistency checks: {[t['consistency_check'] for t in S]}.",
         f"Pool collection-episode mean return per seed: {[t['pool_stats']['mean_return'] for t in S]}.", "",
         "## Arm scores (mean over seeds)", "", "| arm | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in ARMS:
        L.append(f"| {k} | " + " | ".join(f"{sc[k][c].mean():.1f}" for c in cols) + " |")
    diffs = {name: {c: sc[p][c] - sc[q][c] for c in cols} for name, (p, q) in {**prim, **sec}.items()}
    ps = np.array([tci(diffs[nm]["AUC"])[3] for nm in prim]); adj = holm(ps)
    L += ["", "## Primary contrasts (paired; positive = first arm better). Primary endpoint AUC; windows secondary.", "",
          "| contrast | " + " | ".join(cols) + " | AUC t-test p | Holm p | sign-flip p | seeds > 0 |", "|---" * (len(cols) + 5) + "|"]
    for (nm, pr, pa) in zip(prim, ps, adj):
        x = diffs[nm]["AUC"]
        L.append(f"| {nm} | " + " | ".join(fmt(diffs[nm][c]) for c in cols) + f" | {pr:.4f} | {pa:.4f} | {signflip_p(x):.4f} | {(x > 0).sum()}/{n} |")
    L += ["", "## Secondary", "", "| contrast | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for nm in sec:
        L.append(f"| {nm} | " + " | ".join(fmt(diffs[nm][c]) for c in cols) + " |")
    ba, bc = diffs["targeted purge vs retain: B - A"]["AUC"].mean(), diffs["targeted vs sham: B - C"]["AUC"].mean()
    if abs(ba) > 1e-9:
        L.append(f"\nShare of the targeted-purge benefit attributable to targeting (B - C) / (B - A), AUC means: {bc / ba:.0%}.")
    L += ["", f"## Paired seed-level differences (AUC)", "", "| seed | " + " | ".join(list(prim) + list(sec)) + " |",
          "|---" * (len(prim) + len(sec) + 1) + "|"]
    for i, s in enumerate(seeds):
        L.append(f"| {s} | " + " | ".join(f"{diffs[nm]['AUC'][i]:+.1f}" for nm in list(prim) + list(sec)) + " |")
    L += ["| between-seed SD | " + " | ".join(f"{diffs[nm]['AUC'].std(ddof=1):.1f}" for nm in list(prim) + list(sec)) + " |"]
    marks = [0, 10, 50, 100, 140, 150, 200, 250, 300]
    L += ["", "## Buffer composition at eval checkpoints (mean over seeds): restricted-era share / replacement share", "",
          "| arm | " + " | ".join(f"+{m}" for m in marks) + " |", "|---" * (len(marks) + 1) + "|"]
    for k in ARMS:
        L.append(f"| {k} | " + " | ".join(f"{sr[k][:, list(h).index(m)].mean():.2f} / {sp[k][:, list(h).index(m)].mean():.2f}" for m in marks) + " |")
    expo = {k: sp[k][:, 1:].mean() for k in ARMS}
    L += ["", "Mean replacement share over the AUC window (sampling is uniform, so = expected sampled share): "
          + ", ".join(f"{k} {v:.3f}" for k, v in expo.items()) + "."]
    tr = {k: np.concatenate([np.array(t["arms"][k]["train"]["loss"], float) for t in S]) for k in ARMS}
    L += ["", "TD loss, first 30 post-removal updates (mean): " + ", ".join(f"{k} {tr[k][:, :30].mean():.3f}" for k in ARMS)]
    out = Path("results/summary") / f"{a.tag}-analysis.md"
    out.write_text("\n".join(L) + "\n"); print("\n".join(L)); print("wrote", out)

    f, ax = plt.subplots(1, 3, figsize=(19, 4.6), gridspec_kw=dict(width_ratios=[1.3, 0.8, 1]))
    for k in ARMS:
        mu = ev[k].mean(0); se = ev[k].std(0, ddof=1) / np.sqrt(n)
        ax[0].plot(h, mu, color=COL[k], label=k, lw=1.4); ax[0].fill_between(h, mu - se, mu + se, color=COL[k], alpha=0.12)
    ax[0].set_xlabel(f"updates after removal (u{w1})"); ax[0].set_ylabel(a.metric); ax[0].legend(fontsize=8); ax[0].set_title("Targeted vs sham purge")
    for k, ls in (("A_retain", "-"), ("B_targeted", "-"), ("C_sham", "-")):
        ax[1].plot(h, sr[k].mean(0), color=COL[k], ls=ls, label=f"{k} restricted")
        ax[1].plot(h, sp[k].mean(0), color=COL[k], ls=":", label=f"{k} replacement")
    ax[1].set_xlim(0, 300); ax[1].set_xlabel("updates after removal"); ax[1].set_ylabel("share of buffer"); ax[1].legend(fontsize=7)
    names = list(prim) + list(sec)
    for j, nm in enumerate(names):
        x = diffs[nm]["AUC"]; m, lo, hi, _ = tci(x)
        ax[2].scatter(x, np.full(n, j) + np.linspace(-0.15, 0.15, n), s=14, color="0.5", zorder=2)
        ax[2].errorbar(m, j, xerr=[[m - lo], [hi - m]], fmt="o", color="k", capsize=4, zorder=3)
    ax[2].set_yticks(range(len(names)), [nm.split(":")[1].strip() for nm in names], fontsize=8); ax[2].invert_yaxis()
    ax[2].axvline(0, color="k", lw=0.8); ax[2].set_xlabel("paired AUC difference (dots = seeds; bar = 95% t-CI)")
    f.tight_layout(); f.savefig(Path("figures") / f"{a.tag}.png", dpi=130)
    print("figure:", f"figures/{a.tag}.png")


if __name__ == "__main__":
    main()

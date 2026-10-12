"""Analyze the Phase 6 replay-purge experiment (scripts/pqn_purge.py). Pre-registered before the real runs (see LOG.md).

Primary metric: eval_eps05_return_mean (Phase 5 metric rule; greedy = secondary via --metric).
Windows (post-removal updates; Phase 5 AUC definition = mean over checkpoints +E..+R):
  AUC  +10..+650 | early +10..+150 (A's restricted share is flat, Phase 6) | mid +160..+240 (draining)
  late +250..+650 (A's restricted data gone) | post-delay +60..+150 (after C's purge, before A's data drains)
Primary contrasts (paired by seed, 95% t-CI; Holm correction across the three primary AUC tests):
  purge rescue        B_purge - A_retain
  delayed-purge effect B_purge - C_delay50
  residual impairment B_purge - D_clean
Secondary: E_clean_purge - D_clean (purge procedure / replacement data on a clean network);
  (B - A) - (E - D) (purge benefit specific to the damaged agent); C - A; A - D (total damage, retained buffer).
Seed robustness per contrast: between-seed SD, #seeds < 0, 95% prediction interval mean +/- t*sd*sqrt(1+1/n).
Usage: python scripts/analyze_pqn_purge.py results/summary/<p6 purge>.json [...] --tag p6_purge
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ARMS = ("A_retain", "B_purge", "C_delay50", "D_clean", "E_clean_purge")
COL = {"A_retain": "tab:red", "B_purge": "tab:blue", "C_delay50": "tab:purple", "D_clean": "k", "E_clean_purge": "0.55"}


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2 or np.allclose(x, x[0]):
        return m, m, m, 1.0
    se = x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1)
    return m, m - h, m + h, p


def fmt(x):
    m, lo, hi, _ = tci(x)
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]" + ("*" if (lo > 0 or hi < 0) else "")


def holm(ps):
    order = np.argsort(ps); k = len(ps); adj = np.empty(k); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (k - r) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--metric", default="eval_eps05_return_mean")
    ap.add_argument("--tag", default="p6_purge")
    a = ap.parse_args()
    S = [json.loads(Path(f).read_text()) for f in a.summaries]
    for t in S[1:]:
        assert t["eval_updates"] == S[0]["eval_updates"] and t["protocol"]["delay"] == S[0]["protocol"]["delay"], "protocol mismatch"
    seeds = sum((t["seeds"] for t in S), []); n = len(seeds)
    assert len(set(seeds)) == n, "duplicate seeds"
    U = np.array(S[0]["eval_updates"]); w1 = int(U[0]); E = int(U[1] - U[0]); R = int(U[-1] - w1)
    h = U - w1
    delay = S[0]["protocol"]["delay"]
    ev = {k: np.concatenate([np.array(t["arms"][k][a.metric], float) for t in S]) for k in ARMS}
    sh = {k: np.concatenate([np.array(t["arms"][k]["restricted_share"], float) for t in S]) for k in ARMS}
    for k in ("D_clean", "E_clean_purge"):  # clean agents: tagged-window data were collected unrestricted
        sh[k + " (window-era, unrestricted)"] = sh[k]; sh[k] = np.zeros_like(sh[k])
    win = {"AUC": (E, R), "early": (10, 150), "mid": (160, 240), "late": (250, R), "post-delay": (delay + E, 150)}
    sel = {w: (h >= lo) & (h <= hi) for w, (lo, hi) in win.items()}
    sc = {k: {w: ev[k][:, s].mean(1) for w, s in sel.items()} for k in ARMS}
    for k in ARMS:
        for r in (10, 100):
            sc[k][f"+{r}"] = ev[k][:, [i for i, x in enumerate(h) if abs(x - r) <= E and x > 0]].mean(1)
    cols = list(sc["A_retain"].keys())
    prim = {"purge rescue: B - A": ("B_purge", "A_retain"), "delayed-purge effect: B - C": ("B_purge", "C_delay50"),
            "residual impairment: B - D": ("B_purge", "D_clean")}
    sec = {"purge procedure on clean net: E - D": lambda w: sc["E_clean_purge"][w] - sc["D_clean"][w],
           "damage-specific purge benefit: (B - A) - (E - D)": lambda w: (sc["B_purge"][w] - sc["A_retain"][w]) - (sc["E_clean_purge"][w] - sc["D_clean"][w]),
           "C - A": lambda w: sc["C_delay50"][w] - sc["A_retain"][w],
           "total damage, retained buffer: A - D": lambda w: sc["A_retain"][w] - sc["D_clean"][w]}
    allc = {**{k: (lambda w, p=p: sc[p[0]][w] - sc[p[1]][w]) for k, p in prim.items()}, **sec}

    L = [f"# Phase 6 replay purge — {a.tag}", "", f"Manifests: {', '.join(t['manifest'] for t in S)}.",
         f"Seeds (n = {n}): {seeds}. Metric: {a.metric}. Removal at u{w1}; recovery {R}; eval every {E}; delayed purge at +{delay}.",
         f"Consistency checks: {[t['consistency_check'] for t in S]}.",
         f"Replacement buffer (frozen clean actor, eps {S[0]['protocol']['replacement']['eps']}): collection-episode mean return "
         f"{[t['collection_stats']['mean_return'] for t in S]}.", "",
         "Windows (post-removal updates): " + "; ".join(f"{w} +{lo}..+{hi}" for w, (lo, hi) in win.items()) + ". +h readouts average checkpoints h-E..h+E.", "",
         "## Arm scores (mean over seeds)", "", "| arm | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in ARMS:
        L.append(f"| {k} | " + " | ".join(f"{sc[k][c].mean():.1f}" for c in cols) + " |")
    ps = np.array([tci(sc[p[0]]["AUC"] - sc[p[1]]["AUC"])[3] for p in prim.values()])
    adj = holm(ps)
    L += ["", "## Primary contrasts (paired; positive = first arm better)", "", "| contrast | " + " | ".join(cols) + " | AUC p (Holm) |",
          "|---" * (len(cols) + 2) + "|"]
    for (name, p), pa, pr in zip(prim.items(), adj, ps):
        L.append(f"| {name} | " + " | ".join(fmt(sc[p[0]][c] - sc[p[1]][c]) for c in cols) + f" | {pr:.3f} ({pa:.3f}) |")
    L += ["", "## Secondary contrasts", "", "| contrast | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, f in sec.items():
        L.append(f"| {name} | " + " | ".join(fmt(f(c)) for c in cols) + " |")
    L += ["", f"## Seed robustness (AUC, n = {n})", "",
          "| contrast | mean | between-seed SD | seeds < 0 | 95% prediction interval | per-seed (" + ", ".join(map(str, seeds)) + ") |",
          "|---|---|---|---|---|---|"]
    for name, f in allc.items():
        x = f("AUC"); sd = x.std(ddof=1) if n > 1 else float("nan")
        hw = stats.t.ppf(0.975, n - 1) * sd * np.sqrt(1 + 1 / n) if n > 1 else float("nan")
        L.append(f"| {name} | {x.mean():+.1f} | {sd:.1f} | {(x < 0).sum()}/{n} | [{x.mean() - hw:+.1f}, {x.mean() + hw:+.1f}] | "
                 + ", ".join(f"{v:+.1f}" for v in x) + " |")
    marks = [0, 10, delay, delay + E, 100, 150, 200, 250, 300]
    L += ["", "## Restriction-era share of each buffer at eval checkpoints (mean over seeds)", "",
          "| arm | " + " | ".join(f"+{m}" for m in marks) + " |", "|---" * (len(marks) + 1) + "|"]
    for k, v in sh.items():
        L.append(f"| {k} | " + " | ".join(f"{v[:, list(h).index(m)].mean():.3f}" for m in marks) + " |")
    # descriptive check of the diagnostic pattern
    m = {k: sc[k]["AUC"].mean() for k in ARMS}
    L += ["", "## Diagnostic pattern (AUC means; CIs above)", "",
          f"- B > A: {m['B_purge'] > m['A_retain']} ({m['B_purge']:.1f} vs {m['A_retain']:.1f})",
          f"- A <= C <= B: {m['A_retain'] <= m['C_delay50'] <= m['B_purge']} (C {m['C_delay50']:.1f})",
          f"- B < D: {m['B_purge'] < m['D_clean']} (D {m['D_clean']:.1f})"]
    tr = {k: np.concatenate([np.array(t["arms"][k]["train"]["loss"], float) for t in S]) for k in ARMS}
    L += ["", "TD loss, first 30 post-removal updates (mean): " + ", ".join(f"{k} {tr[k][:, :30].mean():.3f}" for k in ARMS)]
    out = Path("results/summary") / f"{a.tag}-analysis.md"
    out.write_text("\n".join(L) + "\n"); print("\n".join(L)); print("wrote", out)

    f, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 4.5), gridspec_kw=dict(width_ratios=[1.3, 0.8, 1]))
    for k in ARMS:
        M = ev[k]; mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(n) if n > 1 else 0 * mu
        a1.plot(h, mu, color=COL[k], label=k, lw=1.4); a1.fill_between(h, mu - se, mu + se, color=COL[k], alpha=0.12)
    a1.axvline(delay, color="tab:purple", ls=":", lw=1); a1.set_xlabel(f"updates after removal (u{w1})"); a1.set_ylabel(a.metric)
    a1.legend(fontsize=8); a1.set_title("Replay purge: recovery")
    for k in ("A_retain", "C_delay50", "B_purge"):
        a2.plot(h, sh[k].mean(0), color=COL[k], label=k)
    a2.set_xlabel("updates after removal"); a2.set_ylabel("restriction-era share of buffer"); a2.legend(fontsize=8); a2.set_xlim(0, 300)
    names = list(allc.keys())
    for j, (w, off) in enumerate((("AUC", -0.27), ("early", 0.0), ("late", 0.27))):
        mu = np.array([allc[c](w).mean() for c in names]); ci = np.array([tci(allc[c](w))[1:3] for c in names])
        a3.barh(np.arange(len(names)) + off, mu, height=0.26, xerr=[mu - ci[:, 0], ci[:, 1] - mu], capsize=2, label=w,
                color=["0.3", "0.6", "0.85"][j])
    a3.set_yticks(range(len(names)), [x.split(":")[0] if ":" in x else x for x in names], fontsize=7); a3.invert_yaxis()
    a3.axvline(0, color="k", lw=0.8); a3.legend(fontsize=8); a3.set_xlabel("paired difference (95% t-CI)")
    f.tight_layout(); f.savefig(Path("figures") / f"{a.tag}.png", dpi=130)
    print("figure:", f"figures/{a.tag}.png")


if __name__ == "__main__":
    main()

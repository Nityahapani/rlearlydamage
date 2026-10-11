"""Analyze PPO state surgery (scripts/surgery_ppo.py) from the committed summary.

Scores per arm and seed (deficit-free eval, see recovery_test.py):
  AUC  : mean eval over all post-surgery checkpoints +E..+R (primary readout)
  +0   : eval at surgery (depends on the actor only, a sanity check)
  +50, +100 : mean of the 3 checkpoints centred on the horizon; +R: mean of the last 2
Effects (paired by seed; CI = paired t-interval over seeds when n < 10, else bootstrap):
  main effect of carrier X  = mean(arms with X=D) - mean(arms with X=C)   (negative = X carries harm)
  share of total damage     = main effect / (DDD - CCC)
  transplant (harm)         = XCC-type arm - CCC: the damaged carrier alone, in an otherwise clean agent
  rescue                    = clean carrier alone in an otherwise damaged agent, minus DDD
  2-way interactions        = half the difference of differences, averaged over the third carrier
  data arms                 = CCC+bD - CCC and DDD+bC - DDD
Usage: python scripts/analyze_surgery.py results/summary/<p3 summary>.json --tag surgery_move_late
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CARR = ("actor", "critic", "opt")


def boot(x, n=5000, seed=0):
    """95% CI for the mean. Paired t-interval when fewer than 10 seeds (a percentile bootstrap is
    anti-conservative at n = 4); percentile bootstrap otherwise."""
    x = np.asarray(x, float)
    if len(x) < 10:
        from scipy import stats
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return np.array([x.mean() - h, x.mean() + h])
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def fmt(x):
    lo, hi = boot(x)
    return f"{np.mean(x):+.1f} [{lo:+.1f}, {hi:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--tag", default="surgery")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    U = np.array(s["eval_updates"])
    w1 = U[0]
    E = U[1] - U[0]
    R = U[-1] - w1
    ev = {k: np.array(v["eval_return_mean"], float) for k, v in s["arms"].items()}  # seeds x checkpoints
    idx = {int(u - w1): i for i, u in enumerate(U)}

    def readouts(M):
        out = {"AUC": M[:, 1:].mean(1), "+0": M[:, 0]}
        for h in (50, 100):
            out[f"+{h}"] = M[:, [idx[h - E], idx[h], idx[h + E]]].mean(1)
        out[f"+{R}"] = M[:, [idx[R - E], idx[R]]].mean(1)
        return out

    sc = {k: readouts(v) for k, v in ev.items()}
    keys = list(sc["CCC"].keys())
    codes = [c for c in sc if len(c) == 3]

    L = ["### Arm scores (mean over seeds, deficit-free eval)", "",
         "| arm | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for k in codes + [c for c in sc if len(c) > 3]:
        L.append(f"| {k} | " + " | ".join(f"{sc[k][r].mean():.1f}" for r in keys) + " |")
    total = {r: sc["DDD"][r] - sc["CCC"][r] for r in keys}
    L += ["", f"Total damage (DDD - CCC): " + ", ".join(f"{r} {fmt(total[r])}" for r in keys), ""]

    L += ["### Main effects (mean[X=D] - mean[X=C]; negative = carrier holds harm) and share of total damage", "",
          "| carrier | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    main = {}
    for i, c in enumerate(CARR):
        main[c] = {r: np.mean([sc[k][r] for k in codes if k[i] == "D"], 0) - np.mean([sc[k][r] for k in codes if k[i] == "C"], 0)
                   for r in keys}
        cells = []
        for r in keys:
            share = np.mean(main[c][r]) / np.mean(total[r]) if abs(np.mean(total[r])) > 1e-6 else np.nan
            cells.append(f"{fmt(main[c][r])} ({100*share:.0f}%)")
        L.append(f"| {c} | " + " | ".join(cells) + " |")

    L += ["", "### Single-carrier transplants (harm, vs CCC) and rescues (vs DDD)", "",
          "| comparison | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for i, c in enumerate(CARR):
        harm = "".join("D" if j == i else "C" for j in range(3))
        resc = "".join("C" if j == i else "D" for j in range(3))
        L.append(f"| damaged {c} only ({harm} - CCC) | " + " | ".join(fmt(sc[harm][r] - sc['CCC'][r]) for r in keys) + " |")
        L.append(f"| clean {c} only ({resc} - DDD) | " + " | ".join(fmt(sc[resc][r] - sc['DDD'][r]) for r in keys) + " |")

    L += ["", "### Two-way interactions (averaged over the third carrier)", "",
          "| pair | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for (i, ci), (j, cj) in combinations(enumerate(CARR), 2):
        def m(vi, vj, r):
            return np.mean([sc[k][r] for k in codes if k[i] == vi and k[j] == vj], 0)
        L.append(f"| {ci} x {cj} | " + " | ".join(
            fmt(0.5 * ((m("D", "D", r) - m("D", "C", r)) - (m("C", "D", r) - m("C", "C", r)))) for r in keys) + " |")

    if "CCC+bD" in sc:
        L += ["", f"### Data arms (behaviour-policy swap for the first {s['protocol']['behaviour_k']} updates)", "",
              "| comparison | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|",
              "| damaged data into clean agent (CCC+bD - CCC) | " + " | ".join(fmt(sc['CCC+bD'][r] - sc['CCC'][r]) for r in keys) + " |",
              "| clean data into damaged agent (DDD+bC - DDD) | " + " | ".join(fmt(sc['DDD+bC'][r] - sc['DDD'][r]) for r in keys) + " |"]
    print("\n".join(L))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    x = (U - w1) * s["protocol"]["steps_per_update"] / 1e6
    for k in codes:
        col = "tab:red" if k[0] == "D" else "tab:blue"
        ls = "--" if k[1] == "D" else "-"
        mk = "o" if k[2] == "D" else None
        ax.plot(x, ev[k].mean(0), color=col, ls=ls, marker=mk, ms=3, lw=1.6, label=k)
    for k, col in (("CCC+bD", "tab:purple"), ("DDD+bC", "tab:green")):
        if k in ev:
            ax.plot(x, ev[k].mean(0), color=col, ls=":", lw=1.8, label=k)
    ax.set_xlabel("env steps since surgery (M)"); ax.set_ylabel("deficit-free eval return")
    ax.set_title("surgery arms (code = actor, critic, opt; C clean / D damaged)\nred = damaged actor, dashed = damaged critic, dots = damaged opt", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    width = 0.25
    for i, c in enumerate(CARR):
        means = [np.mean(main[c][r]) for r in keys]
        cis = np.array([boot(main[c][r]) for r in keys])
        pos = np.arange(len(keys)) + (i - 1) * width
        ax.bar(pos, means, width, label=c, yerr=[np.array(means) - cis[:, 0], cis[:, 1] - np.array(means)], capsize=3)
    ax.plot(np.arange(len(keys)), [np.mean(total[r]) for r in keys], "k_", ms=25, mew=2, label="total (DDD-CCC)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(keys)
    ax.set_ylabel("main effect on eval return (D - C)")
    ax.set_title("which carrier holds the damage? (negative = harm; 95% t-interval over seeds)", fontsize=9)
    ax.legend(fontsize=8)
    fig.suptitle(f"State surgery at removal of late movement restriction ({len(s['seeds'])} seeds)")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = Path("figures") / f"{a.tag}.png"
    fig.savefig(out, dpi=130)
    print("figure:", out)


if __name__ == "__main__":
    main()

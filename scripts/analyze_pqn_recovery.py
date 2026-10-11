"""Analyze Phase 5 PQN / PQN-R recovery runs (scripts/pqn_recovery.py).

Inputs: one or more recovery summaries per variant (control + early + late may live in different files; arms are
merged by variant, and the control must cover the arms' horizon). Readouts, paired by seed, 95% t-CI (n < 10):
  gap(h)            = control(r + h) - arm(r + h), r = removal update; positive = damage
  delay-adjusted(h) = control(r + h - L) - arm(r + h), L = window length (lost-time null; positive = damage
                      beyond lost training time)
  AUC               = mean over post-removal checkpoints +E..+H of control - arm
Horizons: +0 (at removal), +10, +100 (409,600 env steps), +500 (2.05M env steps, = PPO's Q4 readout), end.
Also the control's learning-curve slope (eval gain per 100 updates) in successive blocks: is it slowing?
Usage: python scripts/analyze_pqn_recovery.py results/summary/<p5 recovery>.json [...] --tag p5_recovery
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


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.array([x.mean(), x.mean()])
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return np.array([x.mean() - h, x.mean() + h])


def fmt(x):
    lo, hi = ci(x)
    s = "*" if (lo > 0 or hi < 0) else ""
    return f"{np.mean(x):+.1f} [{lo:+.1f}, {hi:+.1f}]{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--tag", default="p5_recovery")
    ap.add_argument("--metric", default="auto", choices=["auto", "eval_return_mean", "eval_eps05_return_mean"],
                    help="auto = the eval with lower control checkpoint-to-checkpoint noise (prespecified rule)")
    a = ap.parse_args()
    data = {}  # variant -> arm -> dict(eval, U, window, manifest)
    for p in a.summaries:
        s = json.loads(Path(p).read_text())
        v = s["protocol"]["variant"]
        for arm, d in s["arms"].items():
            data.setdefault(v, {})[arm] = dict(evals={k: np.array(d[k], float) for k in ("eval_return_mean", "eval_eps05_return_mean") if k in d},
                                               U=np.array(s["eval_updates"]),
                                               window=d["window_updates"], manifest=s["manifest"], protocol=s["protocol"])
    L = ["# Phase 5 — PQN / PQN-R recovery replication (SpaceInvaders-MinAtar, movement restriction {noop, fire})", "",
         "Deficit-free eval, greedy and eps 0.05 (primary metric chosen per variant below; full action set, clean obs, true reward, common random numbers); paired t-CIs over seeds; "
         "* = CI excludes 0. gap = control − arm (positive = damage); delay-adjusted = control(u − L) − arm(u).", ""]
    fig, axes = plt.subplots(1, len(data), figsize=(7 * len(data), 4.3), squeeze=False)
    for j, (v, arms) in enumerate(sorted(data.items())):
        c = arms["control"]
        Uc = c["U"]
        # reliability (control only, after the epsilon floor): mean |checkpoint-to-checkpoint change| / mean level
        rel = {}
        for k, M in c["evals"].items():
            sel = Uc >= 250
            rel[k] = float(np.abs(np.diff(M[:, sel], axis=1)).mean() / max(M[:, sel].mean(), 1e-8))
        metric = min(rel, key=rel.get) if a.metric == "auto" else a.metric
        for d in arms.values():
            d["eval"] = d["evals"][metric]
        Ec = c["eval"]
        E = int(Uc[1] - Uc[0])
        exp = c["protocol"]["exposure"]["pqn"]
        L += [f"## {v.upper()}  (manifests: " + ", ".join(sorted({d['manifest'] for d in arms.values()})) + ")", "",
              f"Exposure per update: {exp['steps_per_update']} env steps, {exp['grad_steps_per_update']} grad steps of {exp['batch']}, "
              f"replay ratio {exp['replay_ratio']}, buffer {exp['buffer_transitions']} transitions "
              f"(turnover {exp['buffer_turnover_updates']:.1f} updates); {exp['eps_schedule']}; LR {exp['lr']}.", "",
              "Control eval reliability after the eps floor (mean |checkpoint-to-checkpoint change| / mean level; lower = better): "
              + ", ".join(f"{k} {r:.3f}" for k, r in rel.items()) + f". **Primary metric: {metric}.** Readouts at +h > 0 average "
              "the checkpoints h-E, h, h+E (where available).", ""]
        # control slope
        blocks = [(b, b + 100) for b in range(0, int(Uc[-1]), 100)]
        sl = []
        for b0, b1 in blocks:
            i0, i1 = np.searchsorted(Uc, b0), np.searchsorted(Uc, b1)
            if i1 < len(Uc):
                sl.append(f"{b0}-{b1}: {Ec[:, i1].mean() - Ec[:, i0].mean():+.1f}")
        tail = Ec[:, -6:].mean(1)
        L += [f"Control eval: " + ", ".join(f"u{u} {Ec[:, np.searchsorted(Uc, u)].mean():.1f}" for u in (0, 100, 250, 500, 750, int(Uc[-1]))
                                              if u <= Uc[-1]) + f"; final (last 50 updates) {tail.mean():.1f} ± {tail.std(ddof=1):.1f} sd. "
              "Eval gain per 100-update block: " + "; ".join(sl) + ".", ""]
        ax = axes[0, j]
        m, se = Ec.mean(0), Ec.std(0, ddof=1) / np.sqrt(len(Ec))
        ax.plot(Uc, m, "k", label="control"); ax.fill_between(Uc, m - se, m + se, color="k", alpha=0.15)
        colors = {"early": "tab:blue", "late": "tab:red"}
        L += ["| arm | window (updates) | horizon | gap | delay-adjusted |", "|---|---|---|---|---|"]
        for arm in ("early", "late"):
            if arm not in arms:
                continue
            d = arms[arm]
            w0, w1 = d["window"]
            Lw = w1 - w0
            U, M = d["U"], d["eval"]
            idx = lambda u: int(np.where(U == u)[0][0])
            cidx = lambda u: int(np.where(Uc == u)[0][0])
            m2, se2 = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
            ax.plot(U, m2, color=colors[arm], label=f"{arm} [{w0},{w1})")
            ax.fill_between(U, m2 - se2, m2 + se2, color=colors[arm], alpha=0.15)
            ax.axvspan(w0, w1, color=colors[arm], alpha=0.08)
            H = int(min(U[-1], Uc[-1]) - w1)
            def sm(X, ix, u):  # 3-checkpoint mean around u (single checkpoint at removal)
                us = [u] if u == w1 else [x for x in (u - E, u, u + E) if w1 < x <= w1 + H]
                return np.mean([X[:, ix(x)] for x in us], 0)

            for h in (0, 10, 100, 500, H):
                if h > H:
                    continue
                u = w1 + h
                gap = sm(Ec, cidx, u) - sm(M, idx, u)
                da = np.mean([Ec[:, cidx(x - Lw)] for x in ([u] if u == w1 else [x for x in (u - E, u, u + E) if w1 < x <= w1 + H])], 0) - sm(M, idx, u)
                L.append(f"| {arm} | [{w0}, {w1}) | +{h} (u{u}) | {fmt(gap)} | {fmt(da)} |")
            post = [u for u in U if w1 < u <= w1 + H]
            auc = np.mean([Ec[:, cidx(u)] - M[:, idx(u)] for u in post], 0)
            auc_da = np.mean([Ec[:, cidx(u - Lw)] - M[:, idx(u)] for u in post], 0)
            L.append(f"| {arm} | [{w0}, {w1}) | AUC +{E}..+{H} | {fmt(auc)} | {fmt(auc_da)} |")
            pre = Ec[:, cidx(w0)] if w0 > 0 else Ec[:, 0]
            L += [f"| {arm} | | at onset: control {pre.mean():.1f}; at removal arm {M[:, idx(w1)].mean():.1f} vs control "
                  f"{Ec[:, cidx(w1)].mean():.1f} | | |"]
        L.append("")
        ax.set_title(f"{v.upper()} on SpaceInvaders-MinAtar"); ax.set_xlabel("update (4096 env steps)")
        ax.set_ylabel(f"deficit-free eval return ({'greedy' if metric == 'eval_return_mean' else 'eps 0.05'})"); ax.legend(fontsize=8)
    out_md = Path("results/summary") / f"{a.tag}-analysis.md"
    out_md.write_text("\n".join(L) + "\n")
    fig.tight_layout(); fig.savefig(Path("figures") / f"{a.tag}.png", dpi=130)
    print("\n".join(L)); print("wrote", out_md, f"figures/{a.tag}.png")


if __name__ == "__main__":
    main()

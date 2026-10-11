"""Analyze Phase 5 PQN-R params x buffer surgery (scripts/pqn_surgery.py).

Arms: CC, DC, CD, DD (network source, buffer source). Paired by seed, 95% t-CI (n < 10), * = CI excludes 0.
  parameter transfer : DC - CC   (damaged network + clean replay vs clean)
  data transfer      : CD - CC   (clean network + damaged replay vs clean)
  network main effect: mean(DC, DD) - mean(CC, CD);  buffer main effect: mean(CD, DD) - mean(CC, DC)
  interaction        : (DD - CD) - (DC - CC)
  rescues            : CD - DD (clean network into a damaged agent), DC - DD (clean buffer into a damaged agent)
Readouts: AUC (mean eval over +E..+R), +0, +10, +20, +30 (around the buffer turnover), +100, end (mean of last 2).
Usage: python scripts/analyze_pqn_surgery.py results/summary/<p5 surgery>.json --metric eval_eps05_return_mean
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
    return f"{np.mean(x):+.1f} [{lo:+.1f}, {hi:+.1f}]" + ("*" if (lo > 0 or hi < 0) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--metric", default="eval_eps05_return_mean")
    ap.add_argument("--tag", default="p5_surgery")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    U = np.array(s["eval_updates"])
    w1 = U[0]
    E = int(U[1] - U[0])
    R = int(U[-1] - w1)
    ix = {int(u - w1): i for i, u in enumerate(U)}
    ev = {k: np.array(v[a.metric], float) for k, v in s["arms"].items()}

    def ro(M):
        out = {"AUC": M[:, 1:].mean(1), "+0": M[:, 0]}
        for h in (10, 20, 30, 100):
            out[f"+{h}"] = M[:, [ix[x] for x in (h - E, h, h + E) if 0 < x <= R]].mean(1)
        out[f"+{R}"] = M[:, [ix[R - E], ix[R]]].mean(1)
        return out

    sc = {k: ro(v) for k, v in ev.items()}
    keys = list(sc["CC"].keys())
    L = [f"# Phase 5 — PQN-R params x replay-buffer surgery — {s['manifest']}", "",
         f"Metric: {a.metric}. Removal at update {w1}; recovery {R} updates; eval every {E}. Assignment rule: "
         f"{s['protocol']['assignment_rule']}. Buffer turnover: {s['protocol']['buffer_turnover_updates']:.1f} updates; "
         f"fraction of D's buffer written under the restriction at removal: {s['frac_D_buffer_written_in_window']:.2f}. "
         f"Buffer pointer/size identical in C and D: {s['buffer_meta_identical']}. Consistency vs recovery run: {s['consistency_check']}.", "",
         "Readouts at +h > 0 average checkpoints h-E, h, h+E; end = mean of last 2.", "",
         "## Arm scores (mean over seeds)", "", "| arm | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for k in ("CC", "DC", "CD", "DD"):
        if k in sc:
            L.append(f"| {k} | " + " | ".join(f"{sc[k][r].mean():.1f}" for r in keys) + " |")
    con = {
        "total damage: DD - CC": lambda r: sc["DD"][r] - sc["CC"][r],
        "parameter transfer: DC - CC": lambda r: sc["DC"][r] - sc["CC"][r],
        "data transfer: CD - CC": lambda r: sc["CD"][r] - sc["CC"][r],
        "network main effect": lambda r: (sc["DC"][r] + sc["DD"][r] - sc["CC"][r] - sc["CD"][r]) / 2,
        "buffer main effect": lambda r: (sc["CD"][r] + sc["DD"][r] - sc["CC"][r] - sc["DC"][r]) / 2,
        "interaction (DD-CD)-(DC-CC)": lambda r: (sc["DD"][r] - sc["CD"][r]) - (sc["DC"][r] - sc["CC"][r]),
        "rescue by clean network: CD - DD": lambda r: sc["CD"][r] - sc["DD"][r],
        "rescue by clean buffer: DC - DD": lambda r: sc["DC"][r] - sc["DD"][r],
    }
    L += ["", "## Contrasts (paired; negative = harm)", "", "| contrast | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, f in con.items():
        L.append(f"| {name} | " + " | ".join(fmt(f(r)) for r in keys) + " |")
    tot = sc["DD"]["AUC"] - sc["CC"]["AUC"]
    if abs(tot.mean()) > 1e-6:
        L += ["", "Share of total AUC damage: network main effect "
              f"{con['network main effect']('AUC').mean() / tot.mean():.0%}, buffer main effect "
              f"{con['buffer main effect']('AUC').mean() / tot.mean():.0%}."]
    tr = {k: np.array(v["train"]["loss"], float) for k, v in s["arms"].items()}
    L += ["", "TD loss over the first 30 post-removal updates (mean): " + ", ".join(f"{k} {tr[k][:, :30].mean():.3f}" for k in tr)]
    out_md = Path("results/summary") / f"{Path(a.summary).stem}-analysis.md"
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote", out_md)

    f, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
    col = {"CC": "k", "DC": "tab:orange", "CD": "tab:purple", "DD": "tab:red"}
    x = U - w1
    for k in ("CC", "DC", "CD", "DD"):
        M = ev[k]
        m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        a1.plot(x, m, color=col[k], label=f"{k} (net {k[0]}, buffer {k[1]})"); a1.fill_between(x, m - se, m + se, color=col[k], alpha=0.15)
    a1.axvline(s["protocol"]["buffer_turnover_updates"], color="0.5", ls=":", label="buffer fully replaced")
    a1.set_xlabel(f"updates after removal (u{w1})"); a1.set_ylabel(a.metric); a1.legend(fontsize=8); a1.set_title("PQN-R: network x replay buffer")
    names = ["DD - CC", "DC - CC (params)", "CD - CC (data)", "network main", "buffer main", "interaction"]
    fs = [con["total damage: DD - CC"], con["parameter transfer: DC - CC"], con["data transfer: CD - CC"],
          con["network main effect"], con["buffer main effect"], con["interaction (DD-CD)-(DC-CC)"]]
    for j, (r, off) in enumerate((("AUC", -0.2), ("+10", 0.2))):
        mu = [f(r).mean() for f in fs]
        lo = [f(r).mean() - ci(f(r))[0] for f in fs]
        hi = [ci(f(r))[1] - f(r).mean() for f in fs]
        a2.barh(np.arange(len(fs)) + off, mu, height=0.38, xerr=[lo, hi], capsize=3, label=r, color=["0.45", "0.75"][j])
    a2.set_yticks(range(len(fs)), names, fontsize=8); a2.invert_yaxis(); a2.axvline(0, color="k", lw=0.8)
    a2.legend(); a2.set_xlabel("paired difference (95% t-CI)")
    f.tight_layout(); f.savefig(Path("figures") / f"{a.tag}.png", dpi=130)
    print("figure:", f"figures/{a.tag}.png")


if __name__ == "__main__":
    main()

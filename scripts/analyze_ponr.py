"""Analysis of a Phase 7 point-of-no-return sweep (scripts/ponr_sweep.py output).

Outcomes per seed and duration d (removal w1 = w0 + d), budget R:
  L_d  late delay-adjusted deficit: mean_{k in last 5 checkpoints} D_online(w1 + k) - C(w0 + k)   [gate outcome]
  A_d  delay-adjusted AUC:          mean_{k = E..R}              D_online(w1 + k) - C(w0 + k)
  raw  AUC vs clean continuation:   mean_k D_online(w1 + k) - C(w1 + k)
  transplant rescue: D_transplant - D_online (AUC), D_transplant - C(w1 + k) (AUC), late transplant vs pause ref
Paired-t 95% CIs across seeds. Predictor values at removal (D) and their clean-matched differences (D - C at
the same update) are tabulated and, in the pilot, related to L_d / A_d descriptively (Spearman over seed x d).

Usage: python scripts/analyze_ponr.py results/summary/<stem>.json --out p7_ponr_pilot [--gate]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from rled.probes import PROBE_KEYS  # noqa: E402


def tci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else np.nan
    return m, m - h, m + h, p


def fmt(t, d=1):
    m, lo, hi, p = t
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", action="store_true", help="evaluate the Stage 0 gate fixed in LOG.md")
    a = ap.parse_args()
    s = json.load(open(a.summary))
    w0, durs, R = s["onset"], s["durations"], s["recovery"]
    E = s["protocol"]["eval_every"]
    K = R // E
    C = np.array(s["C"]["eval_return_mean"])           # seeds x (dmax + R)/E + 1, starting at w0
    Dm = np.array(s["D_mask"]["eval_return_mean"])     # seeds x dmax/E + 1
    seeds = s["seeds"]
    n = len(seeds)
    lines = [f"# Phase 7 point-of-no-return sweep — {a.out}", "",
             f"Source: `{Path(a.summary).name}`. Seeds {seeds} (n = {n}). Onset w0 = {w0}; durations {durs}; "
             f"budget R = {R} updates; eval every {E}. Restriction: allowed actions {s['protocol']['allowed']}.", "",
             "Delay-adjusted reference = clean trajectory at the same number of updates *after onset* "
             "(pause-equivalent: what the agent would have if training had simply been paused during the window). "
             "Negative = damage beyond lost time. Paired-t 95% CIs; * = CI excludes 0.", ""]

    rows, units = [], []
    arms = s["arms"]
    for d in durs:
        on = np.array(arms[str(d)]["D_online"]["eval_return_mean"])
        ref_delay = C[:, 1:K + 1]
        ref_raw = C[:, d // E + 1:d // E + K + 1]
        L = (on[:, -5:] - ref_delay[:, -5:]).mean(1)
        A = (on[:, 1:] - ref_delay).mean(1)
        raw = (on[:, 1:] - ref_raw).mean(1)
        acute = Dm[:, d // E] - C[:, d // E]
        row = dict(d=d, L=L, A=A, raw=raw, acute=acute, at_removal=Dm[:, d // E], C_at_removal=C[:, d // E],
                   on_end=on[:, -1])
        if "D_transplant" in arms[str(d)]:
            tr = np.array(arms[str(d)]["D_transplant"]["eval_return_mean"])
            row["tr_vs_on"] = (tr[:, 1:] - on[:, 1:]).mean(1)
            row["tr_vs_C"] = (tr[:, 1:] - ref_raw).mean(1)
            row["tr_L"] = (tr[:, -5:] - ref_delay[:, -5:]).mean(1)
        rows.append(row)
        for i in range(n):
            units.append(dict(seed=seeds[i], d=d, L=L[i], A=A[i], acute=acute[i],
                              **{f"D_{k}": s["D_mask"]["probes"][k][i][d // E] for k in PROBE_KEYS},
                              **{f"dC_{k}": s["D_mask"]["probes"][k][i][d // E] - s["C"]["probes"][k][i][d // E]
                                 for k in PROBE_KEYS}))

    lines += ["## Rescue curve (own recovery and transplant rescue)", "",
              "| d (updates) | eval at removal D vs C | **L_d late delay-adj.** | A_d delay-adj. AUC | raw AUC vs C(w1+k) | "
              "transplant − D_online AUC | transplant vs C(w1+k) AUC | transplant late delay-adj. |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['d']} | {r['at_removal'].mean():.1f} vs {r['C_at_removal'].mean():.1f} | **{fmt(tci(r['L']))}** | "
                     f"{fmt(tci(r['A']))} | {fmt(tci(r['raw']))} | "
                     + (f"{fmt(tci(r['tr_vs_on']))} | {fmt(tci(r['tr_vs_C']))} | {fmt(tci(r['tr_L']))} |"
                        if "tr_vs_on" in r else "– | – | – |"))
    lines += ["", "Per-seed L_d:", ""]
    for r in rows:
        lines.append(f"- d = {r['d']}: " + ", ".join(f"{x:+.1f}" for x in r["L"]))

    if a.gate:
        lines += ["", "## Stage 0 gate (rule fixed in LOG.md before the run)", ""]
        proceed = [r["d"] for r in rows if tci(r["L"])[0] <= -5 and tci(r["L"])[2] < 0]
        means = [tci(r["L"])[0] for r in rows]
        mono = all(means[i + 1] < means[i] for i in range(len(means) - 1))
        if proceed:
            dec = f"**PROCEED** (L_d <= -5 with CI upper < 0 at d = {proceed})"
        elif mono and means[-1] <= -5:
            dec = "**EXTEND** (monotone decrease, L_max <= -5, no CI excludes 0) -> Stage 0b at d in {400, 500}"
        else:
            dec = "**STOP** (no persistent impairment beyond lost time within the budget)"
        lines += [f"Seed-mean L_d by d: " + ", ".join(f"{d}: {m:+.1f}" for d, m in zip(durs, means)),
                  f"Monotone decrease: {mono}.", "", f"Decision: {dec}"]

    # predictors at removal
    lines += ["", "## Predictors at removal: D and clean-matched difference D − C (seed means)", "",
              "| probe | " + " | ".join(f"D, d={d}" for d in durs) + " | " + " | ".join(f"D−C, d={d}" for d in durs) + " |",
              "|---|" + "---|" * (2 * len(durs))]
    for k in PROBE_KEYS:
        Dv = [np.mean([u[f"D_{k}"] for u in units if u["d"] == d]) for d in durs]
        dv = [np.mean([u[f"dC_{k}"] for u in units if u["d"] == d]) for d in durs]
        lines.append(f"| {k} | " + " | ".join(f"{x:.3f}" for x in Dv) + " | " + " | ".join(f"{x:+.3f}" for x in dv) + " |")

    # descriptive association (pilot only)
    lines += ["", f"## Descriptive association with outcomes over seed × d units (n = {len(units)}; nested, NOT inferential)", "",
              "Spearman rho. 'within-d' = rho after removing the per-duration mean from both variables "
              "(does the probe explain seed-to-seed differences at the same duration?).", "",
              "| variable | rho with L_d | rho with A_d | within-d rho with A_d |", "|---|---|---|---|"]
    uL = np.array([u["L"] for u in units])
    uA = np.array([u["A"] for u in units])
    ud = np.array([u["d"] for u in units])

    def within(x):
        x = np.asarray(x, float).copy()
        for d in durs:
            x[ud == d] -= x[ud == d].mean()
        return x

    cand = ["d", "acute"] + [f"D_{k}" for k in PROBE_KEYS] + [f"dC_{k}" for k in PROBE_KEYS]
    for c in cand:
        x = ud if c == "d" else np.array([u[c] for u in units])
        if np.std(x) == 0:
            continue
        rl = stats.spearmanr(x, uL)[0]
        ra = stats.spearmanr(x, uA)[0]
        rw = stats.spearmanr(within(x), within(uA))[0] if c != "d" else float("nan")
        lines.append(f"| {c} | {rl:+.2f} | {ra:+.2f} | {rw:+.2f} |")

    out_md = Path("results/summary") / f"{a.out}-analysis.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cu = np.array(s["C"]["updates"])
    ax[0, 0].plot(cu, C.mean(0), "k-", lw=2, label="clean C")
    ax[0, 0].plot(np.array(s["D_mask"]["updates"]), Dm.mean(0), color="0.5", lw=2, label="D during mask (unmasked eval)")
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(durs)))
    for d, col in zip(durs, cols):
        on = np.array(arms[str(d)]["D_online"]["eval_return_mean"])
        ax[0, 0].plot(arms[str(d)]["D_online"]["updates"], on.mean(0), color=col, label=f"D_online d={d}")
        if "D_transplant" in arms[str(d)]:
            tr = np.array(arms[str(d)]["D_transplant"]["eval_return_mean"])
            ax[0, 0].plot(arms[str(d)]["D_transplant"]["updates"], tr.mean(0), color=col, ls=":")
    ax[0, 0].set_xlabel("update"); ax[0, 0].set_ylabel("deficit-free eval return"); ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_title("Learning curves (dotted = actor transplant)")
    for key, lab, off in (("L", "L_d late delay-adj.", -3), ("A", "A_d delay-adj. AUC", 0), ("tr_L", "transplant late delay-adj.", 3)):
        if key not in rows[0]:
            continue
        t = [tci(r[key]) for r in rows]
        ax[0, 1].errorbar(np.array(durs) + off, [x[0] for x in t], yerr=[[x[0] - x[1] for x in t], [x[2] - x[0] for x in t]],
                          marker="o", capsize=3, label=lab)
    ax[0, 1].axhline(0, color="k", lw=0.8); ax[0, 1].axhline(-5, color="r", lw=0.8, ls="--", label="gate −5")
    ax[0, 1].set_xlabel("restriction duration d (updates)"); ax[0, 1].set_ylabel("vs pause-equivalent clean (return)")
    ax[0, 1].set_title("Rescue curve"); ax[0, 1].legend(fontsize=7)
    pu = np.array(s["D_mask"]["updates"])
    for axx, keys in ((ax[1, 0], ["decode_r2_h1", "masked_logit_corr", "value_corr"]),
                      (ax[1, 1], ["kl_onset", "repr_drift_a1", "actor_rel_change"])):
        for k, col in zip(keys, ["C0", "C1", "C2"]):
            axx.plot(pu, np.array(s["D_mask"]["probes"][k]).mean(0), color=col, lw=2, label=f"{k} D")
            axx.plot(pu, np.array(s["C"]["probes"][k]).mean(0), color=col, lw=1, ls="--", label=f"{k} C")
        axx.set_xlabel("update"); axx.legend(fontsize=7); axx.set_title("Probe trajectories during the mask (vs onset)")
    fig.tight_layout()
    fig.savefig(Path("figures") / f"{a.out}.png", dpi=120)
    print("wrote", out_md)


if __name__ == "__main__":
    main()

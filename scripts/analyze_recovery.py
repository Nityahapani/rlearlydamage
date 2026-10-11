"""Analyze recovery tests (scripts/recovery_test.py) from committed summaries only.

For each masked config, versus the matched control (same seeds, same eval protocol, same absolute update):
  Q1 during mask      : (a) deficit-free eval, mean over checkpoints strictly inside the window
                        (b) training-time return as experienced (masked), mean over the window
  Q2 immediately after: deficit-free eval at the removal checkpoint (params trained under the mask,
                        zero unmasked training)
  Q3 fixed recovery   : eval at removal + 50 and + 100 updates (0.41M / 0.82M env steps)
  Q4 later horizon    : eval at removal + 250 updates (2.05M env steps)
  Q5 control          : the same quantities for the control at the same absolute updates
Q3/Q4 use the mean of the three checkpoints centred on the horizon (u-10, u, u+10) to reduce eval noise.
Gap = control - deficit, paired by seed; CI = percentile bootstrap over seeds (crude with 4 seeds).
Secondary: delay-adjusted gap = control(u - window length) - deficit(u).

Usage: python scripts/analyze_recovery.py --summaries results/summary/<move>.json [results/summary/<fire>.json] \
           --control-from results/summary/<move>.json --tag recovery
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HORIZONS = {"Q3 +50": 50, "Q3 +100": 100, "Q4 +250": 250}


def arr(x):
    return np.array([[np.nan if v is None else v for v in row] for row in x], float)


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def fmt(d, c=None):
    lo, hi = boot(d)
    s = f"{np.nanmean(d):+.1f} [{lo:+.1f}, {hi:+.1f}]"
    if c is not None:
        s += f" ({100*np.nanmean(d)/np.nanmean(c):+.0f}%)"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--control-from", required=True)
    ap.add_argument("--tag", default="recovery")
    a = ap.parse_args()

    ctrl_s = json.loads(Path(a.control_from).read_text())
    U = np.array(ctrl_s["eval_updates"])
    E = U[1] - U[0]
    spu = ctrl_s["protocol"]["steps_per_update"]
    C = arr(ctrl_s["configs"]["control"]["eval_return_mean"])
    Ctr = arr(ctrl_s["configs"]["control"]["train_return_by_block"])
    idx = {int(u): i for i, u in enumerate(U)}

    def at(M, u, smooth=False):
        if smooth:
            ii = [idx[v] for v in (u - E, u, u + E) if v in idx]
            return np.nanmean(M[:, ii], axis=1)
        return M[:, idx[u]]

    rows, configs = [], {}
    for sp in a.summaries:
        s = json.loads(Path(sp).read_text())
        assert s["eval_updates"] == ctrl_s["eval_updates"] and s["seeds"] == ctrl_s["seeds"], "protocol mismatch"
        for name, c in s["configs"].items():
            if name == "control":
                continue
            configs[name] = c
    lines = ["| config | quantity | update | deficit | control | gap = control - deficit [95% CI] (% of control) | delay-adj gap |",
             "|---|---|---|---|---|---|---|"]
    for name, c in configs.items():
        D = arr(c["eval_return_mean"])
        Dtr = arr(c["train_return_by_block"])
        w0, w1 = c["window_updates"]
        L = w1 - w0
        inside = [idx[u] for u in U if w0 < u < w1]
        d1, c1 = D[:, inside].mean(1), C[:, inside].mean(1)
        lines.append(f"| {name} | Q1a during mask, unmasked eval | ({w0},{w1}) | {d1.mean():.1f} | {c1.mean():.1f} | {fmt(c1-d1, c1)} | - |")
        blocks = slice(w0 // E, w1 // E)  # train blocks covering (w0, w1]
        d1b, c1b = np.nanmean(Dtr[:, blocks], 1), np.nanmean(Ctr[:, blocks], 1)
        lines.append(f"| {name} | Q1b during mask, as-experienced (masked) | ({w0},{w1}) | {d1b.mean():.1f} | {c1b.mean():.1f} | {fmt(c1b-d1b, c1b)} | - |")
        d2, c2 = at(D, w1), at(C, w1)
        dd2 = at(C, w1 - L) - d2 if (w1 - L) in idx else np.full_like(d2, np.nan)
        lines.append(f"| {name} | Q2 immediately after removal | {w1} | {d2.mean():.1f} | {c2.mean():.1f} | {fmt(c2-d2, c2)} | {fmt(dd2)} |")
        for q, h in HORIZONS.items():
            u = w1 + h
            if u not in idx:
                continue
            dq, cq = at(D, u, True), at(C, u, True)
            ddq = at(C, u - L, True) - dq
            lines.append(f"| {name} | {q} after removal | {u} | {dq.mean():.1f} | {cq.mean():.1f} | {fmt(cq-dq, cq)} | {fmt(ddq)} |")
    print("\n".join(lines))

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    steps = U * spu / 1e6
    ax = axes[0]
    ax.plot(steps, C.mean(0), "k-", lw=2, label="control")
    for name, c in configs.items():
        D = arr(c["eval_return_mean"])
        ln, = ax.plot(steps, D.mean(0), lw=1.8, label=name)
        w0, w1 = c["window_updates"]
        ax.axvspan(w0 * spu / 1e6, w1 * spu / 1e6, color=ln.get_color(), alpha=0.12)
    ax.set_title("deficit-free evaluation return (mean of seeds)")
    ax.set_xlabel("env steps (M)"); ax.set_ylabel("eval return"); ax.legend(fontsize=8)
    ax = axes[1]
    for name, c in configs.items():
        D = arr(c["eval_return_mean"])
        w0, w1 = c["window_updates"]
        ii = [i for i, u in enumerate(U) if u >= w1]
        since = (U[ii] - w1) * spu / 1e6
        gap = C[:, ii] - D[:, ii]
        ln, = ax.plot(since, gap.mean(0), lw=1.8, label=name)
        ax.fill_between(since, gap.min(0), gap.max(0), color=ln.get_color(), alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("gap to control vs time since removal (band = seed range)")
    ax.set_xlabel("env steps since mask removal (M)"); ax.set_ylabel("control - deficit (eval)"); ax.legend(fontsize=8)
    ax = axes[2]
    tsteps = U[1:] * spu / 1e6
    ax.plot(tsteps, np.nanmean(Ctr, 0), "k-", lw=2, label="control")
    for name, c in configs.items():
        ax.plot(tsteps, np.nanmean(arr(c["train_return_by_block"]), 0), lw=1.5, label=name)
    ax.set_title("training-time return (as experienced, masked)")
    ax.set_xlabel("env steps (M)"); ax.set_ylabel("episode return"); ax.legend(fontsize=8)
    fig.suptitle(f"Recovery test: {ctrl_s['manifest'].split('-ppo-')[-1].rsplit('-', 1)[0]}, "
                 f"{len(ctrl_s['seeds'])} seeds, {ctrl_s['protocol']['eval_envs']} eval episodes/checkpoint")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = Path("figures") / f"{a.tag}.png"
    fig.savefig(out, dpi=130)
    print("figure:", out)


if __name__ == "__main__":
    main()

"""Analyze Phase 2 sweeps from committed summaries only (results/summary/*.json).

Damage for a window = control - deficit return, paired by seed (same seeds, same config otherwise).
Two readouts:
  final     : return over the last 10% of training (late windows get less recovery time)
  equal-rec : return a fixed time (recovery_frac * total) after deficit removal, compared with the
              control at that same step (equal recovery time for every window)

Usage:
  python scripts/analyze_sweep.py --control <summary.json with a 'control' entry> --sweeps <summary.json> ...
Writes figures/p2_<tag>.png and prints a table (markdown).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(p):
    return json.loads(Path(p).read_text())


def control_at(ctrl: dict, step: int) -> np.ndarray:
    """Control return per seed near `step`, from the downsampled curves (mean of 3 nearest points)."""
    steps = np.array(ctrl["curve_steps"])
    cur = np.array(ctrl["curves"]["control"], dtype=float)
    i = int(np.clip(np.searchsorted(steps, step), 1, len(steps) - 2))
    return np.nanmean(cur[:, i - 1 : i + 2], axis=1)


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--sweeps", nargs="+", required=True)
    ap.add_argument("--tag", default="pilot")
    a = ap.parse_args()

    cs = load(a.control)
    ctrl_final = np.array(cs["windows"]["control"]["final_return"], float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    lines = ["| deficit | window | final (mean) | damage final [95% CI] | damage equal-rec [95% CI] |",
             "|---|---|---|---|---|"]
    lines.append(f"| control | - | {ctrl_final.mean():.1f} | - | - |")
    for sp in a.sweeps:
        s = load(sp)
        dname = s["sweep"]["deficit"]
        xs, dfin, deq = [], [], []
        for lab, w in s["windows"].items():
            if lab == "control":
                continue
            fin = np.array(w["final_return"], float)
            eq = np.array([np.nan if v is None else v for v in w["equal_recovery_return"]], float)
            d_f = ctrl_final - fin
            d_e = control_at(cs, w["equal_recovery_step"]) - eq
            onset = w["bounds"][0] / s["curve_steps"][-1]
            xs.append(onset); dfin.append(d_f); deq.append(d_e)
            cf, ce = boot_ci(d_f), boot_ci(d_e)
            lines.append(f"| {dname} | {lab} | {fin.mean():.1f} | {np.nanmean(d_f):+.1f} [{cf[0]:+.1f}, {cf[1]:+.1f}] "
                         f"| {np.nanmean(d_e):+.1f} [{ce[0]:+.1f}, {ce[1]:+.1f}] |")
        for ax, d, title in ((axes[0], dfin, "damage at end of training"),
                             (axes[1], deq, "damage at equal recovery time")):
            m = [np.nanmean(v) for v in d]
            ax.plot(xs, m, "o-", label=dname)
            for x, v in zip(xs, d):
                ax.scatter([x] * len(v), v, s=8, alpha=0.4, color=ax.lines[-1].get_color())
            ax.set_title(title)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("deficit onset (fraction of training)")
        ax.set_ylabel("control - deficit return")
        ax.legend()
    fig.suptitle(f"Phase 2 {a.tag}: {cs['manifest'].split('-ppo-')[-1].split('-')[0]} "
                 f"(duration {load(a.sweeps[0])['sweep']['duration']}, {len(ctrl_final)} seeds)")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = Path("figures") / f"p2_{a.tag}.png"
    fig.savefig(out, dpi=130)
    print("\n".join(lines))
    print("figure:", out)


if __name__ == "__main__":
    main()

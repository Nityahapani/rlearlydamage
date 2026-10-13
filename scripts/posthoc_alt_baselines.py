"""POST-HOC (reviewer response, Phase 8): how much do the recovery conclusions depend on the counterfactual baseline?

For each restricted arm (late and early windows; PPO Phase 2, PQN / PQN-R Phase 5; PPO Phase 7 pilot d = 300 when
available) and each seed, compare the restricted agent's post-removal curve D(w1 + k) with three counterfactuals
built from the matched clean run C (same seed, shared history up to w0):

  B1 same-update   : C(w1 + k)        -- counts all lost training time as damage (upper bound on damage)
  B2 pause         : C(w1 + k - L)    -- the delay-adjusted baseline used in the paper (C frozen for the window)
  B3 competence-matched : C(t* + k)   -- t* = first checkpoint at which C (3-checkpoint moving average) reaches D's
                                         competence at removal. Treats unlearning as additional lost time and asks
                                         whether D learns *as fast as a clean agent starting from the same competence*.
                                         D below this baseline = slower learning from equal competence (a plasticity-
                                         or data-type deficit that cannot be expressed as lost time).

Readouts: AUC (mean over checkpoints k = 10..K) and end (mean over the last 5 checkpoints of the window).
Paired t-intervals over seeds; negative = restricted agent worse.
Usage: python scripts/posthoc_alt_baselines.py  ->  results/summary/p8_alt_baselines-posthoc.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

S = Path("results/summary")


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def fmt(x):
    m, lo, hi = tci(x)
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]{star}"


def baselines(U, C, D, w0, w1, K, E=10):
    """U: eval updates; C, D: seeds x checkpoints (D defined for u >= w1). Returns dict of per-seed arrays."""
    U = list(U)
    L = w1 - w0
    i1 = U.index(w1)
    ks = list(range(E, K + 1, E))
    out = {k: [] for k in ("B1_auc", "B2_auc", "B3_auc", "B1_end", "B2_end", "B3_end", "tstar")}
    Cs = np.array([np.convolve(np.pad(c, 1, mode="edge"), np.ones(3) / 3, mode="valid") for c in C])
    for s in range(C.shape[0]):
        d_rem = D[s, i1]
        hit = np.where(Cs[s, : i1 + 1] >= d_rem)[0]
        ts = U[hit[0]] if len(hit) else U[i1]
        out["tstar"].append(ts)
        dk = np.array([D[s, U.index(w1 + k)] for k in ks])
        b1 = np.array([C[s, U.index(w1 + k)] for k in ks])
        b2 = np.array([C[s, U.index(w1 + k - L)] for k in ks])
        b3 = np.array([C[s, U.index(ts + k)] for k in ks])
        for nm, b in (("B1", b1), ("B2", b2), ("B3", b3)):
            out[nm + "_auc"].append((dk - b).mean())
            out[nm + "_end"].append((dk[-5:] - b[-5:]).mean())
    return {k: np.array(v) for k, v in out.items()}


def main():
    rows = []
    # PPO Phase 2 recovery test (move file holds control, move_early, move_late; fire file holds fire arms)
    mv = json.loads((S / "20261011-p2-recovery-move-ppo-SpaceInvaders-MinAtar-fea755e.json").read_text())
    fi = json.loads((S / "20261011-p2-recovery-fire-ppo-SpaceInvaders-MinAtar-806d05a.json").read_text())
    U = mv["eval_updates"]
    C = np.array(mv["configs"]["control"]["eval_return_mean"])
    for src, arm in ((mv, "move_early"), (mv, "move_late"), (fi, "fire_early"), (fi, "fire_late")):
        cfg = src["configs"][arm]
        w0, w1 = cfg["window_updates"]
        D = np.array(cfg["eval_return_mean"])
        assert src["eval_updates"] == U
        rows.append(("PPO", arm, w0, w1, 250, baselines(U, C, D, w0, w1, 250), len(C)))
    # PQN / PQN-R Phase 5 (eps 0.05 eval primary)
    for var, ctl, el in (("PQN", "20261011-p5-recovery-pqn-control-SpaceInvaders-MinAtar-3bc24c1.json",
                          "20261011-p5-recovery-pqn-early-late-SpaceInvaders-MinAtar-fd9d190.json"),
                         ("PQN-R", "20261011-p5-recovery-pqnr-control-SpaceInvaders-MinAtar-3bc24c1.json",
                          "20261011-p5-recovery-pqnr-early-late-SpaceInvaders-MinAtar-8456a9b.json")):
        c = json.loads((S / ctl).read_text())
        e = json.loads((S / el).read_text())
        assert c["seeds"] == e["seeds"] and c["eval_updates"] == e["eval_updates"]
        U = c["eval_updates"]
        C = np.array(c["arms"]["control"]["eval_eps05_return_mean"])
        for arm, K in (("early", 900), ("late", 650)):
            w0, w1 = e["arms"][arm]["window_updates"]
            D = np.array(e["arms"][arm]["eval_eps05_return_mean"])
            rows.append((var, arm, w0, w1, K, baselines(U, C, D, w0, w1, K), len(C)))

    lines = ["# POST-HOC: dependence of recovery conclusions on the counterfactual baseline (Phase 8, reviewer response)", "",
             "Not pre-registered. Negative = restricted agent worse than the baseline; paired 95% t-CI over seeds; * = CI excludes 0.",
             "B1 same-update C(w1+k) (all lost time counted as damage); B2 pause C(w1+k-L) (paper's delay-adjusted baseline); "
             "B3 competence-matched C(t*+k), t* = first checkpoint where C's 3-checkpoint moving average reaches D's competence at removal "
             "(asks whether D learns as fast as a clean agent from equal competence).", "",
             "| agent | arm | n | window | K | t* (mean, range) | B1 AUC | B2 AUC | B3 AUC | B1 end | B2 end | B3 end |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for var, arm, w0, w1, K, r, n in rows:
        ts = r["tstar"]
        lines.append(f"| {var} | {arm} | {n} | [{w0}, {w1}) | {K} | {ts.mean():.0f} ({ts.min()}-{ts.max()}) | " +
                     " | ".join(fmt(r[k]) for k in ("B1_auc", "B2_auc", "B3_auc", "B1_end", "B2_end", "B3_end")) + " |")
    out = S / "p8_alt_baselines-posthoc.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

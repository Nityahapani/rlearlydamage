"""Phase 10 secondary checks for onset-reference surgery (pre-registered (a), (b)).

(a) bit-identity: the onset-reference run's DDD cell must equal the Phase 8 fork-reference DDD cell for the same seeds.
(b) state-level pause vs time-shifted pause: PPP(w1 + k) (onset parameters + Adam, environment from the clean fork at
    w1) compared with C(w1 + k - L) from the Phase 8 run (fork eval during the window for k < L, CCC arm afterwards).
    Also reports DDD - C(w1+k-L) (the delay-adjusted AUC) next to DDD - PPP.
Usage: python scripts/analyze_p10_onsetref.py --onset A.json B.json --fork P8a.json P8b.json --out NAME
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:+.2f} [{m - h:+.2f}, {m + h:+.2f}]" + ("*" if m - h > 0 or m + h < 0 else "")


def load(files):
    out = {}
    for f in files:
        s = json.loads(Path(f).read_text())
        for i, sd in enumerate(s["seeds"]):
            out[sd] = dict(arms={k: np.array(v["eval_return_mean"])[i] for k, v in s["arms"].items()},
                           fork=s.get("fork_eval"), idx=i, window=s["protocol"]["window"], E=s["protocol"]["eval_every"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onset", nargs="+", required=True)
    ap.add_argument("--fork", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    O, F = load(a.onset), load(a.fork)
    seeds = sorted(set(O) & set(F))
    w0, w1 = O[seeds[0]]["window"]
    E = O[seeds[0]]["E"]
    L = w1 - w0
    maxdiff = max(float(np.abs(O[s]["arms"]["DDD"] - F[s]["arms"]["DDD"]).max()) for s in seeds)
    pause, ppp, ddd = [], [], []
    for s in seeds:
        f = F[s]
        assert f["fork"] is not None, "fork-reference run needs --eval-fork"
        fu = f["fork"]["updates"]
        fc = np.array(f["fork"]["C"])[:, f["idx"]]
        ccc = f["arms"]["CCC"]
        K = len(ccc) - 1
        cur = []
        for k in range(0, K + 1):
            u = w1 + k * E - L
            cur.append(fc[fu.index(u)] if u < w1 else ccc[(u - w1) // E])
        pause.append(np.array(cur))
        ppp.append(O[s]["arms"]["CCC"])
        ddd.append(O[s]["arms"]["DDD"])
    pause, ppp, ddd = map(np.array, (pause, ppp, ddd))
    auc = lambda x: x[:, 1:].mean(1)
    lines = [f"# Phase 10 onset-reference checks — {a.out}", "",
             f"Seeds {seeds} (n = {len(seeds)}); window [{w0}, {w1}); AUC over +{E}..+{E * (ddd.shape[1] - 1)}.", "",
             f"- (a) DDD bit-identity vs Phase 8 fork-reference run: max |diff| = {maxdiff:.3g}",
             f"- (b) state-level pause PPP AUC {auc(ppp).mean():.1f} vs time-shifted pause C(w1+k-L) AUC {auc(pause).mean():.1f}; "
             f"PPP - pause: {tci(auc(ppp) - auc(pause))}",
             f"- DDD - PPP (beyond lost time, state-level): {tci(auc(ddd) - auc(ppp))}",
             f"- DDD - C(w1+k-L) (delay-adjusted AUC, time-shifted): {tci(auc(ddd) - auc(pause))}",
             f"- PPP at +0: {ppp[:, 0].mean():.1f}; C(w0) (should match): {pause[:, 0].mean():.1f}"]
    p = Path("results/summary") / f"{a.out}-checks.md"
    p.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Phase 8 analysis of PPO 2^3 component surgery (pre-registered contrasts, LOG 2026-10-13 Phase 8).

Pools seed groups across summary files (an arm may come from different files for different seeds), then reports for
AUC (mean eval over checkpoints +E..+R, as in Phase 3) and selected single checkpoints:
  * total damage DDD - CCC;
  * factorial main effects (actor, critic, opt; mean over D-cells minus mean over C-cells), Holm across the three;
  * TOST against +-SESOI for the critic and optimizer main effects (and actor, descriptively);
  * exact Shapley decomposition of DDD - CCC over the three carriers (sums to the total per seed by construction),
    with seed-paired t-CIs and shares = mean(phi_i) / mean(total);
  * single transplants / rescues;
  * data arms (IS and ratio-neutral) + swap diagnostics, when present;
  * with --fork-eval: unlearning at removal D(w1) - D(w0) and delay-adjusted DDD(w1+k) - C(w1+k-L).
Usage:
  python scripts/analyze_p8_surgery.py --summaries A.json B.json ... --tag NAME [--sesoi 3] [--fresh-seeds 4 5 ...]
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
from scipy import stats

CARR = ("actor", "critic", "opt")
CODES = ["".join(c) for c in __import__("itertools").product("CD", repeat=3)]


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def tp(x):
    return float(stats.ttest_1samp(x, 0.0).pvalue)


def tost(x, d):
    x = np.asarray(x, float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    p_lo = 1 - stats.t.cdf((x.mean() + d) / se, n - 1)  # H0: mu <= -d
    p_hi = stats.t.cdf((x.mean() - d) / se, n - 1)      # H0: mu >= +d
    return float(max(p_lo, p_hi))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def fmt(x, star=True):
    m, lo, hi = tci(x)
    s = "*" if star and (lo > 0 or hi < 0) else ""
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]{s}"


def shapley(v):
    """v: dict frozenset(players) -> per-seed array (value relative to CCC). Returns dict player -> per-seed phi."""
    n = len(CARR)
    phi = {}
    for i in CARR:
        others = [p for p in CARR if p != i]
        acc = 0.0
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                acc = acc + w * (v[frozenset(S) | {i}] - v[frozenset(S)])
        phi[i] = acc
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sesoi", type=float, default=3.0)
    ap.add_argument("--fresh-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--fork-eval", action="store_true")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    data, U0, meta = {}, None, {}
    fork = {"C": {}, "D": {}}
    fork_updates = None
    for f in a.summaries:
        d = json.loads(Path(f).read_text())
        U = d["eval_updates"]
        U0 = U if U0 is None else U0
        assert U == U0, f"eval grid mismatch in {f}"
        for arm, v in d["arms"].items():
            M = np.asarray(v["eval_return_mean"], float)
            for j, s in enumerate(d["seeds"]):
                if s in data.setdefault(arm, {}):
                    assert np.allclose(data[arm][s], M[j], atol=1e-3), f"seed {s} arm {arm} differs across files"
                data[arm][s] = M[j]
            if "diag_first_k" in v:
                meta.setdefault(arm, []).append((len(d["seeds"]), v["diag_first_k"], v.get("diag_next_k")))
        if d.get("fork_eval"):
            fork_updates = d["fork_eval"]["updates"]
            for j, s in enumerate(d["seeds"]):
                fork["C"][s] = np.array([r[j] for r in d["fork_eval"]["C"]])
                fork["D"][s] = np.array([r[j] for r in d["fork_eval"]["D"]])
    U = U0
    E = U[1] - U[0]
    R = U[-1] - U[0]
    seeds = sorted(set.intersection(*[set(data[c]) for c in CODES if c in data]))
    have_all = all(c in data for c in CODES)
    L = []
    L.append(f"# Phase 8 PPO component surgery — {a.tag}")
    if a.title:
        L.append("")
        L.append(a.title)
    L += ["", f"Summaries: {', '.join(Path(f).name for f in a.summaries)}",
          f"Seeds with all factorial arms (n = {len(seeds)}): {seeds}. Removal at u{U[0]}; recovery {R}; eval every {E}. "
          f"AUC = mean eval over +{E}..+{R}. Paired 95% t-CIs; * = CI excludes 0; negative = harm.", ""]

    def M(arm, ss=None):
        ss = seeds if ss is None else ss
        return np.stack([data[arm][s] for s in ss])

    def readouts(X):
        cols = {"AUC": X[:, 1:].mean(1), "+0": X[:, 0]}
        for h in (10, 50, R):
            if h % E == 0 and h // E < X.shape[1]:
                cols[f"+{h}"] = X[:, h // E]
        return cols

    keys = list(readouts(M(CODES[0])).keys()) if have_all else []
    if have_all:
        L += ["## Arm means", "", "| arm | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for c in CODES:
            r = readouts(M(c))
            L.append(f"| {c} | " + " | ".join(f"{r[k].mean():.1f}" for k in keys) + " |")
        X = {c: readouts(M(c)) for c in CODES}
        tot = {k: X["DDD"][k] - X["CCC"][k] for k in keys}
        L += ["", "## Total damage and main effects", "", "| contrast | " + " | ".join(keys) + " | AUC p | Holm p | TOST p (±%.1f) |" % a.sesoi,
              "|---|" + "---|" * (len(keys) + 3)]
        L.append("| total DDD - CCC | " + " | ".join(fmt(tot[k]) for k in keys) + f" | {tp(tot['AUC']):.4f} | | |")
        mains = {}
        for i, cname in enumerate(CARR):
            mains[cname] = {k: np.mean([X[c][k] for c in CODES if c[i] == "D"], 0) - np.mean([X[c][k] for c in CODES if c[i] == "C"], 0)
                            for k in keys}
        ps = np.array([tp(mains[c]["AUC"]) for c in CARR])
        hp = holm(ps)
        for j, cname in enumerate(CARR):
            L.append(f"| {cname} main effect | " + " | ".join(fmt(mains[cname][k]) for k in keys) +
                     f" | {ps[j]:.4f} | {hp[j]:.4f} | {tost(mains[cname]['AUC'], a.sesoi):.4f} |")
        # Shapley
        v = {}
        for c in CODES:
            S = frozenset(CARR[i] for i in range(3) if c[i] == "D")
            v[S] = X[c]["AUC"] - X["CCC"]["AUC"]
        phi = shapley(v)
        chk = float(np.abs(sum(phi.values()) - tot["AUC"]).max())
        L += ["", f"## Exact Shapley decomposition of total AUC damage (sums to DDD - CCC per seed; max |error| = {chk:.1e})", "",
              "| carrier | phi [95% CI] | share of total (ratio of means) | seeds < 0 |", "|---|---|---|---|"]
        for cname in CARR:
            L.append(f"| {cname} | {fmt(phi[cname])} | {phi[cname].mean() / tot['AUC'].mean() * 100:.0f}% | "
                     f"{int((phi[cname] < 0).sum())}/{len(seeds)} |")
        L.append(f"| total | {fmt(tot['AUC'])} | 100% | {int((tot['AUC'] < 0).sum())}/{len(seeds)} |")
        # pre-registered replication criterion (LOG Phase 8, J3/J6/J7)
        _, _, tot_hi = tci(tot["AUC"])
        _, _, act_hi = tci(mains["actor"]["AUC"])
        largest = max(CARR, key=lambda c: -phi[c].mean())
        if tot_hi >= 0:
            verdict = "**NO DAMAGE TO LOCALIZE** (total-damage CI includes 0) — not a failure to replicate"
        elif act_hi < 0 and largest == "actor":
            verdict = "**REPLICATES** (total CI < 0, actor main-effect CI < 0, actor has the largest Shapley share)"
        else:
            verdict = (f"**DOES NOT REPLICATE** (actor main-effect upper bound {act_hi:+.1f}; largest Shapley share: {largest})")
        L += ["", f"Pre-registered replication criterion: {verdict}"]
        L += ["", "## Single transplants and rescues (AUC)", "", "| contrast | AUC | +0 |", "|---|---|---|"]
        for lab, x, y in (("damaged actor only DCC - CCC", "DCC", "CCC"), ("clean actor into damaged CDD - DDD", "CDD", "DDD"),
                          ("damaged critic only CDC - CCC", "CDC", "CCC"), ("clean critic DCD - DDD", "DCD", "DDD"),
                          ("damaged opt only CCD - CCC", "CCD", "CCC"), ("clean opt DDC - DDD", "DDC", "DDD"),
                          ("clean-actor hybrid vs clean CDD - CCC", "CDD", "CCC")):
            L.append(f"| {lab} | {fmt(X[x]['AUC'] - X[y]['AUC'])} | {fmt(X[x]['+0'] - X[y]['+0'])} |")
        if a.fresh_seeds:
            fs = [s for s in a.fresh_seeds if s in seeds]
            Xf = {c: readouts(M(c, fs)) for c in CODES}
            mf = {cname: np.mean([Xf[c]["AUC"] for c in CODES if c[i] == "D"], 0) - np.mean([Xf[c]["AUC"] for c in CODES if c[i] == "C"], 0)
                  for i, cname in enumerate(CARR)}
            L += ["", f"## Replication on fresh seeds only (n = {len(fs)}: {fs}), AUC", "",
                  "| contrast | AUC |", "|---|---|", f"| total DDD - CCC | {fmt(Xf['DDD']['AUC'] - Xf['CCC']['AUC'])} |"]
            L += [f"| {c} main effect | {fmt(mf[c])} |" for c in CARR]
        # seed-level
        L += ["", "## Per-seed AUC (total, actor/critic/opt main effects, Shapley actor)", "",
              "| seed | total | actor | critic | opt | phi actor |", "|---|---|---|---|---|---|"]
        for j, s in enumerate(seeds):
            L.append(f"| {s} | {tot['AUC'][j]:+.1f} | {mains['actor']['AUC'][j]:+.1f} | {mains['critic']['AUC'][j]:+.1f} | "
                     f"{mains['opt']['AUC'][j]:+.1f} | {phi['actor'][j]:+.1f} |")
    # data arms
    darms = [x for x in ("CCC+bD", "DDD+bC", "CCC+bDr", "DDD+bCr") if x in data]
    if darms:
        L += ["", "## Data arms (behaviour swap for the first K updates)", "",
              "| contrast | n | seeds | AUC | +10 | +50 | AUC p |", "|---|---|---|---|---|---|---|"]
        rows, ps = [], []
        for lab, x, y in (("ratio-neutral clean data into damaged: DDD+bCr - DDD", "DDD+bCr", "DDD"),
                          ("ratio-neutral damaged data into clean: CCC+bDr - CCC", "CCC+bDr", "CCC"),
                          ("IS clean data into damaged: DDD+bC - DDD", "DDD+bC", "DDD"),
                          ("IS damaged data into clean: CCC+bD - CCC", "CCC+bD", "CCC"),
                          ("IS - ratio-neutral (clean data, damaged agent): DDD+bC - DDD+bCr", "DDD+bC", "DDD+bCr"),
                          ("IS - ratio-neutral (damaged data, clean agent): CCC+bD - CCC+bDr", "CCC+bD", "CCC+bDr")):
            if x not in data or y not in data:
                continue
            ss = sorted(set(data[x]) & set(data[y]))
            A_, B_ = readouts(np.stack([data[x][s] for s in ss])), readouts(np.stack([data[y][s] for s in ss]))
            dif = {k: A_[k] - B_[k] for k in A_}
            rows.append((lab, ss, dif))
            ps.append(tp(dif["AUC"]))
        hp = holm(np.array(ps[:2])) if len(ps) >= 2 else []
        for j, (lab, ss, dif) in enumerate(rows):
            extra = f"{ps[j]:.4f}" + (f" (Holm {hp[j]:.4f})" if j < 2 and len(hp) else "")
            L.append(f"| {lab} | {len(ss)} | {ss[0]}–{ss[-1]} | {fmt(dif['AUC'])} | {fmt(dif.get('+10', dif['AUC']))} | "
                     f"{fmt(dif.get('+50', dif['AUC']))} | {extra} |")
        L += ["", "Swap diagnostics (seed-weighted mean over files; first K updates = swap, next K = after):", "",
              "| arm | window | clip_frac | ratio_absdev | approx_kl | behaviour_kl |", "|---|---|---|---|---|---|"]
        for arm in ("CCC", "DDD") + tuple(darms):
            if arm not in meta:
                continue
            for wname, idx in (("first K", 1), ("next K", 2)):
                tot_n = sum(m[0] for m in meta[arm])
                vals = {k: sum(m[0] * m[idx][k] for m in meta[arm]) / tot_n for k in meta[arm][0][idx]}
                L.append(f"| {arm} | {wname} | {vals['clip_frac']:.3f} | {vals['ratio_absdev']:.3f} | {vals['approx_kl']:.4f} | {vals['behaviour_kl']:.3f} |")
    # fork-based readouts
    if a.fork_eval and fork_updates and have_all:
        w0, w1 = fork_updates[0], U[0]
        Lw = w1 - w0
        ss = [s for s in seeds if s in fork["C"]]
        Cf = np.stack([fork["C"][s] for s in ss])
        Df = np.stack([fork["D"][s] for s in ss])
        ccc, ddd = M("CCC", ss), M("DDD", ss)
        Cfull = np.concatenate([Cf, ccc], 1)  # C at u = w0, w0+E, ..., w1-E, w1, ..., w1+R
        Uc = list(fork_updates) + list(U)
        d_rem = ddd[:, 0]
        unlearn = d_rem - Df[:, 0]
        ks = list(range(E, R + 1, E))
        delay = np.stack([ddd[:, k // E] - Cfull[:, Uc.index(w1 + k - Lw)] for k in ks], 1)
        L += ["", f"## Fork readouts (n = {len(ss)})", "",
              f"- C at onset u{w0}: {Cf[:, 0].mean():.2f}; D at removal u{w1}: {d_rem.mean():.2f}; C at removal: {ccc[:, 0].mean():.2f}",
              f"- unlearning during the window D(w1) - D(w0): {fmt(unlearn)}",
              f"- raw gap at removal DDD - CCC (+0): {fmt(ddd[:, 0] - ccc[:, 0])}",
              f"- delay-adjusted AUC DDD(w1+k) - C(w1+k-{Lw}), k = {E}..{R}: {fmt(delay.mean(1))}",
              f"- delay-adjusted end (last 5 checkpoints): {fmt(delay[:, -5:].mean(1))}"]
    out = Path("results/summary") / f"{a.tag}-analysis.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

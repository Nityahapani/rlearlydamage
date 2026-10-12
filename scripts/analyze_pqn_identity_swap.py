"""Analyze Phase 6d matched-identity purge (scripts/pqn_identity_swap.py). Pre-registered in LOG.md before any real run.

PRIMARY (the only confirmatory test):
  Delta = AUC(purge_R) - AUC(purge_U), paired by seed; AUC = mean eval_eps05_return_mean over checkpoints +10..+650.
  Positive Delta = removing restriction-era transitions is better than removing matched same-epoch unrestricted ones.
  Two-sided paired t-test, alpha = 0.05 (95% CI). Smallest effect size of interest (SESOI) delta = 1.0 AUC point;
  equivalence by TOST against [-1.0, +1.0] (alpha 0.05 each side = 90% CI).
  Pre-registered decision (95% CI = CI95, 90% CI = CI90):
    A  CI95 lower > 0 and CI90 upper > +1.0      identity effect detected, not shown to be below the SESOI
       (reported additionally: whether CI95 lower > +1.0, i.e. effect shown to exceed the SESOI)
    B  CI95 lower > 0 and CI90 within (-1, +1)   detected but smaller than the SESOI (negligible)
    C  CI95 includes 0 and CI90 within (-1, +1)  no meaningful identity effect (equivalence)
    D  CI95 includes 0 and CI90 not within        inconclusive
    E  CI95 upper < 0                             removing the restricted data is worse (opposite direction)
SECONDARY (descriptive): windows early +10..150, middle +160..240, late +250..650; greedy-eval AUC; sign-flip p
(Monte Carlo, 10^6 fixed-seed draws, n > 20); exact Wilcoxon / normal approx; leave-one-seed-out range; column-parity
check (even vs odd seeds); per-seed differences; exposure (logged realised sampled R fraction; U and pool shares);
TD loss after removal.
Usage: python scripts/analyze_pqn_identity_swap.py results/summary/*p6d-idswap-s*-pqnr-1000k-*.json --tag p6d_idswap
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

SESOI = 1.0
SPU = 64 * 128


def ci(x, level):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return m, m - h, m + h, se


def signflip_mc(x, draws=1_000_000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); obs = abs(x.mean()); hits = 0
    for _ in range(draws // 100_000):
        s = rng.choice([-1.0, 1.0], size=(100_000, len(x)))
        hits += int((np.abs((s * x).mean(1)) >= obs - 1e-12).sum())
    return (hits + 1) / (draws + 1)


def decide(x):
    m, lo95, hi95, _ = ci(x, 0.95); _, lo90, hi90, _ = ci(x, 0.90)
    if hi95 < 0:
        return "E", "removing the restriction-era data is WORSE than removing matched unrestricted data"
    if lo95 > 0:
        if lo90 > -SESOI and hi90 < SESOI:
            return "B", "identity effect detected but smaller than the SESOI (negligible)"
        extra = " and shown to exceed the SESOI (CI95 lower > +1.0)" if lo95 > SESOI else ""
        return "A", "identity effect detected, not shown to be below the SESOI" + extra
    if lo90 > -SESOI and hi90 < SESOI:
        return "C", "no meaningful identity effect (equivalent within +-1.0 AUC)"
    return "D", "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--tag", default="p6d_idswap")
    a = ap.parse_args()
    S = sorted((json.loads(Path(f).read_text()) for f in a.summaries), key=lambda t: t["seeds"][0])
    seeds = [t["seeds"][0] for t in S]; n = len(seeds)
    assert len(set(seeds)) == n, "duplicate seeds"
    for t in S:
        assert t["eval_updates"] == S[0]["eval_updates"] and set(t["arms"]) == {"purge_R", "purge_U"}, "incomplete or mismatched summary"
        s = t["seeds"][0]
        assert t["protocol"]["cols_R"] == ([0, 63] if s % 2 == 0 else [64, 127]), "counterbalance violated"
        if s < 12:
            c = t["consistency_check"]
            assert c.get("forks_match_p6b", {}).get(str(s)) and c.get("pool_stats_match_p6b", {}).get(str(s)), f"seed {s}: Phase 6b consistency check failed or missing: {c}"
    U = np.array(S[0]["eval_updates"]); h = U - U[0]
    win = {"AUC": (10, 650), "early": (10, 150), "middle": (160, 240), "late": (250, 650)}
    get = lambda arm, m: np.stack([np.array(t["arms"][arm][m], float)[0] for t in S])
    ev = {k: get(k, "eval_eps05_return_mean") for k in ("purge_R", "purge_U")}
    evg = {k: get(k, "eval_return_mean") for k in ("purge_R", "purge_U")}
    sc = lambda e, w: e[:, (h >= win[w][0]) & (h <= win[w][1])].mean(1)
    d = {w: sc(ev["purge_R"], w) - sc(ev["purge_U"], w) for w in win}
    x = d["AUC"]; m, lo95, hi95, se = ci(x, 0.95); _, lo90, hi90, _ = ci(x, 0.90)
    t_ = m / se; p = float(2 * stats.t.sf(abs(t_), n - 1))
    p_lo = float(stats.t.sf((m + SESOI) / se, n - 1)); p_hi = float(stats.t.cdf((m - SESOI) / se, n - 1))
    code, text = decide(x)
    L = [f"# Phase 6d matched-identity purge — {a.tag}", "",
         f"Manifests: {', '.join(t['manifest'] for t in S)}.", f"Seeds (n = {n}): {seeds}.",
         f"Consistency (seeds < 12 vs Phase 6b): {[t['consistency_check'] for t in S if t['seeds'][0] < 12]}.",
         f"Pool collection mean return per seed: {[t['pool_stats']['mean_return'][0] for t in S]}.", "",
         "## Primary (pre-registered)", "",
         f"Delta = AUC(purge_R) - AUC(purge_U) = **{m:+.2f}**, 95% CI [{lo95:+.2f}, {hi95:+.2f}], 90% CI [{lo90:+.2f}, {hi90:+.2f}]; "
         f"t({n - 1}) = {t_:.2f}, p = {p:.4f}; TOST vs +-{SESOI}: p_lower = {p_lo:.4f}, p_upper = {p_hi:.4f} "
         f"(equivalence p = {max(p_lo, p_hi):.4f}); seeds > 0: {(x > 0).sum()}/{n}; SD {x.std(ddof=1):.2f}; dz = {m / x.std(ddof=1):.2f}.", "",
         f"**Pre-registered decision: {code} — {text}.**", "",
         "## Arm scores (mean over seeds)", "", "| arm | " + " | ".join(win) + " | greedy AUC |", "|---" * (len(win) + 2) + "|"]
    for k in ("purge_R", "purge_U"):
        L.append(f"| {k} | " + " | ".join(f"{sc(ev[k], w).mean():.2f}" for w in win) + f" | {sc(evg[k], 'AUC').mean():.2f} |")
    L += ["", "## Secondary (descriptive)", "", "| contrast | estimate [95% CI] | p | seeds > 0 |", "|---|---|---|---|"]
    for w in win:
        mm, l_, u_, s_ = ci(d[w], 0.95)
        L.append(f"| purge_R - purge_U, {w} | {mm:+.2f} [{l_:+.2f}, {u_:+.2f}] | {2 * stats.t.sf(abs(mm / s_), n - 1):.4f} | {(d[w] > 0).sum()}/{n} |")
    dg = sc(evg["purge_R"], "AUC") - sc(evg["purge_U"], "AUC"); mm, l_, u_, s_ = ci(dg, 0.95)
    L.append(f"| purge_R - purge_U, greedy-eval AUC | {mm:+.2f} [{l_:+.2f}, {u_:+.2f}] | {2 * stats.t.sf(abs(mm / s_), n - 1):.4f} | {(dg > 0).sum()}/{n} |")
    for par, nm in ((0, "even seeds (R in cols 0-63)"), (1, "odd seeds (R in cols 64-127)")):
        sel = np.array([s % 2 == par for s in seeds])
        if sel.sum() > 1:
            mm, l_, u_, _ = ci(x[sel], 0.95); L.append(f"| primary AUC, {nm} | {mm:+.2f} [{l_:+.2f}, {u_:+.2f}] | | {(x[sel] > 0).sum()}/{sel.sum()} |")
    loso = np.array([np.delete(x, i).mean() for i in range(n)])
    L += ["", f"Sign-flip p (Monte Carlo, 10^6 draws, seed 0): {signflip_mc(x):.4f}. Wilcoxon signed-rank p: {stats.wilcoxon(x).pvalue:.4f}. "
          f"Leave-one-seed-out Delta range: {loso.min():+.2f} .. {loso.max():+.2f} (most influential seed {seeds[int(np.argmax(np.abs(loso - m)))]})."]
    L += ["", "## Per-seed AUC", "", "| seed | cols_R | purge_R | purge_U | Delta |", "|---|---|---|---|---|"]
    for i, t in enumerate(S):
        L.append(f"| {seeds[i]} | {t['protocol']['cols_R']} | {sc(ev['purge_R'], 'AUC')[i]:.2f} | {sc(ev['purge_U'], 'AUC')[i]:.2f} | {x[i]:+.2f} |")
    # exposure
    shr = {k: {s_: get(k, s_) for s_ in ("restricted_share", "U_share", "replacement_share")} for k in ("purge_R", "purge_U")}
    sfw = {k: np.stack([np.array(t["arms"][k]["train"]["sample_frac_window"], float)[0] for t in S]) for k in ("purge_R", "purge_U")}
    marks = [m_ for m_ in (0, 50, 100, 140, 150, 200, 240, 250) if m_ in set(h)]
    L += ["", "## Exposure", "", "Buffer shares R / U / pool at checkpoints (mean over seeds):", "",
          "| arm | " + " | ".join(f"+{m_}" for m_ in marks) + " |", "|---" * (len(marks) + 1) + "|"]
    for k in ("purge_R", "purge_U"):
        L.append(f"| {k} | " + " | ".join(" / ".join(f"{shr[k][s_][:, list(h).index(m_)].mean():.3f}" for s_ in ("restricted_share", "U_share", "replacement_share")) for m_ in marks) + " |")
    up = np.arange(0, int(h[-1]) + 1)
    cum = lambda arr: np.interp(up, h, arr.mean(0))[1:].sum() * SPU / 1e6
    for k in ("purge_R", "purge_U"):
        rr = sfw[k].sum(1) * SPU / 1e6
        L.append(f"- {k}: realised sampled R {rr.mean():.3f}M (per-seed {rr.min():.3f}-{rr.max():.3f}); expected sampled U {cum(shr[k]['U_share']):.3f}M; "
                 f"expected sampled pool {cum(shr[k]['replacement_share']):.3f}M.")
    dpool = np.abs(shr["purge_R"]["replacement_share"] - shr["purge_U"]["replacement_share"]).max()
    L.append(f"- Max |pool-share difference| between arms over all checkpoints and seeds: {dpool:.5f}.")
    tl = {k: np.stack([np.array(t["arms"][k]["train"]["loss"], float)[0] for t in S]) for k in ("purge_R", "purge_U")}
    L.append(f"- TD loss, first 30 post-removal updates: purge_R {tl['purge_R'][:, :30].mean():.3f}, purge_U {tl['purge_U'][:, :30].mean():.3f}.")
    out = Path("results/summary") / f"{a.tag}-analysis.md"; out.write_text("\n".join(L) + "\n"); print("\n".join(L)); print("wrote", out)

    f, ax = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw=dict(width_ratios=[1.3, 0.9, 1]))
    for k, col in (("purge_R", "tab:blue"), ("purge_U", "tab:red")):
        mu = ev[k].mean(0); s_ = ev[k].std(0, ddof=1) / np.sqrt(n)
        ax[0].plot(h, mu, color=col, label=k); ax[0].fill_between(h, mu - s_, mu + s_, color=col, alpha=0.15)
    ax[0].set_xlabel(f"updates after removal (u{U[0]})"); ax[0].set_ylabel("eval_eps05_return_mean"); ax[0].legend(fontsize=8)
    ax[0].set_title("Matched-identity purge: remove R vs remove U", fontsize=10)
    for k, col in (("purge_R", "tab:blue"), ("purge_U", "tab:red")):
        ax[1].plot(h, shr[k]["restricted_share"].mean(0), color=col, lw=2.5 if k == "purge_U" else 1.2, label=f"{k}: R share")
        ax[1].plot(h, shr[k]["U_share"].mean(0), color=col, ls="--", lw=2.5 if k == "purge_R" else 1.2, label=f"{k}: U share")
        ax[1].plot(h, shr[k]["replacement_share"].mean(0), color=col, ls=":", lw=3 if k == "purge_R" else 1.4, label=f"{k}: pool share")
    ax[1].set_xlim(0, 300); ax[1].set_xlabel("updates after removal"); ax[1].set_ylabel("share of buffer"); ax[1].legend(fontsize=7)
    ax[2].scatter(x, np.zeros(n) + np.linspace(-0.2, 0.2, n), color="0.5", s=14)
    ax[2].errorbar(m, 0, xerr=[[m - lo95], [hi95 - m]], fmt="o", color="k", capsize=5, label="95% CI")
    ax[2].errorbar(m, -0.35, xerr=[[m - lo90], [hi90 - m]], fmt="s", color="tab:purple", capsize=5, label="90% CI (TOST)")
    for v in (-SESOI, SESOI):
        ax[2].axvline(v, color="tab:green", ls="--", lw=1)
    ax[2].axvline(0, color="k", lw=0.8); ax[2].set_yticks([]); ax[2].legend(fontsize=7)
    ax[2].set_xlabel(f"Delta AUC per seed (green = SESOI +-{SESOI}); decision {code}")
    f.tight_layout(); f.savefig(Path("figures") / f"{a.tag}.png", dpi=130); print("figure:", f"figures/{a.tag}.png")


if __name__ == "__main__":
    main()

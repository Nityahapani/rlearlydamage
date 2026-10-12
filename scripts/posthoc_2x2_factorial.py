"""POST-HOC (user-requested after Phase 6c results): paired 2x2 factorial model for the identity x residence design.

Not part of the pre-registration (Phase 6c pre-registered B - Cp, Bp - C and the residence main effect with Holm; the
interaction was pre-registered as secondary/descriptive). Everything here is descriptive and labelled post-hoc.

Data: Phase 6b (B_targeted, C_sham, A_retain) + Phase 6c (Bp_targeted_short, Cp_sham_long), 12 paired seeds.
Factors (effect coding +-1/2): identity I (+ = restriction-era block R removed, - = pre-onset block P removed);
residence T (+ = replacement in the newest rows / long, - = older rows / short).
  cells: B (I+, T+), Bp (I+, T-), Cp (I-, T+), C (I-, T-)
Effects (all within-seed contrasts, so the paired model is exact):
  identity main  = [(B + Bp) - (Cp + C)] / 2
  residence main = [(B + Cp) - (Bp + C)] / 2
  interaction    = (B - Bp) - (Cp - C) = (B - Cp) - (Bp - C)
Model 1 (primary post-hoc model): two-way repeated-measures ANOVA, seeds as blocks, separate error term per effect
(effect x seed); F(1, 11) = t^2 of the per-seed contrast. Model 2 (sensitivity): seed-blocked linear model with one
pooled residual (= random-intercept mixed model for this balanced, complete design), F(1, 33).
Exact sign-flip: all 2^12 = 4096 sign assignments of the per-seed contrast; statistics |mean| and |t|; leave-one-seed-out
(2^11 each); the most influential seed removed; exact Wilcoxon signed-rank as a further check.
Exposure: realised sampled restriction-era fraction per update (logged sample_frac_window x 64 x 128 samples);
replacement and pre-onset sampled exposure = exact per-update buffer share (FIFO positions; long position = A_retain's
logged restriction-era buffer share, short position = Cp_sham_long's) x 64 x 128 (uniform sampling; realised count
not logged, binomial SD of a cumulative count <= ~0.1% of it; seam rows <= 0.04% of slots).
Usage: python scripts/posthoc_2x2_factorial.py
"""
from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

METRIC = "eval_eps05_return_mean"
SPU = 64 * 128
CELLS = ("B_targeted", "Bp_targeted_short", "Cp_sham_long", "C_sham")
SHORT = {"B_targeted": "B", "Bp_targeted_short": "Bp", "Cp_sham_long": "Cp", "C_sham": "C", "A_retain": "A"}
I_CODE = np.array([0.5, 0.5, -0.5, -0.5]); T_CODE = np.array([0.5, -0.5, 0.5, -0.5])
EFFECTS = {"identity main": np.array([0.5, 0.5, -0.5, -0.5]), "residence main": np.array([0.5, -0.5, 0.5, -0.5]),
           "interaction": np.array([1.0, -1.0, -1.0, 1.0])}


def load():
    b = [json.loads(Path(f).read_text()) for f in glob.glob("results/summary/*p6b-tpurge-s*-pqnr-1000k-*.json")]
    c = [json.loads(Path(f).read_text()) for f in glob.glob("results/summary/*p6c-res2x2-s*-pqnr-1000k-*.json")]
    R = {}
    for t, arms in [(t, ("B_targeted", "C_sham", "A_retain")) for t in b] + [(t, ("Bp_targeted_short", "Cp_sham_long")) for t in c]:
        for i, s in enumerate(t["seeds"]):
            for k in arms:
                R.setdefault(s, {})[k] = dict(ev=np.array(t["arms"][k][METRIC], float)[i],
                                              sfw=np.array(t["arms"][k]["train"]["sample_frac_window"], float)[i],
                                              bfw=np.array(t["arms"][k]["train"]["buf_frac_window"], float)[i])
    for t in c:
        for k, v in t["consistency_check"].items():
            assert all(v.values()), (k, v)
    seeds = sorted(s for s in R if all(k in R[s] for k in CELLS + ("A_retain",)))
    return R, seeds, np.array(b[0]["eval_updates"])


def tstat(x):
    n = len(x); sd = x.std(ddof=1)
    return x.mean() / (sd / np.sqrt(n)) if sd > 0 else np.inf * np.sign(x.mean())


def flips(n):
    return np.array(list(itertools.product([1, -1], repeat=n)), float)


def signflip(x, F=None):
    F = flips(len(x)) if F is None else F
    m = (F * x).mean(1); sd = (F * x).std(1, ddof=1); t = m / (sd / np.sqrt(len(x)))
    return float((np.abs(m) >= abs(x.mean()) - 1e-12).mean()), float((np.abs(t) >= abs(tstat(x)) - 1e-9).mean()), m


def rm_effect(c):
    n = len(c); m = c.mean(); se = c.std(ddof=1) / np.sqrt(n); t = m / se
    h = stats.t.ppf(0.975, n - 1) * se; F = t * t
    return dict(est=m, lo=m - h, hi=m + h, F=F, p=float(2 * stats.t.sf(abs(t), n - 1)), eta=F / (F + n - 1))


def main():
    R, seeds, U = load(); n = len(seeds)
    h = U - U[0]
    win = {"AUC": (10, 650), "early": (10, 150), "middle": (160, 240), "late": (250, 650)}
    Y = {w: np.array([[R[s][k]["ev"][(h >= lo) & (h <= hi)].mean() for k in CELLS] for s in seeds]) for w, (lo, hi) in win.items()}
    YA = {w: np.array([R[s]["A_retain"]["ev"][(h >= lo) & (h <= hi)].mean() for s in seeds]) for w, (lo, hi) in win.items()}
    F = flips(n)
    L = ["# POST-HOC: paired 2x2 factorial model — identity x residence (Phase 6c)", "",
         "User-requested after the Phase 6c results. Not pre-registered except the interaction (pre-registered as a secondary, "
         "descriptive contrast). All p-values below are unadjusted and descriptive.", "",
         f"Seeds (n = {n}): {seeds}. Metric: {METRIC}. AUC = mean over +10..+650; windows early +10..150, middle +160..240, late +250..650.",
         "Coding (+-1/2): identity I+ = restriction-era block removed (B, Bp), I- = pre-onset block removed (Cp, C); "
         "residence T+ = replacement in newest rows / long (B, Cp), T- = older rows / short (Bp, C).",
         "Interaction = (B - Bp) - (Cp - C) = (B - Cp) - (Bp - C): positive = the identity effect is larger when the replacement stays long.", ""]
    # ---- cell means
    L += ["## Cell means (AUC) and marginal means", "", "| | T+ long | T- short | identity margin |", "|---|---|---|---|"]
    m = Y["AUC"].mean(0)
    L += [f"| I+ (R removed) | B {m[0]:.2f} | Bp {m[1]:.2f} | {(m[0] + m[1]) / 2:.2f} |",
          f"| I- (P removed) | Cp {m[2]:.2f} | C {m[3]:.2f} | {(m[2] + m[3]) / 2:.2f} |",
          f"| residence margin | {(m[0] + m[2]) / 2:.2f} | {(m[1] + m[3]) / 2:.2f} | grand {m.mean():.2f} |", "",
          f"Reference A_retain AUC {YA['AUC'].mean():.2f}.", ""]
    # ---- Model 1: RM ANOVA table (explicit sums of squares; check F = t^2)
    y = Y["AUC"]; gm = y.mean()
    ss_seed = 4 * ((y.mean(1) - gm) ** 2).sum()
    rows = []
    for name, w in EFFECTS.items():
        c = y @ w
        k = (w ** 2).sum()  # SS_effect = n * mean(c)^2 / sum(w^2); SS_error = sum((c - mean c)^2) / sum(w^2)
        ss_e = n * c.mean() ** 2 / k; ss_err = ((c - c.mean()) ** 2).sum() / k
        Fv = ss_e / (ss_err / (n - 1)); r = rm_effect(c)
        assert np.isclose(Fv, r["F"])
        rows.append((name, ss_e, ss_err, Fv, r))
    ss_tot = ((y - gm) ** 2).sum()
    assert np.isclose(ss_tot, ss_seed + sum(r[1] + r[2] for r in rows))
    L += ["## Model 1: two-way repeated-measures ANOVA (seeds as blocks; error = effect x seed), AUC", "",
          "| source | SS | df | MS | F | p | partial eta^2 | effect [95% CI] |", "|---|---|---|---|---|---|---|---|",
          f"| seed (blocks) | {ss_seed:.2f} | {n - 1} | {ss_seed / (n - 1):.2f} | | | | |"]
    for name, ss_e, ss_err, Fv, r in rows:
        L.append(f"| {name} | {ss_e:.2f} | 1 | {ss_e:.2f} | F(1,{n - 1}) = {Fv:.2f} | {r['p']:.4f} | {r['eta']:.2f} | "
                 f"{r['est']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] |")
        L.append(f"| {name} x seed (error) | {ss_err:.2f} | {n - 1} | {ss_err / (n - 1):.2f} | | | | |")
    L.append(f"| total | {ss_tot:.2f} | {4 * n - 1} | | | | | |")
    # ---- Model 2: pooled-error blocked model (= random-intercept LMM for balanced data)
    X = np.column_stack([np.kron(np.eye(n), np.ones((4, 1))), np.tile(I_CODE, n), np.tile(T_CODE, n), np.tile(I_CODE * T_CODE, n)])  # +-1/4 coding -> coefficient = (B - Bp) - (Cp - C)
    yy = y.reshape(-1); beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    res = yy - X @ beta; dfr = len(yy) - X.shape[1]; s2 = res @ res / dfr
    cov = s2 * np.linalg.inv(X.T @ X)
    L += ["", f"## Model 2 (sensitivity): seed-blocked linear model, pooled residual (df = {dfr}); equals the random-intercept "
          "mixed model for this balanced design", "", "| effect | estimate | SE | t | p | 95% CI |", "|---|---|---|---|---|---|"]
    for j, name in zip(range(n, n + 3), EFFECTS):
        se = np.sqrt(cov[j, j]); est = beta[j]; t = est / se; hw = stats.t.ppf(0.975, dfr) * se
        L.append(f"| {name} | {est:+.2f} | {se:.2f} | {t:.2f} | {2 * stats.t.sf(abs(t), dfr):.4f} | [{est - hw:+.2f}, {est + hw:+.2f}] |")
    errs = [r[2] / (n - 1) for r in rows]
    L.append(f"\nModel 2 assumes one error variance for all effects; the per-effect error MS in Model 1 are "
             + ", ".join(f"{nm} {e:.2f}" for nm, e in zip(EFFECTS, errs)) + f"; pooled error MS {s2:.2f}.")
    # ---- windows
    L += ["", "## Effects by window (Model 1; estimate [95% CI], p; exact sign-flip p on |mean|)", "",
          "| effect | " + " | ".join(win) + " |", "|---" * (len(win) + 1) + "|"]
    for name, w in EFFECTS.items():
        cells = []
        for wn in win:
            c = Y[wn] @ w; r = rm_effect(c); pm, _, _ = signflip(c, F)
            cells.append(f"{r['est']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}], p {r['p']:.3f}, flip {pm:.3f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    # ---- per-seed table
    C = {name: y @ w for name, w in EFFECTS.items()}
    simple = {"B - Cp": y[:, 0] - y[:, 2], "Bp - C": y[:, 1] - y[:, 3], "B - Bp": y[:, 0] - y[:, 1], "Cp - C": y[:, 2] - y[:, 3]}
    L += ["", "## Per-seed cell AUCs and effects", "",
          "| seed | B | Bp | Cp | C | (A) | identity main | residence main | interaction | B - Cp | Bp - C | B - Bp | Cp - C |",
          "|---" * 13 + "|"]
    for i, s in enumerate(seeds):
        L.append(f"| {s} | " + " | ".join(f"{v:.1f}" for v in y[i]) + f" | {YA['AUC'][i]:.1f} | "
                 + " | ".join(f"{C[nm][i]:+.2f}" for nm in EFFECTS) + " | " + " | ".join(f"{simple[k][i]:+.2f}" for k in simple) + " |")
    L.append("| mean | " + " | ".join(f"{v:.2f}" for v in y.mean(0)) + f" | {YA['AUC'].mean():.2f} | "
             + " | ".join(f"{C[nm].mean():+.2f}" for nm in EFFECTS) + " | " + " | ".join(f"{simple[k].mean():+.2f}" for k in simple) + " |")
    L.append("| SD | | | | | | " + " | ".join(f"{C[nm].std(ddof=1):.2f}" for nm in EFFECTS) + " | "
             + " | ".join(f"{simple[k].std(ddof=1):.2f}" for k in simple) + " |")
    L.append("| seeds > 0 | | | | | | " + " | ".join(f"{(C[nm] > 0).sum()}/{n}" for nm in EFFECTS) + " | "
             + " | ".join(f"{(simple[k] > 0).sum()}/{n}" for k in simple) + " |")
    # ---- exact sign-flip sensitivity
    L += ["", f"## Exact sign-flip sensitivity (AUC; all 2^{n} = {2 ** n} sign assignments; two-sided)", "",
          "| effect | t-test p | flip p, mean | flip p, t | Wilcoxon exact p | LOSO estimate range | LOSO flip p (mean) range | most influential seed | without it: est, flip p |",
          "|---|---|---|---|---|---|---|---|---|"]
    F11 = flips(n - 1); null_int = None
    for name in list(EFFECTS) + list(simple):
        c = C.get(name, simple.get(name))
        pm, pt, nullm = signflip(c, F)
        if name == "interaction":
            null_int = nullm
        wil = stats.wilcoxon(c, method="exact").pvalue if np.count_nonzero(c) == n else float("nan")
        lo_est, lo_p = [], []
        for i in range(n):
            cc = np.delete(c, i); lo_est.append(cc.mean()); lo_p.append(signflip(cc, F11)[0])
        infl = int(np.argmax(np.abs(np.array(lo_est) - c.mean())))
        L.append(f"| {name} | {rm_effect(c)['p']:.4f} | {pm:.4f} | {pt:.4f} | {wil:.4f} | {min(lo_est):+.2f} .. {max(lo_est):+.2f} | "
                 f"{min(lo_p):.4f} .. {max(lo_p):.4f} | seed {seeds[infl]} ({c[infl]:+.2f}) | {lo_est[infl]:+.2f}, {lo_p[infl]:.4f} |")
    L.append(f"\nInteraction: {int(round((np.abs(null_int) >= abs(C['interaction'].mean()) - 1e-12).sum()))} of {2 ** n} sign assignments give "
             f"|mean| >= the observed |{C['interaction'].mean():.3f}|. Smallest attainable two-sided flip p with n = {n}: {2 / 2 ** n:.5f}.")
    # ---- exposure
    long_sh = np.mean([R[s]["A_retain"]["bfw"] for s in seeds], 0)     # share of a block in the newest position (P1)
    short_sh = np.mean([R[s]["Cp_sham_long"]["bfw"] for s in seeds], 0)  # share of a block in the older position (P2)
    assert np.allclose([R[s]["A_retain"]["bfw"] for s in seeds], long_sh) and np.allclose([R[s]["Cp_sham_long"]["bfw"] for s in seeds], short_sh)
    pos = {"B_targeted": ("P", "pool"), "Bp_targeted_short": ("pool", "P"), "Cp_sham_long": ("R", "pool"),
           "C_sham": ("pool", "R"), "A_retain": ("P", "R")}
    upd = np.arange(1, 651)
    wins_u = {"+1..150": (1, 150), "+151..240": (151, 240), "+241..650": (241, 650), "+1..650": (1, 650)}

    def expo(k, what, lo, hi):
        sel = (upd >= lo) & (upd <= hi); p2, p1 = pos[k]
        sh = (short_sh if p2 == what else 0 * short_sh) + (long_sh if p1 == what else 0 * long_sh)
        return sh[sel].sum() * SPU / 1e6
    L += ["", "## Sampled exposure per cell (millions of sampled transitions; 64 x 128 = 8,192 samples per update)", "",
          "Restriction-era (R): realised, from the logged per-update sampled fraction (mean over seeds; per-seed range in brackets). "
          "Replacement (pool) and pre-onset block (P, rows written u150-249): expected = exact per-update buffer share x 8,192 "
          "(uniform sampling; realised counts not logged; cumulative binomial SD <= ~0.1%).", "",
          "| cell | layout P2 / P1 | " + " | ".join(f"R {w}" for w in wins_u) + " | " + " | ".join(f"pool {w}" for w in wins_u)
          + " | P +1..650 |", "|---" * (2 + 2 * len(wins_u) + 1) + "|"]
    for k in CELLS + ("A_retain",):
        rr = []
        for w, (lo, hi) in wins_u.items():
            per = np.array([R[s][k]["sfw"][lo - 1:hi].sum() * SPU / 1e6 for s in seeds])
            rr.append(f"{per.mean():.3f} [{per.min():.3f}-{per.max():.3f}]")
            exp_r = expo(k, "R", lo, hi)
            assert abs(per.mean() - exp_r) < 0.01, (k, w, per.mean(), exp_r)
        pp = [f"{expo(k, 'pool', lo, hi):.3f}" for lo, hi in wins_u.values()]
        L.append(f"| {SHORT[k]} | {pos[k][0]} / {pos[k][1]} | " + " | ".join(rr) + " | " + " | ".join(pp) + f" | {expo(k, 'P', 1, 650):.3f} |")
    xi = lambda k, what: expo(k, what, 1, 650)
    L += ["", "Exposure contrasts implied by the design (+1..650, M sampled):",
          f"- B vs Cp (identity, long): R {xi('B_targeted', 'R') - xi('Cp_sham_long', 'R'):+.3f}, P {xi('B_targeted', 'P') - xi('Cp_sham_long', 'P'):+.3f}, pool {xi('B_targeted', 'pool') - xi('Cp_sham_long', 'pool'):+.3f} (R and P swapped one-for-one in the older position).",
          f"- Bp vs C (identity, short): R {xi('Bp_targeted_short', 'R') - xi('C_sham', 'R'):+.3f}, P {xi('Bp_targeted_short', 'P') - xi('C_sham', 'P'):+.3f}, pool {xi('Bp_targeted_short', 'pool') - xi('C_sham', 'pool'):+.3f} (swapped in the newest position).",
          f"- B vs Bp (residence, R removed): pool {xi('B_targeted', 'pool') - xi('Bp_targeted_short', 'pool'):+.3f}, P {xi('B_targeted', 'P') - xi('Bp_targeted_short', 'P'):+.3f}; Cp vs C (residence, P removed): pool {xi('Cp_sham_long', 'pool') - xi('C_sham', 'pool'):+.3f}, R {xi('Cp_sham_long', 'R') - xi('C_sham', 'R'):+.3f}.",
          "- Hence the residence factor is also a swap of which retained block is long-resident, and the identity contrast at short "
          "residence (Bp - C) swaps 2.1x more R exposure than at long residence (B - Cp)."]
    # descriptive exposure-dose check: per-seed linear model AUC ~ seed + bR * R + bpool * pool (P is the reference)
    Ex = np.array([[xi(k, "R"), xi(k, "pool")] for k in CELLS])
    Xd = np.column_stack([np.ones(4), Ex]); lof = np.linalg.svd(Xd.T)[2][-1]; lof = lof / np.abs(lof).max()
    cl = y @ lof; rl = rm_effect(cl); pl, _, _ = signflip(cl, F)
    bet = np.array([np.linalg.lstsq(Xd, y[i], rcond=None)[0] for i in range(n)])
    L += ["", "Descriptive exposure-dose check (4 cells; per seed AUC = a + bR * R exposure + bpool * pool exposure; P is the reference "
          "data; 1 df lack of fit):",
          f"- bR = {bet[:, 1].mean():+.2f} per M sampled restriction-era transitions [{rm_effect(bet[:, 1])['lo']:+.2f}, {rm_effect(bet[:, 1])['hi']:+.2f}]; "
          f"bpool = {bet[:, 2].mean():+.2f} per M replacement [{rm_effect(bet[:, 2])['lo']:+.2f}, {rm_effect(bet[:, 2])['hi']:+.2f}] (relative to P).",
          f"- Lack-of-fit contrast (weights B, Bp, Cp, C = {np.round(lof, 2).tolist()}): {rl['est']:+.2f} [{rl['lo']:+.2f}, {rl['hi']:+.2f}], "
          f"p {rl['p']:.3f}, flip p {pl:.3f}. A linear-in-exposure model predicts Bp - C ~ 2.1x B - Cp; observed {simple['Bp - C'].mean():+.2f} vs {simple['B - Cp'].mean():+.2f}."]
    out = Path("results/summary/p6c_2x2-factorial-posthoc.md"); out.write_text("\n".join(L) + "\n"); print("\n".join(L)); print("wrote", out)

    # ---- figure
    f, ax = plt.subplots(1, 4, figsize=(21, 4.6), gridspec_kw=dict(width_ratios=[0.8, 1.1, 0.9, 1.0]))
    for i in range(n):
        ax[0].plot([0, 1], [y[i, 1], y[i, 0]], color="tab:blue", alpha=0.25, lw=0.8)
        ax[0].plot([0, 1], [y[i, 3], y[i, 2]], color="tab:orange", alpha=0.25, lw=0.8)
    for idx, col, lab in (((1, 0), "tab:blue", "R removed (Bp -> B)"), ((3, 2), "tab:orange", "P removed (C -> Cp)")):
        mu = y[:, list(idx)].mean(0); se = y[:, list(idx)].std(0, ddof=1) / np.sqrt(n)
        ax[0].errorbar([0, 1], mu, yerr=se, color=col, lw=2.2, marker="o", capsize=4, label=lab)
    ax[0].set_xticks([0, 1], ["short", "long"]); ax[0].set_xlabel("replacement residence"); ax[0].set_ylabel("AUC"); ax[0].legend(fontsize=7)
    ax[0].set_title("Interaction plot (lines = seeds)", fontsize=9)
    names = list(EFFECTS) + list(simple)
    for j, nm in enumerate(names):
        c = C.get(nm, simple.get(nm)); r = rm_effect(c)
        ax[1].scatter(c, np.full(n, j) + np.linspace(-0.18, 0.18, n), s=13, color="0.5", zorder=2)
        for i in range(n):
            ax[1].annotate(str(seeds[i]), (c[i], j + np.linspace(-0.18, 0.18, n)[i]), fontsize=5, color="0.4", xytext=(2, 0), textcoords="offset points")
        ax[1].errorbar(r["est"], j, xerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt="o", color="k", capsize=4, zorder=3)
    ax[1].set_yticks(range(len(names)), names, fontsize=8); ax[1].invert_yaxis(); ax[1].axvline(0, color="k", lw=0.8)
    ax[1].axhline(2.5, color="0.8", lw=0.8); ax[1].set_xlabel("per-seed AUC effect (labels = seed; bar = 95% t-CI)")
    ax[2].hist(null_int, bins=60, color="0.7"); ax[2].axvline(C["interaction"].mean(), color="r", lw=1.5, label=f"observed {C['interaction'].mean():+.2f}")
    ax[2].axvline(-C["interaction"].mean(), color="r", lw=1, ls="--"); ax[2].legend(fontsize=7)
    ax[2].set_xlabel(f"interaction mean under all {2 ** n} sign flips"); ax[2].set_title("Exact sign-flip null: interaction", fontsize=9)
    for k, col, lw in (("B_targeted", "tab:blue", 5), ("Bp_targeted_short", "tab:cyan", 5), ("Cp_sham_long", "tab:brown", 2), ("C_sham", "tab:orange", 2)):
        p2, p1 = pos[k]
        pool_cum = np.cumsum(((short_sh if p2 == "pool" else 0) + (long_sh if p1 == "pool" else 0)) * SPU) / 1e6
        r_cum = np.cumsum(np.mean([R[s][k]["sfw"] for s in seeds], 0) * SPU) / 1e6
        ax[3].plot(upd, pool_cum, color=col, lw=lw, alpha=0.75, label=f"{SHORT[k]} replacement")
        ax[3].plot(upd, r_cum, color=col, lw=lw * 0.6, ls=(0, (2, 1.5)), alpha=0.9, label=f"{SHORT[k]} restriction-era (realised)")
    ax[3].set_xlabel("updates after removal"); ax[3].set_ylabel("cumulative sampled transitions (M)"); ax[3].legend(fontsize=6, ncol=2)
    ax[3].set_title("Sampled exposure (wide = B/Bp, narrow = Cp/C; matched pairs coincide)", fontsize=8)
    f.tight_layout(); f.savefig("figures/p6c_2x2_factorial.png", dpi=130); print("figure: figures/p6c_2x2_factorial.png")


if __name__ == "__main__":
    main()

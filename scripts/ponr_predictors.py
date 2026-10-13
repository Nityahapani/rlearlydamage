"""Phase 7b: do clean-fork-relative measurements at removal predict recovery cost?

Implements exactly the pre-specification in LOG.md (commit aabd1ae): four predictors, fixed criteria
R0-R3, one candidate at most. Uses only saved pilot checkpoints (no training).

Predictors at removal w1 (D vs clean fork C at the same update, unmasked policies, onset bank):
  P1 kl_clean        KL(pi_C || pi_D), full action set
  P2 cka_clean       1 - linear CKA(h1_D, h1_C), actor a1 features
  P3 move_gap_rel    on the half of the bank with highest pi_C(masked actions): mean p_C(masked) - p_D(masked)
  P4 value_err_clean mean (V_D - V_C)^2 / var(V_C)
Outcomes: A_d (delay-adjusted AUC, k = 10..R) primary; A_early (k = 10..100) secondary.
Criteria: R0 split-half reliability >= 0.8; R1 pooled Spearman <= -0.5; R2 within-duration Spearman <= -0.3
and per-duration rho < 0 in >= 3 of 4 durations; R3 LOSO MSE reduction >= 10% over duration-only model.

Usage: python scripts/ponr_predictors.py results/summary/<pilot stem>.json --out p7b_predictors
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

PRED = ("kl_clean", "cka_clean", "move_gap_rel", "value_err_clean")
EXPECTED_SIGN = {k: -1 for k in PRED}


def load_params(path):
    z = np.load(path)
    out = {}
    for k in z.files:
        layer, leaf = k.split("/")
        out.setdefault(layer, {})[leaf] = z[k]
    return out


def seed_slice(pp, i):
    return {l: {k: v[i] for k, v in sub.items()} for l, sub in pp.items()}


def relu(x):
    return np.maximum(x, 0.0)


def actor(pp, x):
    h0 = relu(x @ pp["a0"]["kernel"] + pp["a0"]["bias"])
    h1 = relu(h0 @ pp["a1"]["kernel"] + pp["a1"]["bias"])
    return h1, h1 @ pp["Dense_0"]["kernel"] + pp["Dense_0"]["bias"]


def critic(pp, x):
    c0 = relu(x @ pp["c0"]["kernel"] + pp["c0"]["bias"])
    c1 = relu(c0 @ pp["c1"]["kernel"] + pp["c1"]["bias"])
    return (c1 @ pp["Dense_1"]["kernel"] + pp["Dense_1"]["bias"])[:, 0]


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.sum((X.T @ Y) ** 2)
    return hsic / (np.sqrt(np.sum((X.T @ X) ** 2)) * np.sqrt(np.sum((Y.T @ Y) ** 2)) + 1e-12)


def predictors(pD, pC, bank, masked):
    x = bank.astype(np.float64)
    hD, lD = actor(pD, x)
    hC, lC = actor(pC, x)
    lpD, lpC = log_softmax(lD), log_softmax(lC)
    pDm, pCm = np.exp(lpD)[:, masked].sum(-1), np.exp(lpC)[:, masked].sum(-1)
    rel = pCm >= np.median(pCm)
    vD, vC = critic(pD, x), critic(pC, x)
    return dict(kl_clean=float(np.mean(np.sum(np.exp(lpC) * (lpC - lpD), -1))),
                cka_clean=float(1 - linear_cka(hC, hD)),
                move_gap_rel=float(np.mean(pCm[rel] - pDm[rel])),
                value_err_clean=float(np.mean((vD - vC) ** 2) / (np.var(vC) + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = json.load(open(a.summary))
    stem = Path(a.summary).stem
    rdir = Path("results") / stem
    w0, durs, R = s["onset"], s["durations"], s["recovery"]
    E = s["protocol"]["eval_every"]
    K = R // E
    seeds = s["seeds"]
    n = len(seeds)
    allowed = s["protocol"]["allowed"]
    bank = np.load(rdir / "bank.npz")["bank"]
    n_act = load_params(rdir / f"onset_u{w0}.npz")["Dense_0"]["kernel"].shape[-1]
    masked = [i for i in range(n_act) if i not in allowed]
    C = np.array(s["C"]["eval_return_mean"])
    Dm = np.array(s["D_mask"]["eval_return_mean"])

    units = []
    for d in durs:
        w1 = w0 + d
        pC_all, pD_all = load_params(rdir / f"C_u{w1}.npz"), load_params(rdir / f"D_u{w1}.npz")
        on = np.array(s["arms"][str(d)]["D_online"]["eval_return_mean"])
        A = (on[:, 1:] - C[:, 1:K + 1]).mean(1)
        Ae = (on[:, 1:11] - C[:, 1:11]).mean(1)
        for i in range(n):
            pC, pD = seed_slice(pC_all, i), seed_slice(pD_all, i)
            b = bank[i]
            full = predictors(pD, pC, b, masked)
            h1 = predictors(pD, pC, b[: len(b) // 2], masked)
            h2 = predictors(pD, pC, b[len(b) // 2:], masked)
            units.append(dict(seed=seeds[i], d=d, A=float(A[i]), A_early=float(Ae[i]),
                              acute=float(Dm[i, d // E] - C[i, d // E]),
                              **full, **{f"{k}_h1": h1[k] for k in PRED}, **{f"{k}_h2": h2[k] for k in PRED}))

    ud = np.array([u["d"] for u in units])
    us = np.array([u["seed"] for u in units])
    col = lambda k: np.array([u[k] for u in units], float)

    def centre(x):
        x = x.copy()
        for d in durs:
            x[ud == d] -= x[ud == d].mean()
        return x

    def loso_gain(P, Y):
        p0, p1 = np.zeros_like(Y), np.zeros_like(Y)
        for sd in seeds:
            tr, te = us != sd, us == sd
            mA = {d: Y[tr & (ud == d)].mean() for d in durs}
            mP = {d: P[tr & (ud == d)].mean() for d in durs}
            cA = np.array([Y[j] - mA[ud[j]] for j in np.where(tr)[0]])
            cP = np.array([P[j] - mP[ud[j]] for j in np.where(tr)[0]])
            beta = (cP @ cA) / (cP @ cP + 1e-12)
            for j in np.where(te)[0]:
                p0[j] = mA[ud[j]]
                p1[j] = mA[ud[j]] + beta * (P[j] - mP[ud[j]])
        m0, m1 = np.mean((Y - p0) ** 2), np.mean((Y - p1) ** 2)
        return 1 - m1 / m0, beta

    def evaluate(Y):
        res = {}
        for k in PRED + ("acute",):
            P = col(k)
            rel = stats.spearmanr(col(f"{k}_h1"), col(f"{k}_h2"))[0] if k in PRED else float("nan")
            pooled = stats.spearmanr(P, Y)[0]
            within = stats.spearmanr(centre(P), centre(Y))[0]
            per_d = [stats.spearmanr(P[ud == d], Y[ud == d])[0] for d in durs]
            gain, beta = loso_gain(P, Y)
            r = dict(rel=rel, pooled=pooled, within=within, per_d=per_d, gain=gain, beta=beta)
            if k in PRED:
                sg = EXPECTED_SIGN[k]
                r["R0"] = rel >= 0.8
                r["R1"] = sg * pooled >= 0.5
                r["R2"] = (sg * within >= 0.3) and (sum(sg * x > 0 for x in per_d) >= 3)
                r["R3"] = gain >= 0.10
                r["pass"] = r["R0"] and r["R1"] and r["R2"] and r["R3"]
            res[k] = r
        return res

    res = evaluate(col("A"))
    res_e = evaluate(col("A_early"))
    acute_within = abs(res["acute"]["within"])

    L = [f"# Phase 7b — removal-time predictors of recovery cost ({a.out})", "",
         f"Pre-specification: LOG.md, commit aabd1ae (written before any of these metrics were computed). "
         f"Data: `{stem}` saved removal checkpoints, {len(units)} units (seeds {seeds} × d {durs}), movement restriction. "
         "Predictors compare D with the clean fork C at the same update on the onset bank (unmasked policies). "
         "Outcome A_d = delay-adjusted AUC over the 250-update budget (more negative = slower recovery beyond lost time).", "",
         "## Primary: criteria vs A_d", "",
         "| predictor | R0 split-half rel. (≥0.8) | R1 pooled ρ (≤−0.5) | R2 within-d ρ (≤−0.3) | per-d ρ (d=" + ",".join(map(str, durs)) +
         ") | R3 LOSO MSE gain (≥10%) | passes |", "|---|---|---|---|---|---|---|"]
    ok = lambda b: "✓" if b else "✗"
    for k in PRED:
        r = res[k]
        L.append(f"| {k} | {r['rel']:+.2f} {ok(r['R0'])} | {r['pooled']:+.2f} {ok(r['R1'])} | {r['within']:+.2f} | "
                 f"{', '.join(f'{x:+.2f}' for x in r['per_d'])} {ok(r['R2'])} | {100*r['gain']:+.0f}% {ok(r['R3'])} | "
                 f"**{'PASS' if r['pass'] else 'fail'}** |")
    r = res["acute"]
    L.append(f"| *acute deficit (baseline)* | – | {r['pooled']:+.2f} | {r['within']:+.2f} | "
             f"{', '.join(f'{x:+.2f}' for x in r['per_d'])} | {100*r['gain']:+.0f}% | – |")
    passing = [k for k in PRED if res[k]["pass"]]
    if passing:
        best = max(passing, key=lambda k: abs(res[k]["within"]))
        verdict = (f"**Candidate: `{best}`** (passes R0–R3; within-d ρ {res[best]['within']:+.2f}; "
                   f"beats acute deficit within-d: {abs(res[best]['within']) > acute_within}).")
    else:
        verdict = ("**No predictor passes R0–R3.** Pre-specified conclusion: no robust removal-time predictor among policy "
                   "divergence, representation divergence, relevant-state action change and critic error. No causal test is "
                   "pre-registered from this analysis.")
    L += ["", "## Verdict (pre-specified rule)", "", verdict, "",
          "## Secondary: same metrics vs A_early (k = 10..100; not used for selection)", "",
          "| predictor | pooled ρ | within-d ρ | per-d ρ | LOSO gain |", "|---|---|---|---|---|"]
    for k in PRED + ("acute",):
        r = res_e[k]
        L.append(f"| {k} | {r['pooled']:+.2f} | {r['within']:+.2f} | {', '.join(f'{x:+.2f}' for x in r['per_d'])} | "
                 f"{100*r['gain']:+.0f}% |")
    L += ["", "## Unit-level values", "", "| seed | d | A_d | A_early | acute | " + " | ".join(PRED) + " |",
          "|---|---|---|---|---|" + "---|" * len(PRED)]
    for u in units:
        L.append(f"| {u['seed']} | {u['d']} | {u['A']:+.1f} | {u['A_early']:+.1f} | {u['acute']:+.1f} | "
                 + " | ".join(f"{u[k]:.3f}" for k in PRED) + " |")
    L += ["", "Duration means of predictors: " + "; ".join(
        f"{k}: " + ", ".join(f"{col(k)[ud == d].mean():.3f}" for d in durs) for k in PRED)]
    md = Path("results/summary") / f"{a.out}-analysis.md"
    md.write_text("\n".join(L) + "\n")
    Path("results/summary") .joinpath(f"{a.out}-units.json").write_text(json.dumps(dict(units=units, durations=durs)))
    print("\n".join(L))

    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    cols = dict(zip(durs, plt.cm.viridis(np.linspace(0, 0.9, len(durs)))))
    Y = col("A")
    for j, k in enumerate(PRED):
        P = col(k)
        for d in durs:
            m = ud == d
            ax[0, j].scatter(P[m], Y[m], color=cols[d], label=f"d={d}", s=40)
            ax[1, j].scatter(centre(P)[m], centre(Y)[m], color=cols[d], s=40)
        ax[0, j].set_title(f"{k}\npooled ρ={res[k]['pooled']:+.2f}", fontsize=10)
        ax[1, j].set_title(f"within-duration ρ={res[k]['within']:+.2f}", fontsize=10)
        ax[0, j].set_xlabel(k); ax[1, j].set_xlabel(f"{k} (duration-centred)")
        ax[0, j].axhline(0, color="k", lw=0.6); ax[1, j].axhline(0, color="k", lw=0.6)
    ax[0, 0].set_ylabel("A_d (delay-adjusted AUC)"); ax[1, 0].set_ylabel("A_d (duration-centred)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Phase 7b: removal-time predictors of recovery cost (16 units; pre-specified, aabd1ae)")
    fig.tight_layout()
    fig.savefig(Path("figures") / f"{a.out}.png", dpi=120)
    print("wrote", md)


if __name__ == "__main__":
    main()

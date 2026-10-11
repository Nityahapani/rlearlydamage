"""Analyze Phase 4 causal ablations (scripts/phase4_ablations.py) from the committed summary.

Readouts per arm and seed (deficit-free eval, identical protocol for every arm):
  AUC  : mean eval over all post-removal checkpoints +E..+R (primary readout, as in Phase 3)
  +0   : eval at removal;  +10, +20 : first checkpoints (immediate effect of a K = 10 update intervention)
  +50  : mean of the 3 checkpoints centred on +50;  +R : mean of the last 2 checkpoints
Effects are paired by seed; CI = paired t-interval over seeds (n < 10).
Usage: python scripts/analyze_phase4.py results/summary/<p4 summary>.json --tag phase4_move_late
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

MASKED = (1, 2)  # SpaceInvaders-MinAtar actions: 0 noop, 1 left, 2 right, 3 fire; the mask removes 1 and 2
# Phase 2 recovery-test control (fresh agent, no deficit): eval at update 50 and 150 (from-scratch reference)
FROM_SCRATCH = {50: 28.2, 150: 58.7}


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.array([x.mean(), x.mean()])
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return np.array([x.mean() - h, x.mean() + h])


def fmt(x, p=1):
    lo, hi = ci(x)
    return f"{np.mean(x):+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


def sig(x):
    lo, hi = ci(x)
    return "*" if (lo > 0 or hi < 0) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--tag", default="phase4")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    U = np.array(s["eval_updates"])
    FU = np.array(s["fork_updates"])
    w0, w1 = FU[0], FU[-1]
    E = int(U[1] - U[0])
    R = int(U[-1] - w1)
    K = s["protocol"]["behaviour_k"]
    idx = {int(u - w1): i for i, u in enumerate(U)}
    arms = s["arms"]
    forks = s["forks"]
    ev = {k: np.array(v["eval_return_mean"], float) for k, v in arms.items()}

    def readouts(M):
        return {"AUC": M[:, 1:].mean(1), "+0": M[:, 0], "+10": M[:, idx[10]], "+20": M[:, idx[20]],
                "+50": M[:, [idx[50 - E], idx[50], idx[50 + E]]].mean(1), f"+{R}": M[:, [idx[R - E], idx[R]]].mean(1)}

    sc = {k: readouts(v) for k, v in ev.items()}
    keys = list(sc["D_online"].keys())
    L = [f"# Phase 4 causal ablations — {s['manifest']}", "",
         f"Seeds {s['seeds']}; mask window updates [{w0}, {w1}) (movement removed, allowed {s['protocol']['allowed']}); "
         f"recovery {R} updates; eval every {E}; K = {K} behaviour updates; anchor beta = {s['protocol']['anchor_coef']}. "
         "All arms evaluated deficit-free with common random numbers. CIs: paired t over seeds.", ""]
    if s.get("bit_identity_check"):
        L += [f"Bit-identity of the C/D forks vs the Phase 3 saved states: {s['bit_identity_check']}", ""]
    L += ["## Arm scores (mean over seeds)", "", "| arm | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for k in sc:
        L.append(f"| {k} | " + " | ".join(f"{sc[k][r].mean():.1f}" for r in keys) + " |")

    def table(title, pairs):
        out = ["", f"## {title}", "", "| contrast | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for name, x, y in pairs:
            if x in sc and y in sc:
                out.append(f"| {name} | " + " | ".join(fmt(sc[x][r] - sc[y][r]) + sig(sc[x][r] - sc[y][r])
                                                      for r in keys) + " |")
        return out

    # ---------------- ablation 1
    L += table("Ablation 1 — actor reset vs transplant (paired differences, * = CI excludes 0)", [
        ("total damage: D_online - C_cont", "D_online", "C_cont"),
        ("actor reset - D_online", "D_actor_reset", "D_online"),
        ("actor transplant - D_online", "D_actor_transplant", "D_online"),
        ("actor+opt reset - D_online", "D_actor_opt_reset", "D_online"),
        ("actor+opt reset - actor reset", "D_actor_opt_reset", "D_actor_reset"),
        ("transplant - actor reset", "D_actor_transplant", "D_actor_reset"),
        ("actor reset - C_cont", "D_actor_reset", "C_cont"),
        ("transplant - C_cont", "D_actor_transplant", "C_cont"),
    ])
    if "D_actor_reset" in ev:
        M = ev["D_actor_reset"]
        L += ["", "From-scratch reference (Phase 2 recovery-test control, fresh agent): eval "
              + ", ".join(f"{v} at update {u}" for u, v in FROM_SCRATCH.items())
              + f". Actor-reset arm at +50: {M[:, idx[50]].mean():.1f}, at +{R}: {M[:, idx[R]].mean():.1f} "
              "(a fresh actor that keeps the critic + optimizer, trained from the clean env state)."]

    # ---------------- ablation 2
    L += table("Ablation 2 — breaking the actor–experience loop (first K updates from a behaviour policy)", [
        ("clean behaviour - D_online", "D_beh_clean", "D_online"),
        ("frozen self (fixed pre-intervention data) - D_online", "D_beh_frozen_self", "D_online"),
        ("x-seed clean behaviour - D_online", "D_beh_xseed_clean", "D_online"),
        ("tempered self (KL-matched) - D_online", "D_beh_tempered", "D_online"),
        ("clean - x-seed clean (matchedness)", "D_beh_clean", "D_beh_xseed_clean"),
        ("clean - tempered (quality at matched KL)", "D_beh_clean", "D_beh_tempered"),
        ("x-seed clean - tempered", "D_beh_xseed_clean", "D_beh_tempered"),
        ("clean behaviour - actor transplant", "D_beh_clean", "D_actor_transplant"),
    ])
    T = s["tempered"]
    L += ["", f"Tempered behaviour: T per seed {np.round(T['T'], 2).tolist()}; KL target (KL(pi_C||pi_D) on bank) "
          f"{np.round(T['kl_target_clean_vs_D'], 3).tolist()}, achieved {np.round(T['kl_achieved'], 3).tolist()}.", ""]
    beh = [k for k in ("D_online", "D_beh_clean", "D_beh_frozen_self", "D_beh_xseed_clean", "D_beh_tempered") if k in arms]
    tm = ("episode_return", "behaviour_kl", "ratio_mean", "ratio_absdev", "clip_frac", "approx_kl", "entropy")
    L += [f"### Update diagnostics, mean over the K = {K} intervention updates (and the next K, after the swap ends)", "",
          "Training rollouts are collected by the behaviour policy (masked logits never apply here: no deficit). "
          "episode_return = return of completed training episodes (i.e. behaviour-policy return during the swap); "
          "behaviour_kl = KL(pi_b || pi_learner) on visited states; ratio = pi_learner/pi_b (stored log-prob).", "",
          "| arm | window | " + " | ".join(tm) + " |", "|---" * (len(tm) + 2) + "|"]
    for k in beh:
        tr = {m: np.array(arms[k]["train"][m], float) for m in tm}
        for lab, sl in ((f"1..{K}", slice(0, K)), (f"{K + 1}..{2 * K}", slice(K, 2 * K))):
            L.append(f"| {k} | {lab} | " + " | ".join(f"{np.nanmean(tr[m][:, sl]):.3f}" for m in tm) + " |")

    # ---------------- ablation 3
    fd = {k: {m: np.array(v, float) for m, v in f["drift"].items()} for k, f in forks.items()}
    fe = {k: np.array(f["eval_return_mean"], float) for k, f in forks.items()}
    dm = ("kl_from_onset", "entropy", "value", "repr_drift_a0", "repr_drift_a1", "dead_a1")
    L += ["", "## Ablation 3 — policy drift during the mask (fixed onset bank)", "",
          f"State at removal (update {w1}); KL = KL(pi_onset || pi_t) on the bank; p(masked) = mean prob of left+right; "
          "repr drift = 1 - linear CKA vs onset; dead_a1 = fraction of a1 units never active on the bank.", "",
          "| fork | eval | p(masked) | " + " | ".join(dm) + " |", "|---" * (len(dm) + 3) + "|"]
    pm = {k: fd[k]["probs"][:, :, list(MASKED)].sum(-1) for k in fd}
    L.append(f"| onset (u{w0}) | {fe['D'][:, 0].mean():.1f} | {pm['D'][:, 0].mean():.3f} | "
             + " | ".join(f"{fd['D'][m][:, 0].mean():.3f}" for m in dm) + " |")
    for k in fd:
        L.append(f"| {k} | {fe[k][:, -1].mean():.1f} | {pm[k][:, -1].mean():.3f} | "
                 + " | ".join(f"{fd[k][m][:, -1].mean():.3f}" for m in dm) + " |")
    L += ["", "Manipulation check and outcome at removal (paired differences):", ""]
    for k in ("F", "A", "C"):
        if k in fd:
            L.append(f"- {k} - D: KL_onset {fmt(fd[k]['kl_from_onset'][:, -1] - fd['D']['kl_from_onset'][:, -1], 3)}, "
                     f"p(masked) {fmt(pm[k][:, -1] - pm['D'][:, -1], 3)}, "
                     f"repr drift a1 {fmt(fd[k]['repr_drift_a1'][:, -1] - fd['D']['repr_drift_a1'][:, -1], 3)}, "
                     f"eval at removal {fmt(fe[k][:, -1] - fe['D'][:, -1])}")
    L += table("Ablation 3 — recovery after drift interventions", [
        ("F_cont (frozen a0/a1 during mask) - D_online", "F_cont", "D_online"),
        ("A_cont (anchored during mask) - D_online", "A_cont", "D_online"),
        ("F_cont - C_cont", "F_cont", "C_cont"),
        ("A_cont - C_cont", "A_cont", "C_cont"),
    ])
    # does drift predict collapse? across forks x seeds at removal
    xs = np.concatenate([fd[k]["kl_from_onset"][:, -1] for k in "DFA" if k in fd])
    ys = np.concatenate([fe[k][:, -1] for k in "DFA" if k in fd])
    xp = np.concatenate([pm[k][:, -1] for k in "DFA" if k in fd])
    r1, p1 = stats.spearmanr(xs, ys)
    r2, p2 = stats.spearmanr(xp, ys)
    L += ["", f"Across masked forks (D, F, A) x seeds (n = {len(ys)}): Spearman(eval at removal, KL_onset) = {r1:+.2f} "
          f"(p = {p1:.3f}); Spearman(eval at removal, p(masked)) = {r2:+.2f} (p = {p2:.3f}). Descriptive only."]
    ad = {k: {m: np.array(v, float) for m, v in arms[k]["drift"].items()} for k in arms}
    L += ["", "KL_onset along recovery (mean over seeds) at +0 / +10 / +50 / +R:", ""]
    for k in ad:
        x = ad[k]["kl_from_onset"]
        L.append(f"- {k}: " + " / ".join(f"{x[:, idx[h]].mean():.3f}" for h in (0, 10, 50, R)))

    out_md = Path("results/summary") / f"{Path(a.summary).stem}-analysis.md"
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", out_md)

    # ---------------- figures
    fig_dir = Path("figures")
    x = U - w1
    col = {"C_cont": "k", "D_online": "tab:red", "D_actor_reset": "tab:orange", "D_actor_transplant": "tab:green",
           "D_actor_opt_reset": "tab:brown", "D_beh_clean": "tab:green", "D_beh_frozen_self": "tab:purple",
           "D_beh_xseed_clean": "tab:cyan", "D_beh_tempered": "tab:olive", "F_cont": "tab:blue", "A_cont": "tab:pink"}

    def curve(ax, k, ls="-"):
        M = ev[k]
        m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        ax.plot(x, m, ls, color=col[k], label=k, lw=1.6)
        ax.fill_between(x, m - se, m + se, color=col[k], alpha=0.15)

    def bars(ax, pairs, r="AUC"):
        names, mus, los, his = [], [], [], []
        for nm, p, q in pairs:
            if p in sc and q in sc:
                d = sc[p][r] - sc[q][r]
                lo, hi = ci(d)
                names.append(nm); mus.append(d.mean()); los.append(d.mean() - lo); his.append(hi - d.mean())
        ax.barh(range(len(names)), mus, xerr=[los, his], color="0.6", capsize=3)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.invert_yaxis()
        ax.set_xlabel(f"paired {r} difference (95% t-CI)")

    # fig 1: reset vs transplant
    f, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for k in ("C_cont", "D_online", "D_actor_reset", "D_actor_opt_reset", "D_actor_transplant"):
        if k in ev:
            curve(a1, k)
    a1.set_xlabel(f"updates after mask removal (u{w1})"); a1.set_ylabel("deficit-free eval return")
    a1.set_title("Ablation 1: actor reset vs transplant"); a1.legend(fontsize=8)
    bars(a2, [("reset − D", "D_actor_reset", "D_online"), ("actor+opt reset − D", "D_actor_opt_reset", "D_online"),
              ("transplant − D", "D_actor_transplant", "D_online"), ("transplant − reset", "D_actor_transplant", "D_actor_reset"),
              ("D − C (total damage)", "D_online", "C_cont")])
    f.tight_layout(); f.savefig(fig_dir / f"{a.tag}_reset_transplant.png", dpi=130); plt.close(f)

    # fig 2: behaviour arms
    f, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    for k in beh + (["C_cont"] if "C_cont" in ev else []):
        curve(ax[0], k)
    ax[0].axvspan(0, K, color="0.9"); ax[0].set_xlim(0, 60)
    ax[0].set_title(f"Ablation 2: behaviour swap for K={K} updates (shaded)"); ax[0].legend(fontsize=7)
    ax[0].set_xlabel("updates after removal")
    for j, m in ((1, "behaviour_kl"), (2, "clip_frac")):
        for k in beh:
            tr = np.array(arms[k]["train"][m], float)[:, :3 * K]
            ax[j].plot(np.arange(1, tr.shape[1] + 1), tr.mean(0), color=col[k], label=k)
        ax[j].axvline(K + 0.5, color="0.5", ls=":"); ax[j].set_title(m); ax[j].set_xlabel("update after removal")
    bars(ax[3], [("clean − D", "D_beh_clean", "D_online"), ("frozen self − D", "D_beh_frozen_self", "D_online"),
                 ("x-seed clean − D", "D_beh_xseed_clean", "D_online"), ("tempered − D", "D_beh_tempered", "D_online"),
                 ("clean − x-seed", "D_beh_clean", "D_beh_xseed_clean"), ("clean − tempered", "D_beh_clean", "D_beh_tempered")])
    f.tight_layout(); f.savefig(fig_dir / f"{a.tag}_behaviour.png", dpi=130); plt.close(f)

    # fig 3: drift
    fc = {"C": "k", "D": "tab:red", "F": "tab:blue", "A": "tab:pink"}
    f, ax = plt.subplots(1, 5, figsize=(23, 4.2))
    for k in fd:
        for j, (y, t) in enumerate(((fe[k], "eval (deficit-free)"), (fd[k]["kl_from_onset"], "KL(onset || t) on bank"),
                                    (pm[k], "p(left)+p(right) on bank"), (fd[k]["repr_drift_a1"], "1 − CKA(a1) vs onset"))):
            ax[j].plot(FU, y.mean(0), "o-", color=fc[k], label=k)
            ax[j].set_title(t); ax[j].set_xlabel("update (mask window)")
    ax[0].legend()
    for k in ("C_cont", "D_online", "F_cont", "A_cont"):
        if k in ev:
            curve(ax[4], k)
    ax[4].set_title("recovery after drift interventions"); ax[4].set_xlabel("updates after removal"); ax[4].legend(fontsize=8)
    f.tight_layout(); f.savefig(fig_dir / f"{a.tag}_drift.png", dpi=130); plt.close(f)
    print("figures:", [f"figures/{a.tag}_{n}.png" for n in ("reset_transplant", "behaviour", "drift")])


if __name__ == "__main__":
    main()

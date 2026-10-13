"""Cross-algorithm causal summary figure (figures/causal_summary.png).

Every plotted number is copied from a committed analysis file in results/summary/.
The script refuses to plot unless the exact "est [lo, hi]" string appears verbatim
in the cited source file, so the figure cannot drift from the analyses.

All values are paired-seed differences in post-removal return AUC (mean eval
return over the post-deficit window), 95% paired-t CI. AUC units are NOT comparable
across panels (PPO and PQN-R reach different return scales); compare signs, CI
positions and within-panel magnitudes only.

Usage: python scripts/make_causal_summary.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.environ.get("RLED_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = os.path.join(ROOT, "results", "summary")

P3 = "20261011-p3-surgery-move-late-analysis.md"
P4M = "20261011-p4-phase4-move-late-ppo-SpaceInvaders-MinAtar-9a789ef-analysis.md"
P4F = "20261011-p4-phase4-fire-late-ppo-SpaceInvaders-MinAtar-8456a9b-analysis.md"
Q100 = "p6_surgery_pqnr_n12-analysis.md"
Q1M = "p6_surgery_pqnr1m_n8-analysis.md"
PUR = "p6_purge-analysis.md"
PURH = "p6_purge-posthoc.md"
P6B = "p6b_targeted-analysis.md"
P6C = "p6c_2x2-analysis.md"
P6D = "p6d_idswap-analysis.md"
P8 = "p8_ppo_surgery_n12-analysis.md"
P8R = "p8_rflip_surgery-analysis.md"
P8H = "p8_h256_surgery-analysis.md"
Q8 = "p8_pqnr_split-analysis.md"

# (label, est, lo, hi, source, decimals, status)
# status: "pre" = pre-registered / primary design contrast, "post" = post-hoc
PANEL_A = [
    ("Actor main effect\nmove, n=12", -13.7, -14.9, -12.5, P8, 1, "pre"),
    ("Critic main effect\nmove, n=12", -1.9, -3.5, -0.4, P8, 1, "pre"),
    ("Optimizer-state main effect\nmove, n=12 (TOST \u00b13 p=0.0001)", +0.2, -0.9, +1.3, P8, 1, "pre"),
    ("Clean-actor transplant into damaged agent\nmove, n=12", +13.3, +11.0, +15.6, P8, 1, "pre"),
    ("Clean behaviour data, importance-weighted\nmove, n=12", +8.3, +6.1, +10.5, P8, 1, "pre"),
    ("Clean behaviour data, ratio-neutral\nmove, n=12", -5.4, -7.0, -3.7, P8, 1, "pre"),
    ("Actor main effect\nreward flip, n=8", -32.8, -36.9, -28.7, P8R, 1, "pre"),
    ("Actor main effect\nmove, 2x256 network, n=8", -15.7, -17.6, -13.8, P8H, 1, "pre"),
    ("Actor reset to fresh init\nmove, n=4", -27.2, -39.4, -15.1, P4M, 1, "pre"),
]
PANEL_B = [
    ("Network main effect\n100k buffer, n=12", -5.6, -6.3, -4.9, Q100, 1, "pre"),
    ("Buffer main effect\n100k buffer, n=12", -1.0, -2.0, -0.1, Q100, 1, "pre"),
    ("Clean parameters, damaged opt + buffer\n(rescue) 100k, n=8", +6.4, +4.8, +8.0, Q8, 1, "pre"),
    ("Clean optimizer state, damaged params + buffer\n(rescue) 100k, n=8", +1.3, -0.4, +3.1, Q8, 1, "pre"),
    ("Network main effect\n1M buffer, n=8", -3.2, -4.2, -2.2, Q1M, 1, "pre"),
    ("Buffer main effect\n1M buffer, n=8", -2.4, -3.4, -1.4, Q1M, 1, "pre"),
]
PANEL_C = [
    ("Purge restriction-era data vs retain\n(B-A), n=8", +4.6, +2.2, +6.9, PUR, 1, "pre"),
    ("Targeted purge vs age-matched sham\n(6b B-C), n=12  [identity + residence]", +2.6, +0.9, +4.3, P6B, 1, "pre"),
    ("Residence main effect (6c), n=12\n[unadjusted CI; Holm p=0.0726]", +1.5, +0.2, +2.7, P6C, 1, "pre"),
    ("Identity at matched long residence\n(6c B-Cp), n=12", +1.1, -0.2, +2.5, P6C, 1, "pre"),
    ("Identity, exposure fully matched (6d primary)\nn=32; TOST p=0.0495 (equivalence, marginal)", +0.46, -0.19, +1.11, P6D, 2, "pre"),
]
SESOI = 1.0


def fmt(x, d):
    return f"{x:+.{d}f}"


def verify(rows):
    for label, est, lo, hi, src, d, _ in rows:
        needle = f"{fmt(est, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"
        with open(os.path.join(S, src)) as f:
            text = f.read()
        if needle not in text:
            sys.exit(f"VERIFY FAILED: '{needle}' ({label!r}) not found in {src}")
    return True


def forest(ax, rows, title, color, xlim=None, sesoi=False):
    n = len(rows)
    ys = list(range(n))[::-1]
    for y, (label, est, lo, hi, src, d, status) in zip(ys, rows):
        sig = lo > 0 or hi < 0
        face = color if sig else "white"
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="butt")
        ax.plot(est, y, marker="o" if status == "pre" else "D", ms=8,
                mfc=face, mec=color, mew=1.8, zorder=3)
        ax.text(1.02, y, f"{fmt(est, d)} [{fmt(lo, d)}, {fmt(hi, d)}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5,
                family="monospace")
    ax.axvline(0, color="k", lw=0.8)
    if sesoi:
        ax.axvspan(-SESOI, SESOI, color="0.85", zorder=0)
        ax.text(SESOI + 0.08, n - 0.45, "SESOI ±1.0", ha="left", fontsize=8, color="0.3")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.7, n - 0.2)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("paired difference in post-removal return AUC (95% CI)", fontsize=8.5)
    ax.grid(axis="x", color="0.9")


def main():
    for rows in (PANEL_A, PANEL_B, PANEL_C):
        verify(rows)
    fig, axes = plt.subplots(3, 1, figsize=(10, 13.2),
                             gridspec_kw={"height_ratios": [9, 6, 5]})
    forest(axes[0], PANEL_A,
           "A  PPO state surgery and behaviour swap — SpaceInvaders, late window",
           "#1f5fa8", xlim=(-42, 20))
    forest(axes[1], PANEL_B,
           "B  PQN-R parameters / optimizer / buffer surgery — late restriction",
           "#b5452a", xlim=(-7, 9))
    forest(axes[2], PANEL_C,
           "C  Replay composition (secondary, qualified) — PQN-R, 1M buffer",
           "#5b7f2b", xlim=(-1.5, 7.5), sesoi=True)
    fig.text(0.01, 0.005,
             "Filled marker = 95% CI excludes 0; circle = pre-registered/design contrast; "
             "diamond = post-hoc.\nMain effects: negative = carrier holds harm; transplants/rescues: positive = clean component helps; positive in C = "
             "removing/altering the data helps.\nAUC scales differ between PPO and PQN-R; "
             "compare within panels. Every value is asserted verbatim against its "
             "results/summary/*.md source by this script.",
             fontsize=7.8, color="0.25")
    fig.tight_layout(rect=(0, 0.05, 0.88, 1))
    out = os.environ.get("RLED_FIG_OUT", os.path.join(ROOT, "figures", "causal_summary.png"))
    fig.savefig(out, dpi=150)
    print("verified", len(PANEL_A) + len(PANEL_B) + len(PANEL_C), "numbers; wrote", out)


if __name__ == "__main__":
    main()

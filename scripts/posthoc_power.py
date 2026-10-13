"""POST-HOC (Phase 8 reviewer response, weakness 2): minimum detectable effects of the small-n null results.

For each null contrast reported in the paper (est, 95% paired-t CI, n), back out the SE from the CI half-width and
report the minimum detectable effect at 80% power (two-sided alpha 0.05, paired t: (t_.975 + t_.80) * SE) and the
largest harm the CI excludes (its lower bound). Values are copied from the cited analysis files.
"""
from pathlib import Path

from scipy import stats

ROWS = [  # label, est, lo, hi, n, source
    ("PPO move late, delay-adj. at +250", 2.0, -0.1, 4.1, 4, "p2_recovery_all-analysis.md"),
    ("PPO fire late, delay-adj. at +250", 2.4, -3.6, 8.3, 4, "p2_recovery_all-analysis.md"),
    ("PQN late, delay-adj. AUC", -1.4, -3.5, 0.7, 4, "p5_recovery_move-analysis.md"),
    ("PQN-R late, delay-adj. at +650", -0.2, -7.7, 7.3, 4, "p5_recovery_move-analysis.md"),
    ("PPO drift suppression F - D, AUC", 1.1, -3.4, 5.7, 4, "20261011-p4-phase4-move-late-...-analysis.md"),
    ("PPO drift suppression A - D, AUC", -0.0, -1.8, 1.7, 4, "20261011-p4-phase4-move-late-...-analysis.md"),
    ("PPO critic main effect, AUC (Phase 3)", -2.3, -6.2, 1.7, 4, "20261011-p3-surgery-move-late-analysis.md"),
    ("PPO optimizer main effect, AUC (Phase 3)", -0.1, -2.8, 2.6, 4, "20261011-p3-surgery-move-late-analysis.md"),
    ("PONR pilot L_300", -1.4, -8.2, 5.5, 4, "p7_ponr_pilot-analysis.md"),
]
L = ["# POST-HOC: minimum detectable effects of small-n null results", "",
     "MDE = (t_.975 + t_.80) * SE with SE from the 95% CI half-width; 'harm excluded beyond' = CI lower bound.", "",
     "| contrast | n | estimate [95% CI] | SE | MDE (80% power) | harm excluded beyond | source |", "|---|---|---|---|---|---|---|"]
for lab, e, lo, hi, n, src in ROWS:
    se = (hi - lo) / 2 / stats.t.ppf(0.975, n - 1)
    mde = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) * se
    L.append(f"| {lab} | {n} | {e:+.1f} [{lo:+.1f}, {hi:+.1f}] | {se:.2f} | {mde:.1f} | {lo:+.1f} | {src} |")
Path("results/summary/p8_power-posthoc.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

"""POST-HOC (Phase 8 reviewer response, weakness 7): sensitivity of the Phase 6d equivalence call to the SESOI.

Reads the per-seed Delta column of results/summary/p6d_idswap-analysis.md (n = 32) and reports the TOST p-value as a
function of the equivalence bound delta, and the smallest bound at which equivalence holds at alpha = 0.05 / 0.01
(= the upper limit of the 90% / 98% CI). The pre-registered decision (SESOI 1.0) is unchanged; this only states how
fragile it is.
"""
import re
from pathlib import Path

import numpy as np
from scipy import stats

txt = Path("results/summary/p6d_idswap-analysis.md").read_text().split("## Per-seed AUC")[1]
d = np.array([float(m.group(1)) for m in re.finditer(r"^\| \d+ \| \[.*?\] \| [\d.]+ \| [\d.]+ \| ([+-][\d.]+) \|", txt, re.M)])
assert len(d) == 32, len(d)
n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
L = ["# POST-HOC: Phase 6d equivalence — sensitivity to the SESOI", "",
     f"n = {n}; Delta = {m:+.3f}; SE {se:.3f}; 95% CI [{m - stats.t.ppf(.975, n - 1) * se:+.2f}, {m + stats.t.ppf(.975, n - 1) * se:+.2f}]", "",
     "| bound delta | TOST p | equivalent at 0.05? |", "|---|---|---|"]
for delta in (0.75, 1.0, 1.1, 1.25, 1.5, 2.0):
    p = max(1 - stats.t.cdf((m + delta) / se, n - 1), stats.t.cdf((m - delta) / se, n - 1))
    L.append(f"| ±{delta:.2f} | {p:.4f} | {'yes' if p < 0.05 else 'no'} |")
for a in (0.05, 0.01):
    L.append(f"\nSmallest symmetric bound giving equivalence at alpha = {a}: ±{max(abs(m - stats.t.ppf(1 - a, n - 1) * se), m + stats.t.ppf(1 - a, n - 1) * se):.2f} AUC")
L.append(f"\nMinimum detectable effect (two-sided alpha 0.05, power 0.8, paired t, this SD): {(stats.t.ppf(.975, n - 1) + stats.t.ppf(.8, n - 1)) * se:.2f} AUC")
Path("results/summary/p8_6d_sensitivity-posthoc.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

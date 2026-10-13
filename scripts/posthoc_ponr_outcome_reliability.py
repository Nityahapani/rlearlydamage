"""POST-HOC (not pre-registered): is recovery cost a reliable per-agent trait at a fixed duration?

Split-half reliability of A_d within duration, using odd vs even eval checkpoints of the same recovery run
(k = 10..250), plus early (k <= 100) vs late (k > 100) consistency and a between/within variance comparison.
Appends a labelled section to results/summary/<out>-analysis.md.
Usage: python scripts/posthoc_ponr_outcome_reliability.py results/summary/<pilot stem>.json --out p7b_predictors
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("summary")
ap.add_argument("--out", required=True)
a = ap.parse_args()
s = json.load(open(a.summary))
C = np.array(s["C"]["eval_return_mean"])
durs, K = s["durations"], s["recovery"] // s["protocol"]["eval_every"]
rows = []
for d in durs:
    on = np.array(s["arms"][str(d)]["D_online"]["eval_return_mean"])
    diff = on[:, 1:K + 1] - C[:, 1:K + 1]
    for i in range(diff.shape[0]):
        rows.append((d, diff[i, 0::2].mean(), diff[i, 1::2].mean(), diff[i, :10].mean(), diff[i, 10:].mean()))
r = np.array(rows)
ud = r[:, 0]
cen = lambda x: np.concatenate([x[ud == d] - x[ud == d].mean() for d in durs])
odd, even = r[:, 1], r[:, 2]
pw = np.corrcoef(cen(odd), cen(even))[0, 1]
A = (odd + even) / 2
res = dict(pooled_rho=stats.spearmanr(odd, even)[0], within_rho=stats.spearmanr(cen(odd), cen(even))[0],
           within_pearson=pw, spearman_brown=2 * pw / (1 + pw),
           early_late_within_rho=stats.spearmanr(cen(r[:, 3]), cen(r[:, 4]))[0],
           between_sd=float(np.std([A[ud == d].mean() for d in durs], ddof=1)),
           within_sd=float(np.sqrt(np.mean([A[ud == d].var(ddof=1) for d in durs]))))
lines = ["", "## POST-HOC (labelled, not pre-registered): is recovery cost a reliable per-agent trait at fixed duration?", "",
         f"- A_d odd vs even eval checkpoints (same run): pooled ρ {res['pooled_rho']:+.2f}; **within-duration ρ "
         f"{res['within_rho']:+.2f}**, Pearson {res['within_pearson']:+.2f} → Spearman-Brown reliability **{res['spearman_brown']:+.2f}**.",
         f"- Early (k ≤ 100) vs late (k > 100) recovery cost, within duration: ρ {res['early_late_within_rho']:+.2f}.",
         f"- SD of duration means {res['between_sd']:.2f} vs pooled within-duration SD {res['within_sd']:.2f}.",
         f"- Implication: with 4 seeds and 64-episode evals, seed-to-seed differences in recovery cost at a fixed duration are "
         f"mostly measurement noise. The ceiling on any predictor's within-duration correlation is about √{res['spearman_brown']:.2f} ≈ "
         f"{np.sqrt(max(res['spearman_brown'], 0)):.2f}. Reliable variation in recovery cost comes from duration."]
md = Path("results/summary") / f"{a.out}-analysis.md"
md.write_text(md.read_text() + "\n".join(lines) + "\n")
print("\n".join(lines))

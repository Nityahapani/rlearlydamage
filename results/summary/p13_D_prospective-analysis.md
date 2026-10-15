# Phase 13 D — frozen prospective diagnostic comparison

Independent prospective cohort; global Phase 13 Holm applied separately; not new-task or causal validation

| frozen model | held-out MSE | rank correlation (secondary) |
|---|---|---|
| training_mean | 33.053 | None |
| acute_gap | 32.271 | 0.25 |
| removal_entropy | 33.521 | 0.7735294117647058 |
| KL | 61.642 | 0.23823529411764705 |
| acute_gap_plus_KL | 68.780 | 0.2352941176470588 |
| removal_score | 38.287 | 0.10882352941176471 |

Primary gain vs acute gap: -113.1%; paired loss difference [-36.50948000726162, -98.03068279026618, 25.01172277574294]; one-sided p=0.887397.
Practical gain threshold = 10%, combined with global Phase 13 multiplicity requirement. No coefficients were fitted to this cohort.

# Phase 9 E3 — independent confirmatory point-of-no-return study and predictor test (p9_ponr_predictors)
> **Status (LOG Phase 9 deviation 2):** the pre-registered n = 32 was not reached (n = 16, seeds 300–315). Per the pre-registered fallback, all results below are **descriptive**; the decision labels printed by the script (TOST decision, predictor support) are NOT applied.

Summaries: 20261014-p7-p9-ponr-confirm2-s300to303-ppo-SpaceInvaders-MinAtar-63a8f1d.json, 20261014-p7-p9-ponr-confirm2-s304to307-ppo-SpaceInvaders-MinAtar-4a02dd2.json, 20261014-p7-p9-ponr-confirm2-s308to311-ppo-SpaceInvaders-MinAtar-88383ca.json, 20261014-p7-p9-ponr-confirm2-s312to315-ppo-SpaceInvaders-MinAtar-88383ca.json.
Fresh seeds (n = 16): 300–315. Movement restriction from u100 for d = 300; 250-update budget. Negative = damage beyond lost time.

## Primary 1: point of no return

- **L_300 = +6.0 [+1.9, +10.2]***; TOST ±5: p_lower 0.0000, p_upper 0.7006 → **INCONCLUSIVE** (TOST p = 0.7006)
- A_300 = +1.3 [-0.7, +3.3]; split-half (odd/even checkpoints) Spearman +0.64, Spearman–Brown 0.78
- Descriptive pooling with Phase 8 J5 seeds 100–107 (n = 24): L_300 +4.4 [+0.8, +7.9]* (TOST p = 0.3610); A_300 -0.7 [-3.4, +2.1]

## Primary 2: removal-time predictors vs A_300 (one-sided, expected ρ < 0; Holm across 4)

| predictor | split-half rel. | ρ with A_300 | 95% CI for ρ | one-sided p | Holm p | supported? | LOO R² gain | ρ with L_300 |
|---|---|---|---|---|---|---|---|---|
| kl_clean | +0.87 | -0.24 | [-0.66, +0.30] | 0.1871 | 0.7485 | no | -18% | -0.09 |
| cka_clean | +0.71 | -0.13 | [-0.59, +0.39] | 0.3164 | 0.8689 | no | -21% | -0.30 |
| move_gap_rel | +0.96 | -0.15 | [-0.60, +0.38] | 0.2896 | 0.8689 | no | -14% | -0.30 |
| value_err_clean | +0.65 | +0.18 | [-0.35, +0.62] | 0.7505 | 0.8689 | no | -8% | +0.53 |

Pre-registered conclusion: **no predictor supported.** Strongest negative associations compatible with the data (lower 95% bounds): kl_clean -0.66, cka_clean -0.59, move_gap_rel -0.60, value_err_clean -0.35.

Per-seed (seed: L_300, A_300): 300: +7.2, +3.6; 301: +13.0, +3.5; 302: +14.1, +7.2; 303: +9.8, -0.7; 304: +5.8, +4.0; 305: +2.5, -3.5; 306: +8.5, +5.8; 307: +16.1, +8.0; 308: -2.8, -2.3; 309: -4.3, -4.4; 310: -4.3, +1.0; 311: -1.2, -2.2; 312: +16.2, +1.5; 313: -2.1, -1.5; 314: +1.1, -0.6; 315: +17.2, +1.7

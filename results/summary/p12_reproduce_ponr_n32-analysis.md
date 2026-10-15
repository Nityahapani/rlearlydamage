# Phase 9 E3 — independent confirmatory point-of-no-return study and predictor test (p12_reproduce_ponr_n32)

Summaries: 20261014-p7-p9-ponr-confirm2-s300to303-ppo-SpaceInvaders-MinAtar-63a8f1d.json, 20261014-p7-p9-ponr-confirm2-s304to307-ppo-SpaceInvaders-MinAtar-4a02dd2.json, 20261014-p7-p9-ponr-confirm2-s308to311-ppo-SpaceInvaders-MinAtar-88383ca.json, 20261014-p7-p9-ponr-confirm2-s312to315-ppo-SpaceInvaders-MinAtar-88383ca.json, 20261014-p7-p9-ponr-confirm2-s316to319-ppo-SpaceInvaders-MinAtar-ac63459.json, 20261014-p7-p9-ponr-confirm2-s320to323-ppo-SpaceInvaders-MinAtar-adf81f2.json, 20261014-p7-p9-ponr-confirm2-s324to327-ppo-SpaceInvaders-MinAtar-adf81f2.json, 20261014-p7-p9-ponr-confirm2-s328to331-ppo-SpaceInvaders-MinAtar-adf81f2.json.
Fresh seeds (n = 32): 300–331. Movement restriction from u100 for d = 300; 250-update budget. Negative = damage beyond lost time.

## Primary 1: point of no return

- **L_300 = +2.3 [-4.6, +9.2]**; TOST ±5: p_lower 0.0188, p_upper 0.2147 → **INCONCLUSIVE** (TOST p = 0.2147)
- A_300 = -3.3 [-8.4, +1.9]; split-half (odd/even checkpoints) Spearman +0.76, Spearman–Brown 0.86

## Primary 2: removal-time predictors vs A_300 (one-sided, expected ρ < 0; Holm across 4)

| predictor | split-half rel. | ρ with A_300 | 95% CI for ρ | one-sided p | Holm p | supported? | LOO R² gain | ρ with L_300 |
|---|---|---|---|---|---|---|---|---|
| kl_clean | +0.95 | -0.48 | [-0.72, -0.14] | 0.0026 | 0.0103 | **yes** | +5% | -0.22 |
| cka_clean | +0.71 | -0.19 | [-0.51, +0.17] | 0.1443 | 0.2885 | no | -2% | -0.26 |
| move_gap_rel | +0.96 | -0.24 | [-0.55, +0.12] | 0.0904 | 0.2712 | no | -5% | -0.33 |
| value_err_clean | +0.71 | +0.03 | [-0.32, +0.38] | 0.5656 | 0.5656 | no | -3% | +0.10 |

Pre-registered conclusion: **supported predictor(s): kl_clean**

Per-seed (seed: L_300, A_300): 300: +7.2, +3.6; 301: +13.0, +3.5; 302: +14.1, +7.2; 303: +9.8, -0.7; 304: +5.8, +4.0; 305: +2.5, -3.5; 306: +8.5, +5.8; 307: +16.1, +8.0; 308: -2.8, -2.3; 309: -4.3, -4.4; 310: -4.3, +1.0; 311: -1.2, -2.2; 312: +16.2, +1.5; 313: -2.1, -1.5; 314: +1.1, -0.6; 315: +17.2, +1.7; 316: -11.2, -20.5; 317: +8.0, +2.5; 318: +13.8, +5.5; 319: +8.7, -3.4; 320: +5.7, -3.4; 321: +1.9, -8.1; 322: +8.1, -8.0; 323: +6.4, -8.1; 324: +14.2, -4.9; 325: -6.9, -2.4; 326: -7.5, -6.0; 327: +0.5, -1.3; 328: +0.3, -1.1; 329: +22.6, +9.1; 330: -92.0, -75.2; 331: +4.7, -0.7

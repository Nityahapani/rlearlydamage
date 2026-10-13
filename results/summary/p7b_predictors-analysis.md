# Phase 7b — removal-time predictors of recovery cost (p7b_predictors)

Pre-specification: LOG.md, commit aabd1ae (written before any of these metrics were computed). Data: `20261013-p7-ponr-pilot-move-ppo-SpaceInvaders-MinAtar-ba1802f` saved removal checkpoints, 16 units (seeds [0, 1, 2, 3] × d [50, 100, 200, 300]), movement restriction. Predictors compare D with the clean fork C at the same update on the onset bank (unmasked policies). Outcome A_d = delay-adjusted AUC over the 250-update budget (more negative = slower recovery beyond lost time).

## Primary: criteria vs A_d

| predictor | R0 split-half rel. (≥0.8) | R1 pooled ρ (≤−0.5) | R2 within-d ρ (≤−0.3) | per-d ρ (d=50,100,200,300) | R3 LOSO MSE gain (≥10%) | passes |
|---|---|---|---|---|---|---|
| kl_clean | +0.98 ✓ | -0.57 ✓ | +0.05 | +0.80, -0.40, -0.80, +0.80 ✗ | -25% ✗ | **fail** |
| cka_clean | +0.99 ✓ | -0.68 ✓ | -0.11 | +0.80, -0.80, +0.20, +0.60 ✗ | -8% ✗ | **fail** |
| move_gap_rel | +0.97 ✓ | -0.15 ✗ | +0.13 | +0.40, +0.20, -0.40, +0.40 ✗ | -8% ✗ | **fail** |
| value_err_clean | +0.87 ✓ | -0.54 ✓ | -0.35 | +0.40, -0.40, -0.80, -0.20 ✓ | -83% ✗ | **fail** |
| *acute deficit (baseline)* | – | +0.64 | +0.17 | +0.60, -0.40, +0.60, +0.80 | -16% | – |

## Verdict (pre-specified rule)

**No predictor passes R0–R3.** Pre-specified conclusion: no robust removal-time predictor among policy divergence, representation divergence, relevant-state action change and critic error. No causal test is pre-registered from this analysis.

## Secondary: same metrics vs A_early (k = 10..100; not used for selection)

| predictor | pooled ρ | within-d ρ | per-d ρ | LOSO gain |
|---|---|---|---|---|
| kl_clean | -0.49 | -0.08 | +0.20, -0.40, -0.20, +0.00 | -69% |
| cka_clean | -0.56 | -0.06 | +0.20, -0.80, +0.80, -1.00 | -42% |
| move_gap_rel | -0.25 | +0.04 | +0.40, +0.20, -1.00, +0.40 | -45% |
| value_err_clean | -0.47 | -0.09 | -0.60, -0.40, +0.00, -0.20 | -124% |
| acute | +0.61 | +0.01 | +0.40, -0.40, +0.40, -0.80 | -84% |

## Unit-level values

| seed | d | A_d | A_early | acute | kl_clean | cka_clean | move_gap_rel | value_err_clean |
|---|---|---|---|---|---|---|---|---|
| 0 | 50 | -2.7 | -3.1 | -42.2 | 0.359 | 0.194 | 0.233 | 5.439 |
| 1 | 50 | +1.2 | -5.8 | -24.4 | 0.452 | 0.353 | -0.019 | 8.071 |
| 2 | 50 | -1.6 | -8.3 | -52.5 | 0.768 | 0.378 | 0.292 | 8.009 |
| 3 | 50 | +1.8 | +1.4 | -41.4 | 0.992 | 0.445 | 0.309 | 6.843 |
| 0 | 100 | +1.6 | -2.7 | -46.8 | 0.621 | 0.351 | 0.209 | 8.739 |
| 1 | 100 | -5.7 | -8.0 | -45.2 | 0.687 | 0.491 | 0.106 | 9.018 |
| 2 | 100 | +1.1 | -4.0 | -68.9 | 1.046 | 0.392 | 0.417 | 9.785 |
| 3 | 100 | -3.0 | -5.6 | -59.5 | 2.478 | 0.647 | 0.501 | 12.867 |
| 0 | 200 | -1.1 | -6.0 | -78.1 | 1.396 | 0.447 | 0.360 | 11.966 |
| 1 | 200 | +1.1 | -1.9 | -84.4 | 1.074 | 0.601 | 0.207 | 13.883 |
| 2 | 200 | -1.5 | -8.0 | -95.0 | 1.382 | 0.584 | 0.399 | 16.334 |
| 3 | 200 | -1.8 | -2.8 | -86.1 | 1.452 | 0.591 | 0.261 | 20.018 |
| 0 | 300 | -8.2 | -15.4 | -107.3 | 2.053 | 0.621 | 0.360 | 13.900 |
| 1 | 300 | -7.9 | -17.0 | -95.5 | 1.840 | 0.712 | 0.239 | 15.591 |
| 2 | 300 | -4.9 | -16.3 | -104.4 | 2.288 | 0.677 | 0.473 | 16.086 |
| 3 | 300 | -9.3 | -15.6 | -108.7 | 1.818 | 0.676 | 0.297 | 30.624 |

Duration means of predictors: kl_clean: 0.643, 1.208, 1.326, 2.000; cka_clean: 0.343, 0.470, 0.556, 0.672; move_gap_rel: 0.204, 0.308, 0.307, 0.342; value_err_clean: 7.090, 10.102, 15.550, 19.050

## POST-HOC (labelled, not pre-registered): is recovery cost a reliable per-agent trait at fixed duration?

- A_d odd vs even eval checkpoints (same run): pooled ρ +0.62; **within-duration ρ +0.19**, Pearson +0.17 → Spearman-Brown reliability **+0.30**.
- Early (k ≤ 100) vs late (k > 100) recovery cost, within duration: ρ -0.01.
- SD of duration means 3.41 vs pooled within-duration SD 2.37.
- Implication: with 4 seeds and 64-episode evals, seed-to-seed differences in recovery cost at a fixed duration are mostly measurement noise. The ceiling on any predictor's within-duration correlation is about √0.30 ≈ 0.54. Reliable variation in recovery cost comes from duration.

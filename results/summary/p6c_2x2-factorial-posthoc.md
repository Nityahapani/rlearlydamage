# POST-HOC: paired 2x2 factorial model — identity x residence (Phase 6c)

User-requested after the Phase 6c results. Not pre-registered except the interaction (pre-registered as a secondary, descriptive contrast). All p-values below are unadjusted and descriptive.

Seeds (n = 12): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Metric: eval_eps05_return_mean. AUC = mean over +10..+650; windows early +10..150, middle +160..240, late +250..650.
Coding (+-1/2): identity I+ = restriction-era block removed (B, Bp), I- = pre-onset block removed (Cp, C); residence T+ = replacement in newest rows / long (B, Cp), T- = older rows / short (Bp, C).
Interaction = (B - Bp) - (Cp - C) = (B - Cp) - (Bp - C): positive = the identity effect is larger when the replacement stays long.

## Cell means (AUC) and marginal means

| | T+ long | T- short | identity margin |
|---|---|---|---|
| I+ (R removed) | B 36.66 | Bp 35.16 | 35.91 |
| I- (P removed) | Cp 35.53 | C 34.07 | 34.80 |
| residence margin | 36.10 | 34.61 | grand 35.35 |

Reference A_retain AUC 33.12.

## Model 1: two-way repeated-measures ANOVA (seeds as blocks; error = effect x seed), AUC

| source | SS | df | MS | F | p | partial eta^2 | effect [95% CI] |
|---|---|---|---|---|---|---|---|
| seed (blocks) | 101.55 | 11 | 9.23 | | | | |
| identity main | 14.81 | 1 | 14.81 | F(1,11) = 4.91 | 0.0488 | 0.31 | +1.11 [+0.01, +2.21] |
| identity main x seed (error) | 33.20 | 11 | 3.02 | | | | |
| residence main | 26.57 | 1 | 26.57 | F(1,11) = 6.82 | 0.0242 | 0.38 | +1.49 [+0.23, +2.74] |
| residence main x seed (error) | 42.87 | 11 | 3.90 | | | | |
| interaction | 0.01 | 1 | 0.01 | F(1,11) = 0.00 | 0.9697 | 0.00 | +0.04 [-2.35, +2.43] |
| interaction x seed (error) | 38.87 | 11 | 3.53 | | | | |
| total | 257.87 | 47 | | | | | |

## Model 2 (sensitivity): seed-blocked linear model, pooled residual (df = 33); equals the random-intercept mixed model for this balanced design

| effect | estimate | SE | t | p | 95% CI |
|---|---|---|---|---|---|
| identity main | +1.11 | 0.54 | 2.06 | 0.0472 | [+0.01, +2.21] |
| residence main | +1.49 | 0.54 | 2.76 | 0.0093 | [+0.39, +2.58] |
| interaction | +0.04 | 1.08 | 0.04 | 0.9690 | [-2.15, +2.23] |

Model 2 assumes one error variance for all effects; the per-effect error MS in Model 1 are identity main 3.02, residence main 3.90, interaction 3.53; pooled error MS 3.48.

## Effects by window (Model 1; estimate [95% CI], p; exact sign-flip p on |mean|)

| effect | AUC | early | middle | late |
|---|---|---|---|---|
| identity main | +1.11 [+0.01, +2.21], p 0.049, flip 0.048 | +1.48 [+0.79, +2.16], p 0.001, flip 0.001 | +0.78 [-0.35, +1.91], p 0.155, flip 0.154 | +1.05 [-0.42, +2.52], p 0.144, flip 0.159 |
| residence main | +1.49 [+0.23, +2.74], p 0.024, flip 0.027 | +0.05 [-0.36, +0.45], p 0.802, flip 0.793 | +1.08 [+0.24, +1.92], p 0.017, flip 0.009 | +2.11 [+0.13, +4.08], p 0.038, flip 0.040 |
| interaction | +0.04 [-2.35, +2.43], p 0.970, flip 0.975 | -1.53 [-2.50, -0.56], p 0.005, flip 0.006 | -0.17 [-2.43, +2.09], p 0.870, flip 0.869 | +0.66 [-3.04, +4.37], p 0.701, flip 0.720 |

## Per-seed cell AUCs and effects

| seed | B | Bp | Cp | C | (A) | identity main | residence main | interaction | B - Cp | Bp - C | B - Bp | Cp - C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 37.7 | 33.1 | 34.2 | 32.7 | 36.2 | +1.98 | +3.03 | +3.09 | +3.53 | +0.44 | +4.58 | +1.49 |
| 1 | 37.0 | 37.5 | 32.6 | 33.2 | 33.8 | +4.36 | -0.49 | +0.08 | +4.40 | +4.32 | -0.45 | -0.52 |
| 2 | 34.8 | 32.7 | 35.1 | 34.2 | 32.4 | -0.88 | +1.46 | +1.27 | -0.25 | -1.52 | +2.09 | +0.83 |
| 3 | 36.4 | 38.5 | 37.6 | 29.8 | 33.5 | +3.79 | +2.84 | -9.88 | -1.15 | +8.73 | -2.10 | +7.77 |
| 4 | 39.3 | 37.0 | 37.6 | 36.1 | 33.9 | +1.25 | +1.93 | +0.89 | +1.70 | +0.81 | +2.38 | +1.48 |
| 5 | 34.9 | 30.8 | 33.9 | 31.1 | 29.1 | +0.33 | +3.41 | +1.28 | +0.97 | -0.31 | +4.05 | +2.77 |
| 6 | 36.0 | 34.8 | 35.4 | 37.6 | 31.4 | -1.07 | -0.47 | +3.45 | +0.65 | -2.80 | +1.26 | -2.19 |
| 7 | 34.7 | 37.2 | 34.1 | 35.9 | 35.9 | +0.97 | -2.09 | -0.68 | +0.63 | +1.31 | -2.43 | -1.76 |
| 8 | 38.9 | 34.7 | 39.0 | 33.9 | 32.2 | +0.34 | +4.67 | -0.91 | -0.12 | +0.79 | +4.22 | +5.13 |
| 9 | 32.7 | 34.1 | 34.7 | 33.5 | 33.2 | -0.71 | -0.06 | -2.65 | -2.03 | +0.62 | -1.39 | +1.26 |
| 10 | 39.6 | 37.1 | 39.0 | 36.1 | 33.7 | +0.74 | +2.67 | -0.33 | +0.57 | +0.90 | +2.50 | +2.84 |
| 11 | 37.9 | 34.5 | 33.2 | 34.7 | 32.0 | +2.24 | +0.96 | +4.90 | +4.69 | -0.21 | +3.41 | -1.49 |
| mean | 36.66 | 35.16 | 35.53 | 34.07 | 33.12 | +1.11 | +1.49 | +0.04 | +1.13 | +1.09 | +1.51 | +1.47 |
| SD | | | | | | 1.74 | 1.97 | 3.76 | 2.11 | 2.94 | 2.52 | 2.92 |
| seeds > 0 | | | | | | 9/12 | 8/12 | 7/12 | 8/12 | 8/12 | 8/12 | 8/12 |

## Exact sign-flip sensitivity (AUC; all 2^12 = 4096 sign assignments; two-sided)

| effect | t-test p | flip p, mean | flip p, t | Wilcoxon exact p | LOSO estimate range | LOSO flip p (mean) range | most influential seed | without it: est, flip p |
|---|---|---|---|---|---|---|---|---|
| identity main | 0.0488 | 0.0479 | 0.0479 | 0.0640 | +0.82 .. +1.31 | 0.0234 .. 0.0957 | seed 1 (+4.36) | +0.82, 0.0957 |
| residence main | 0.0242 | 0.0269 | 0.0269 | 0.0425 | +1.20 .. +1.81 | 0.0088 .. 0.0537 | seed 7 (-2.09) | +1.81, 0.0088 |
| interaction | 0.9697 | 0.9746 | 0.9746 | 0.5186 | -0.40 .. +0.94 | 0.1816 .. 0.9805 | seed 3 (-9.88) | +0.94, 0.1816 |
| B - Cp | 0.0899 | 0.0972 | 0.0972 | 0.1294 | +0.81 .. +1.42 | 0.0312 .. 0.1943 | seed 11 (+4.69) | +0.81, 0.1943 |
| Bp - C | 0.2259 | 0.2505 | 0.2505 | 0.2036 | +0.40 .. +1.44 | 0.0742 .. 0.5010 | seed 3 (+8.73) | +0.40, 0.5010 |
| B - Bp | 0.0622 | 0.0674 | 0.0674 | 0.0771 | +1.23 .. +1.87 | 0.0293 .. 0.1299 | seed 7 (-2.43) | +1.87, 0.0293 |
| Cp - C | 0.1095 | 0.1079 | 0.1079 | 0.2036 | +0.89 .. +1.80 | 0.0518 .. 0.2158 | seed 3 (+7.77) | +0.89, 0.2158 |

Interaction: 3992 of 4096 sign assignments give |mean| >= the observed |0.042|. Smallest attainable two-sided flip p with n = 12: 0.00049.

## Sampled exposure per cell (millions of sampled transitions; 64 x 128 = 8,192 samples per update)

Restriction-era (R): realised, from the logged per-update sampled fraction (mean over seeds; per-seed range in brackets). Replacement (pool) and pre-onset block (P, rows written u150-249): expected = exact per-update buffer share x 8,192 (uniform sampling; realised counts not logged; cumulative binomial SD <= ~0.1%).

| cell | layout P2 / P1 | R +1..150 | R +151..240 | R +241..650 | R +1..650 | pool +1..150 | pool +151..240 | pool +241..650 | pool +1..650 | P +1..650 |
|---|---|---|---|---|---|---|---|---|---|---|
| B | P / pool | 0.000 [0.000-0.000] | 0.000 [0.000-0.000] | 0.000 [0.000-0.000] | 0.000 [0.000-0.000] | 0.503 | 0.147 | 0.000 | 0.650 | 0.314 |
| Bp | pool / P | 0.000 [0.000-0.000] | 0.000 [0.000-0.000] | 0.000 [0.000-0.000] | 0.000 [0.000-0.000] | 0.314 | 0.000 | 0.000 | 0.314 | 0.650 |
| Cp | R / pool | 0.314 [0.313-0.315] | 0.000 [0.000-0.000] | 0.000 [0.000-0.000] | 0.314 [0.313-0.315] | 0.503 | 0.147 | 0.000 | 0.650 | 0.000 |
| C | pool / R | 0.503 [0.501-0.503] | 0.147 [0.146-0.147] | 0.000 [0.000-0.000] | 0.650 [0.648-0.650] | 0.314 | 0.000 | 0.000 | 0.314 | 0.000 |
| A | P / R | 0.503 [0.501-0.503] | 0.147 [0.146-0.147] | 0.000 [0.000-0.000] | 0.650 [0.648-0.650] | 0.000 | 0.000 | 0.000 | 0.000 | 0.314 |

Exposure contrasts implied by the design (+1..650, M sampled):
- B vs Cp (identity, long): R -0.314, P +0.314, pool +0.000 (R and P swapped one-for-one in the older position).
- Bp vs C (identity, short): R -0.650, P +0.650, pool +0.000 (swapped in the newest position).
- B vs Bp (residence, R removed): pool +0.336, P -0.336; Cp vs C (residence, P removed): pool +0.336, R -0.336.
- Hence the residence factor is also a swap of which retained block is long-resident, and the identity contrast at short residence (Bp - C) swaps 2.1x more R exposure than at long residence (B - Cp).

Descriptive exposure-dose check (4 cells; per seed AUC = a + bR * R exposure + bpool * pool exposure; P is the reference data; 1 df lack of fit):
- bR = -2.04 per M sampled restriction-era transitions [-4.45, +0.36]; bpool = +3.41 per M replacement [-0.43, +7.25] (relative to P).
- Lack-of-fit contrast (weights B, Bp, Cp, C = [1.0, -0.48, -1.0, 0.48]): +0.61 [-1.07, +2.28], p 0.444, flip p 0.441. A linear-in-exposure model predicts Bp - C ~ 2.1x B - Cp; observed +1.09 vs +1.13.

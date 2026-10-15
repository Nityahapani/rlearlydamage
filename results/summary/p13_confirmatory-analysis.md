# Phase 13 — registered claim-family analysis

five NEW Phase 13 claims only, not retrospective paper-wide control

| study | n | primary p (compound where specified) | Phase-13 Holm p | status |
|---|---|---|---|---|
| A | 8 | 0.0310171 | 0.0930512 | planned cohort complete |
| B | 8 | 3.43335e-05 | 0.000171667 | planned cohort complete |
| C | 8 | 0.00083166 | 0.00332664 | planned cohort complete |
| E | 0 | 1 | 1 | not run / incomplete (no primary inference) |
| D | 0 | 1 | 1 | not run / incomplete (no primary inference) |

## A: natural benchmark and contexts (n=8)

Native damage beyond historical pause: -155.01 [-220.07, -89.96]*; last five checkpoints -22.54 [-78.93, +33.85].
Natural-state identity checks: True.
Context shift, standardized DDD minus natural D: -12.54 [-44.17, +19.09]; joint-scale TOST p=0.2107.

| reference/context/moments | total gap | actor effect / bad-actor transfer | good-actor rescue |
|---|---|---|---|
| fork/clean/donor | -195.14 [-287.55, -102.74]* | -266.32 [-355.06, -177.58]* | +124.81 [+11.38, +238.24]* |
| fork/damaged/donor | -181.37 [-245.81, -116.92]* | -253.00 [-332.44, -173.56]* | +97.68 [+8.53, +186.83]* |
| pause/clean/donor | -160.56 [-251.89, -69.23]* | -147.95 [-245.48, -50.42]* | +103.00 [+8.15, +197.85]* |
| pause/clean/zero | -179.01 [-280.70, -77.33]* | -126.91 [-236.80, -17.02]* | zero-moment 2x2 |
| pause/damaged/donor | -141.80 [-189.77, -93.82]* | -139.17 [-225.13, -53.20]* | +105.98 [+38.21, +173.75]* |
| pause/reset/donor | -152.30 [-254.77, -49.83]* | -122.32 [-234.60, -10.05]* | +117.79 [+24.88, +210.69]* |

fork/clean/donor: actor/critic/Adam main effects actor -183.39 [-276.49, -90.29]*, critic -18.63 [-27.29, -9.97]*, opt +19.05 [-1.79, +39.89].
Shapley shares (only if total CI <0): {'actor': 96.05572548014818, 'critic': 11.625878092004825, 'opt': -7.681603572153007}.

fork/damaged/donor: actor/critic/Adam main effects actor -167.70 [-243.71, -91.70]*, critic -25.73 [-62.96, +11.51], opt +19.70 [+7.81, +31.59]*.
Shapley shares (only if total CI <0): {'actor': 93.86970580675916, 'critic': 15.588192640646021, 'opt': -9.457898447405183}.

pause/clean/donor: actor/critic/Adam main effects actor -117.37 [-195.48, -39.25]*, critic -48.64 [-69.47, -27.80]*, opt +13.55 [-4.38, +31.47].
Shapley shares (only if total CI <0): {'actor': 74.78032366070825, 'critic': 31.97457847498964, 'opt': -6.754902135697888}.

pause/damaged/donor: actor/critic/Adam main effects actor -118.41 [-187.50, -49.33]*, critic -35.53 [-51.47, -19.60]*, opt +16.31 [-3.35, +35.97].
Shapley shares (only if total CI <0): {'actor': 84.48491231807667, 'critic': 26.038403005048, 'opt': -10.523315323124683}.

## B: natural benchmark and contexts (n=8)

Native damage beyond historical pause: -1.57 [-1.84, -1.30]*; last five checkpoints -0.88 [-1.56, -0.19]*.
Natural-state identity checks: True.
Context shift, standardized DDD minus natural D: +0.02 [-0.30, +0.34]; joint-scale TOST p=0.04369.

| reference/context/moments | total gap | actor effect / bad-actor transfer | good-actor rescue |
|---|---|---|---|
| pause/clean/donor | -1.44 [-1.84, -1.03]* | -1.33 [-1.70, -0.97]* | +1.25 [+0.98, +1.51]* |
| pause/damaged/donor | -1.50 [-1.86, -1.14]* | -1.39 [-1.79, -0.99]* | +1.46 [+1.22, +1.70]* |

pause/clean/donor: actor/critic/Adam main effects actor -1.47 [-1.70, -1.24]*, critic -0.20 [-0.40, -0.01]*, opt +0.06 [-0.15, +0.26].
Shapley shares (only if total CI <0): {'actor': 98.0838486104505, 'critic': 9.944282122715226, 'opt': -8.028130733165733}.

## C: natural benchmark and contexts (n=8)

Native damage beyond historical pause: +3.49 [+0.21, +6.77]*; last five checkpoints +5.62 [+3.26, +7.97]*.
Natural-state identity checks: True.
Context shift, standardized DDD minus natural D: -1.12 [-3.91, +1.68]; joint-scale TOST p=0.01419.

| reference/context/moments | total gap | actor effect / bad-actor transfer | good-actor rescue |
|---|---|---|---|
| fork/clean/donor | -12.45 [-18.41, -6.50]* | -11.57 [-14.96, -8.18]* | +11.80 [+7.22, +16.37]* |
| fork/clean/zero | -13.01 [-17.40, -8.63]* | -12.50 [-16.99, -8.00]* | zero-moment 2x2 |
| fork/damaged/donor | -12.72 [-18.52, -6.92]* | -14.90 [-21.59, -8.21]* | +10.16 [+7.18, +13.14]* |

fork/clean/donor: actor/critic/Adam main effects actor -11.85 [-16.45, -7.24]*, critic -0.92 [-2.44, +0.60], opt +0.15 [-1.66, +1.96].
Shapley shares (only if total CI <0): {'actor': 94.70817039252178, 'critic': 6.943892594993632, 'opt': -1.652062987515426}.

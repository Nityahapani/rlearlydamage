# Phase 13 — registered claim-family analysis

five NEW Phase 13 claims only, not retrospective paper-wide control

| study | n | primary p (compound where specified) | Phase-13 Holm p | status |
|---|---|---|---|---|
| A | 8 | 0.0310171 | 0.155085 | planned cohort complete |
| B | 0 | 1 | 1 | not run / incomplete (no primary inference) |
| C | 0 | 1 | 1 | not run / incomplete (no primary inference) |
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

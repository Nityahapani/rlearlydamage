# POST-HOC: factorial interactions and relative effects — p10_interactions_move_n12

Summaries: 20261011-p3-surgery-move-late-ppo-SpaceInvaders-MinAtar-365cf2b.json, 20261013-p3-p8-surgery-move-late-s4to7-ppo-SpaceInvaders-MinAtar-8c62fe0.json, 20261013-p3-p8-surgery-move-late-s8to11-ppo-SpaceInvaders-MinAtar-8a2de07.json; seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] (n = 12). AUC over +E..+R; paired 95% t-CIs. Reference cell CCC mean AUC 80.6; 'relative' = mean effect / reference AUC.

| contrast | AUC effect | relative |
|---|---|---|
| total DDD - CCC | -15.09 [-17.21, -12.98]* | -18.7% |
| actor swap from reference, DCC - CCC | -13.34 [-16.23, -10.45]* | -16.6% |
| critic swap from reference, CDC - CCC | -0.05 [-1.94, +1.84] | -0.1% |
| Adam swap from reference, CCD - CCC | +1.08 [-0.79, +2.95] | +1.3% |
| actor swap into damaged, DDD - CDD | -13.29 [-15.57, -11.01]* | -16.5% |
| critic swap into damaged, DDD - DCD | -3.04 [-5.31, -0.78]* | -3.8% |
| Adam swap into damaged, DDD - DDC | +0.05 [-1.94, +2.04] | +0.1% |
| interaction actor x critic | -0.48 [-1.88, +0.91] | -0.6% |
| interaction actor x Adam | +0.50 [-1.04, +2.05] | +0.6% |
| interaction critic x Adam | -1.02 [-2.13, +0.09] | -1.3% |
| three-way interaction | +0.40 [-0.41, +1.20] | +0.5% |
| non-additivity: total - sum of swaps from reference | -2.78 [-5.91, +0.35] | -3.5% |

Cell means: CCC 80.6, CCD 81.6, CDC 80.5, CDD 78.8, DCC 67.2, DCD 68.5, DDC 65.4, DDD 65.5

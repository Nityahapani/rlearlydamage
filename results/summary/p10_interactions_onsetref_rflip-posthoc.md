# POST-HOC: factorial interactions and relative effects — p10_interactions_onsetref_rflip

Summaries: 20261014-p3-p10-onsetref-rflip-s0to3-ppo-SpaceInvaders-MinAtar-f618d6c.json, 20261014-p3-p10-onsetref-rflip-s4to7-ppo-SpaceInvaders-MinAtar-36ced59.json; seeds [0, 1, 2, 3, 4, 5, 6, 7] (n = 8). AUC over +E..+R; paired 95% t-CIs. Reference cell PPP mean AUC 67.4; 'relative' = mean effect / reference AUC.

| contrast | AUC effect | relative |
|---|---|---|
| total DDD - PPP | -19.78 [-27.81, -11.76]* | -29.4% |
| actor swap from reference, DPP - PPP | -20.75 [-24.70, -16.80]* | -30.8% |
| critic swap from reference, PDP - PPP | +0.09 [-2.51, +2.70] | +0.1% |
| Adam swap from reference, PPD - PPP | -0.31 [-2.37, +1.76] | -0.5% |
| actor swap into damaged, DDD - PDD | -21.19 [-27.51, -14.88]* | -31.5% |
| critic swap into damaged, DDD - DPD | +7.52 [-3.15, +18.19] | +11.2% |
| Adam swap into damaged, DDD - DDP | -0.97 [-3.54, +1.60] | -1.4% |
| interaction actor x critic | +1.91 [-2.23, +6.05] | +2.8% |
| interaction actor x Adam | -2.13 [-4.09, -0.18]* | -3.2% |
| interaction critic x Adam | +1.80 [-0.28, +3.88] | +2.7% |
| three-way interaction | +0.99 [-0.21, +2.19] | +1.5% |
| non-additivity: total - sum of swaps from reference | +1.18 [-7.23, +9.58] | +1.7% |

Cell means: PPP 67.4, PPD 67.1, PDP 67.5, PDD 68.8, DPP 46.6, DPD 40.1, DDP 48.6, DDD 47.6

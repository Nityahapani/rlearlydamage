# Phase 12 G7 — PQN-R reward-flip network × buffer surgery

Registered G7 primary (n=8), bf9a946

Seeds: [0, 1, 2, 3, 4, 5, 6, 7]; epsilon=.05; deficit [250,350), primary recovery +10..+650 (2.662M steps).
Network = parameters + RAdam state; buffer = all buffer arrays. Environment/obs/RNG/counter from C. Wrong rewards are stored during the restriction window; evaluation always uses the true reward.

| contrast | primary AUC, 95% t-CI | primary p | matched-horizon AUC (secondary), 95% t-CI |
|---|---|---|---|
| total DD - CC | -6.78 [-9.42, -4.15]* | 0.000492507 | -11.85 [-15.11, -8.59]* |
| network main | -6.22 [-8.32, -4.13]* | 0.000208067 | -9.71 [-11.94, -7.48]* |
| buffer main | -0.56 [-1.73, +0.61] | 0.296641 | -2.14 [-4.00, -0.28]* |
| network transfer DC - CC | -5.05 [-6.84, -3.26]* | 0.000288898 | -8.16 [-10.60, -5.71]* |
| data transfer CD - CC | +0.62 [-1.03, +2.26] | 0.40439 | -0.59 [-3.01, +1.83] |
| network rescue CD - DD | +7.40 [+4.59, +10.21]* | 0.00043576 | +11.26 [+8.13, +14.40]* |
| buffer rescue DC - DD | +1.74 [+0.19, +3.28]* | 0.0327879 | +3.69 [+1.05, +6.33]* |
| interaction | -2.35 [-4.51, -0.19]* | 0.0367035 | -3.11 [-6.53, +0.32] |

## Registered primary decisions

- Same-update gap detected (DD−CC upper CI <0): True.
- Network carrier criterion (negative main effect; larger magnitude than buffer): True.
- Buffer carrier criterion (negative main effect): False.
- Buffer TOST at ±4.933 (10% of |CC AUC|): p=2.45712e-05.
- Shares (only if total gap detected): {'network': 91.74620451725241, 'buffer': 8.25379548274759}.

## Matched horizon (registered secondary, 3ddffe2)

PQN-R +20,+40,...,+300 exactly matches PPO +10,+20,...,+150 in environment steps: 15 checkpoints over 1,228,800 recovery steps. Deficit duration is also matched (409,600 steps). This is not the original G7 primary.

## Additional sensitivity (post-hoc)

Holm across network/buffer: network 0.000416134, buffer 0.296641. Covariance-aware control-scaled buffer TOST p=1.3756e-05.

## Overwrite diagnostics

| arm | first post-removal buffer fraction | fraction at +25 | first update with fraction zero | mean sampled window-era fraction |
|---|---|---|---|---|
| CC | 0.9590 | 0.0000 | 25 | 0.0180 |
| DC | 0.9590 | 0.0000 | 25 | 0.0180 |
| CD | 0.9590 | 0.0000 | 25 | 0.0180 |
| DD | 0.9590 | 0.0000 | 25 | 0.0180 |

## Historical pause benchmark (post-hoc)

This compares the standardized-context DD continuation with historical C shifted by 100 updates. It is not a new attribution to network/buffer and does not change the registered same-update decisions.
- Full-horizon pause-relative AUC: -2.48 [-4.98, +0.03].
- Matched-horizon pause-relative AUC: -4.81 [-7.78, -1.85]*.
- Last five checkpoints: +0.52 [-3.40, +4.43].

# Phase 8 J5 — confirmatory point-of-no-return test at d = 300 (PPO, movement restriction)

Confirmatory summaries: 20261013-p7-p8-ponr-confirm-s100to103-ppo-SpaceInvaders-MinAtar-975ad08.json, 20261013-p7-p8-ponr-confirm-s104to107-ppo-SpaceInvaders-MinAtar-b634055.json; pilot: 20261013-p7-ponr-pilot-move-ppo-SpaceInvaders-MinAtar-ba1802f.json.
Fresh seeds (n = 8): [100, 101, 102, 103, 104, 105, 106, 107]. Onset 100, removal 400, budget 250 updates. Negative = damage beyond lost time.

- Unmasked eval: C at onset 46.3; at removal D 11.6 vs C 106.2
- **L_300 (primary): +1.1 [-6.6, +8.7]**; TOST ±5: p_lower 0.0513, p_upper 0.1302
- Pre-registered decision: **INCONCLUSIVE** (TOST p = 0.1302; CI [-6.6, +8.7])
- A_300 (delay-adjusted AUC): -4.6 [-12.3, +3.0]
- raw AUC vs C(w1 + k): -45.9 [-53.4, -38.3]*
- transplant − D_online (AUC): +39.4 [+32.0, +46.9]*; transplant vs C(w1 + k): -6.4 [-9.5, -3.4]*; transplant late delay-adj.: +26.6 [+20.5, +32.7]*

Per-seed L_300: 100: +8.3, 101: -17.6, 102: +2.2, 103: -5.1, 104: +9.6, 105: +8.9, 106: -1.0, 107: +3.1
Per-seed A_300: 100: +3.8, 101: -21.3, 102: -7.5, 103: -4.3, 104: -13.5, 105: +4.7, 106: +3.4, 107: -2.4

## Secondary (descriptive)

- Pooled with pilot seeds 0–3 (n = 12): L_300 +0.2 [-4.7, +5.1] (TOST ±5 p = 0.0279); A_300 -5.6 [-10.4, -0.8]*
- Split-half reliability of A_300 across the 8 confirmatory seeds: odd/even-checkpoint Spearman +0.90, Spearman-Brown 0.95

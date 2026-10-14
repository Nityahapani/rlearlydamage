# Phase 9 E1 — confirmatory recovery estimands (p9_estimands)

Sources: 20261014-p2-p9-recovery-move-late-s200to203-ppo-SpaceInvaders-MinAtar-63a8f1d.json, 20261014-p2-p9-recovery-move-late-s204to207-ppo-SpaceInvaders-MinAtar-63a8f1d.json, 20261014-p2-p9-recovery-move-late-s208to211-ppo-SpaceInvaders-MinAtar-63a8f1d.json; arm `move_late` window [100, 150); seeds [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211] (n = 12, fresh). AUC over k = 10..250 after removal; paired 95% t-CIs; negative = restricted agent worse.

- Eval at removal: D 24.4; C at onset 41.3; C at removal 60.7
- t* (competence-matched restart point): mean 52, range 30–90

**E_damage (primary; pause counterfactual): +1.9 [-0.7, +4.4]**; TOST ±3: p_lower 0.0007, p_upper 0.1710 → **INCONCLUSIVE** (TOST p = 0.1710)
**E_plast (primary for plasticity; competence-matched): +14.2 [+10.9, +17.5]*** (p = 0.0000) → **SAVINGS** (faster relearning from equal competence)
E_total (descriptive; same-update): -8.9 [-11.1, -6.6]*

Per-seed (damage / plast / total): 200: +3.1 / +21.9 / -6.7; 201: -3.0 / +10.3 / -11.6; 202: +0.6 / +16.0 / -9.1; 203: -0.2 / +11.1 / -11.3; 204: +9.9 / +18.2 / -1.4; 205: +2.7 / +21.6 / -8.4; 206: +0.7 / +16.6 / -9.9; 207: -0.7 / +10.5 / -12.4; 208: +4.5 / +17.1 / -6.4; 209: +4.7 / +13.0 / -6.9; 210: +4.9 / +7.4 / -7.3; 211: -5.0 / +6.6 / -14.8

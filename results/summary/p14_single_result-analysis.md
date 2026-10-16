# Phase 14 — one focused independent actor-repair result

fixed n=32 complete; one registered claim, all robustness retained

Primary alpha=.01; compound p=3.3191367e-08; prespecified single claim success: **True**.
Seedwise consistency: {'native_AUC_negative': 32, 'actor_rescue_positive': 32}.

| primary contrast | effect [95% CI] | 99% CI | threshold-shifted p |
|---|---|---|---|
| native_AUC_harm | -1.722 [-1.940, -1.504] | -1.722 [-2.015, -1.429] | 6.021439e-13 |
| native_late_harm | -1.187 [-1.444, -0.930] | -1.187 [-1.533, -0.841] | 3.319137e-08 |
| onset_actor_rescue | +1.580 [+1.371, +1.790] | +1.580 [+1.298, +1.862] | 4.789038e-12 |
| reciprocal_actor_transfer | -1.636 [-1.828, -1.445] | -1.636 [-1.894, -1.379] | 1.390402e-13 |

Point fraction of native historical-pause AUC cost repaired: 91.8%; paired-bootstrap95 interval [0.862667053031215, 0.9750230100266715]. Not an information fraction.

Residual vs matched state-pause: {'CI95': [-0.2308837890625, -0.37030659033183555, -0.09146098779316444], 'TOST_pm_030': 0.15990964402648608}.
Residual vs historical pause: {'CI95': [-0.14147949218750006, -0.24615047841647208, -0.03680850595852801], 'TOST_pm_030': 0.002108786698276272}.
Common fresh-Adam checks: {'actor_main': [-1.59415283203125, -1.76687868499767, -1.4214269790648297], 'critic_main': [-0.12918701171874997, -0.22911518148629534, -0.02925884195120458], 'actor_rescue': [1.6259033203125, 1.4202937059717333, 1.8315129346532666], 'actor_in_onset_critic': [-1.56240234375, -1.7365034177805425, -1.3883012697194577], 'actor_in_damaged_critic': [-1.6259033203125, -1.8315129346532666, -1.4202937059717333], 'critic_in_onset_actor': [-0.09743652343750003, -0.22712063180180378, 0.03224758492680371], 'critic_in_damaged_actor': [-0.1609374999999999, -0.28771039610137705, -0.03416460389862272], 'total': [-1.7233398437499998, -1.936496782914738, -1.5101829045852617]}.
Reset-context actor rescue: +1.515 [+1.308, +1.722].
Independent evaluation key54321: {'native_AUC_harm': [-1.59921875, -1.7624982420662434, -1.4359392579337564], 'native_late_harm': [-1.0700195312499998, -1.3149073506863802, -0.8251317118136194], 'onset_actor_rescue': [1.3981445312499998, 1.227266563468765, 1.5690224990312347]}.

All CIs pair training seeds, not episodes or checkpoints. Secondary equivalence/robustness does not rescue a failed primary. The result is scoped to Asterix PPO, this impairment and finite horizon; it does not identify a unique naturally encoded representation or prove irreversibility.

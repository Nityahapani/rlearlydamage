# Positioning the focused result

These notes support the expanded discussion, not a claim of priority or a new literature benchmark.

- Primacy-bias resets periodically reinitialize network components while retaining replay data; our PPO treatment restores its own known onset actor and has no replay buffer. This distinguishes treatments, not demonstrated superiority. [1](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf)
- Plasticity injection changes trainable state while avoiding an immediate prediction change. Our actor rollback deliberately changes the policy and future data, so its same-task return improvement cannot diagnose new-target capacity by itself. [4](https://proceedings.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf)
- On-policy domain-shift studies find that remedies can behave differently from off-policy settings; this motivates an explicit algorithm/context limit rather than a universal recovery rule. [14](https://proceedings.neurips.cc/paper_files/paper/2024/file/ce7984e36d58659211a8dc7d5457cd6f-Paper-Conference.pdf)
- Ma etal study visual-RL plasticity and identify the critic as a bottleneck in their tested regimes. That is not contradicted by a useful actor intervention in our different setting; neither observation establishes a context-free module ranking. [5](https://arxiv.org/html/2310.07418)

The contribution here is a counterfactual/reference-aware assay and a fixed32-seed confirmation within one prior-selected condition: a donor from BEFORE impairment repairs measured pause-relative cost without importing the clean fork's extra50 updates. We do not compare fresh resets, plasticity injection or distillation head-to-head on the primary Asterix condition. State reconstruction is reproducibility engineering, not an additional independent empirical confirmation.

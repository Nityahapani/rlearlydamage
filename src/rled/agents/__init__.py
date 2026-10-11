"""Agents. Phase 1 will add:
- pqn.py : off-policy, PQN/DQN-style value learning with a replay buffer ("data" = buffer)
- ppo.py : on-policy PPO ("data" = behaviour-policy params used for rollouts)

Both must expose: init(cfg, key) -> TrainerState dict with keys params/opt_state/data/(env,rng),
step(state, deficit_active) -> state, and evaluate(state) -> returns.
"""

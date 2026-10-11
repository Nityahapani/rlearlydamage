"""On-policy PPO for gymnax/MinAtar, PureJaxRL style, with deficit hooks and segmentable training.

Design for state surgery: training is split into `init(rng) -> runner_state` and
`run(runner_state, n_updates) -> (runner_state, metrics)`. Training can therefore stop at the
deficit-removal step, we can operate on the state, and training can resume.

Carriers in the runner state:
  params    -> runner_state["train_state"].params
  opt_state -> runner_state["train_state"].opt_state
  deficit   -> runner_state["deficit_bounds"] = [start, end) env steps (overridable after init)
  data      -> on-policy: the behaviour policy. Swapping it is handled by `run(..., behaviour_params=...)`
               (Phase 3).

Deficits (applied on the agent side; LogWrapper still records the *true* episode return):
  obs_noise    : obs + sigma * N(0, 1)
  action_mask  : only `allowed_actions` (fixed list, same for all seeds) are available
                 [legacy: `allowed_frac` draws a random subset per seed]
  reward_flip  : r -> -r with probability p
  reward_delay : rewards are banked and paid out every `delay` steps (or at episode end)
"""
from __future__ import annotations

from typing import NamedTuple

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

DEFAULT_CONFIG = dict(
    ENV_NAME="Breakout-MinAtar",
    TOTAL_TIMESTEPS=int(5e6),
    NUM_ENVS=64,
    NUM_STEPS=128,
    UPDATE_EPOCHS=4,
    NUM_MINIBATCHES=8,
    LR=5e-4,               # constant LR by design (critical-period sweeps)
    GAMMA=0.99,
    GAE_LAMBDA=0.95,
    CLIP_EPS=0.2,
    ENT_COEF=0.01,
    VF_COEF=0.5,
    MAX_GRAD_NORM=0.5,
    HIDDEN=256,
    LAYERNORM=False,
    # deficit window, in env steps (set from DeficitWindow.bounds)
    DEFICIT="none",
    DEFICIT_START=0,
    DEFICIT_END=0,
    DEFICIT_PARAMS={},
)


class ActorCritic(nn.Module):
    action_dim: int
    hidden: int = 256
    layernorm: bool = False

    @nn.compact
    def __call__(self, x):
        def block(h, name):
            h = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=name)(h)
            if self.layernorm:
                h = nn.LayerNorm(name=name + "_ln")(h)
            return nn.relu(h)

        a = block(x, "a0")
        a = block(a, "a1")
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        c = block(x, "c0")
        c = block(c, "c1")
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return logits, jnp.squeeze(v, -1)


ACTOR_KEYS = ("a0", "a1", "Dense_0")   # actor tower + logits head
CRITIC_KEYS = ("c0", "c1", "Dense_1")  # critic tower + value head


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray
    info: dict


def make(config: dict):
    cfg = {**DEFAULT_CONFIG, **config}
    cfg["NUM_UPDATES"] = cfg["TOTAL_TIMESTEPS"] // (cfg["NUM_STEPS"] * cfg["NUM_ENVS"])
    cfg["MINIBATCH_SIZE"] = cfg["NUM_ENVS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]
    steps_per_update = cfg["NUM_STEPS"] * cfg["NUM_ENVS"]

    env, env_params = gymnax.make(cfg["ENV_NAME"])
    env = LogWrapper(FlattenObservationWrapper(env))
    n_actions = env.action_space(env_params).n
    obs_dim = int(np.prod(env.observation_space(env_params).shape))
    net = ActorCritic(n_actions, cfg["HIDDEN"], cfg["LAYERNORM"])
    dp = cfg["DEFICIT_PARAMS"]
    dtype = cfg["DEFICIT"]
    d_start, d_end = cfg["DEFICIT_START"], cfg["DEFICIT_END"]

    def init(rng):
        rng, k_net, k_env, k_mask = jax.random.split(rng, 4)
        params = net.init(k_net, jnp.zeros((obs_dim,)))
        tx = optax.chain(optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]), optax.adam(cfg["LR"], eps=1e-5))
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_env, cfg["NUM_ENVS"]), env_params)
        if "allowed_actions" in dp:  # fixed, semantically chosen set, identical for every seed
            action_mask = jnp.zeros((n_actions,), bool).at[jnp.array(dp["allowed_actions"])].set(True)
        else:  # legacy: random subset drawn per seed (differs in kind across seeds; avoid)
            n_allowed = max(1, int(round(dp.get("allowed_frac", 1.0) * n_actions)))
            perm = jax.random.permutation(k_mask, n_actions)
            action_mask = jnp.zeros((n_actions,), bool).at[perm[:n_allowed]].set(True)
        return dict(
            train_state=ts,
            env_state=env_state,
            obs=obs,
            rng=rng,
            update=jnp.array(0, jnp.int32),
            reward_bank=jnp.zeros((cfg["NUM_ENVS"],)),
            action_mask=action_mask,
            # deficit window [start, end) in env steps; a traced array so one compile serves a whole sweep
            deficit_bounds=jnp.array([d_start, d_end], jnp.int32),
            # behaviour-policy swap (surgery "data" carrier): while update < behaviour_until, rollouts
            # are collected with `behaviour_actor` instead of the learner's actor (the critic is always
            # the learner's). PPO's ratio uses the behaviour log-prob, i.e. importance-weighted updates.
            behaviour_actor={k: v for k, v in params["params"].items() if k in ACTOR_KEYS},
            behaviour_until=jnp.array(0, jnp.int32),
        )

    def _is_active(t, bounds):
        if dtype == "none":
            return jnp.array(False)
        return (t >= bounds[0]) & (t < bounds[1])

    def _update(rs, _):
        ts = rs["train_state"]

        def env_step(carry, step_idx):
            env_state, obs, rng, bank = carry
            t = rs["update"] * steps_per_update + step_idx * cfg["NUM_ENVS"]
            active = _is_active(t, rs["deficit_bounds"])
            rng, k_noise, k_act, k_step, k_flip = jax.random.split(rng, 5)
            obs_in = obs
            if dtype == "obs_noise":
                obs_in = obs + active * dp.get("sigma", 1.0) * jax.random.normal(k_noise, obs.shape)
            mask = jnp.ones((cfg["NUM_ENVS"], n_actions), bool)
            if dtype == "action_mask":
                mask = jnp.where(active, jnp.broadcast_to(rs["action_mask"], mask.shape), mask)
            use_b = rs["update"] < rs["behaviour_until"]
            p_act = {"params": {k: (jax.tree.map(lambda b, q: jnp.where(use_b, b, q), rs["behaviour_actor"][k], v)
                                    if k in ACTOR_KEYS else v) for k, v in ts.params["params"].items()}}
            logits, value = net.apply(p_act, obs_in)
            pi = distrax.Categorical(logits=jnp.where(mask, logits, -1e9))
            action = pi.sample(seed=k_act)
            log_prob = pi.log_prob(action)
            obs2, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(k_step, cfg["NUM_ENVS"]), env_state, action, env_params
            )
            r = reward
            if dtype == "reward_flip":
                flip = active & (jax.random.uniform(k_flip, r.shape) < dp.get("prob", 1.0))
                r = jnp.where(flip, -r, r)
            if dtype == "reward_delay":
                bank = bank + r
                release = (~active) | (step_idx % dp.get("delay", 20) == 0) | done
                r = jnp.where(release, bank, 0.0)
                bank = jnp.where(release, 0.0, bank)
            tr = Transition(done, action, value, r, log_prob, obs_in, mask, info)
            return (env_state, obs2, rng, bank), tr

        (env_state, obs, rng, bank), traj = jax.lax.scan(
            env_step, (rs["env_state"], rs["obs"], rs["rng"], rs["reward_bank"]), jnp.arange(cfg["NUM_STEPS"])
        )
        # bootstrap value (deficit view at the next step, for consistency)
        t_next = (rs["update"] + 1) * steps_per_update
        rng, k_noise = jax.random.split(rng)
        obs_last = obs
        if dtype == "obs_noise":
            obs_last = obs + _is_active(t_next, rs["deficit_bounds"]) * dp.get("sigma", 1.0) * jax.random.normal(k_noise, obs.shape)
        _, last_val = net.apply(ts.params, obs_last)

        def gae_fn(carry, tr):
            gae, next_val = carry
            delta = tr.reward + cfg["GAMMA"] * next_val * (1 - tr.done) - tr.value
            gae = delta + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1 - tr.done) * gae
            return (gae, tr.value), gae

        _, adv = jax.lax.scan(gae_fn, (jnp.zeros_like(last_val), last_val), traj, reverse=True)
        targets = adv + traj.value

        def epoch(carry, _):
            ts, rng = carry
            rng, k = jax.random.split(rng)
            bs = cfg["MINIBATCH_SIZE"] * cfg["NUM_MINIBATCHES"]
            perm = jax.random.permutation(k, bs)
            batch = (traj, adv, targets)
            batch = jax.tree.map(lambda x: x.reshape((bs,) + x.shape[2:]), batch)
            batch = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), batch)
            mbs = jax.tree.map(lambda x: x.reshape((cfg["NUM_MINIBATCHES"], -1) + x.shape[1:]), batch)

            def mb_step(ts, mb):
                tr, a, tgt = mb

                def loss_fn(p):
                    logits, v = net.apply(p, tr.obs)
                    pi = distrax.Categorical(logits=jnp.where(tr.mask, logits, -1e9))
                    lp = pi.log_prob(tr.action)
                    v_clip = tr.value + (v - tr.value).clip(-cfg["CLIP_EPS"], cfg["CLIP_EPS"])
                    v_loss = 0.5 * jnp.maximum((v - tgt) ** 2, (v_clip - tgt) ** 2).mean()
                    ratio = jnp.exp(lp - tr.log_prob)
                    an = (a - a.mean()) / (a.std() + 1e-8)
                    pg = -jnp.minimum(ratio * an, ratio.clip(1 - cfg["CLIP_EPS"], 1 + cfg["CLIP_EPS"]) * an).mean()
                    ent = pi.entropy().mean()
                    return pg + cfg["VF_COEF"] * v_loss - cfg["ENT_COEF"] * ent, (v_loss, pg, ent)

                (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                return ts.apply_gradients(grads=g), (loss, optax.global_norm(g))

            ts, (losses, gnorms) = jax.lax.scan(mb_step, ts, mbs)
            return (ts, rng), (losses, gnorms)

        (ts, rng), (losses, gnorms) = jax.lax.scan(epoch, (ts, rng), None, cfg["UPDATE_EPOCHS"])
        info = traj.info
        done_mask = info["returned_episode"]
        ep_ret = jnp.where(done_mask.sum() > 0,
                           (info["returned_episode_returns"] * done_mask).sum() / jnp.maximum(done_mask.sum(), 1),
                           jnp.nan)
        metrics = dict(
            step=(rs["update"] + 1) * steps_per_update,
            episode_return=ep_ret,
            deficit_active=_is_active(rs["update"] * steps_per_update, rs["deficit_bounds"]).astype(jnp.float32),
            loss=losses.mean(),
            grad_norm=gnorms.mean(),
        )
        new_rs = dict(rs, train_state=ts, env_state=env_state, obs=obs, rng=rng,
                      update=rs["update"] + 1, reward_bank=bank)
        return new_rs, metrics

    def run(runner_state, n_updates: int):
        return jax.lax.scan(_update, runner_state, None, n_updates)

    return cfg, init, run


def make_eval(config: dict, num_envs: int = 64):
    """Deficit-free evaluation: full action set, clean observations, true env reward.

    `evaluate(params, key)` runs `num_envs` episodes (first episode per env, capped at the env's
    max_steps_in_episode) with actions sampled from the *unmasked* policy, and returns per-episode
    returns and lengths. Pass the same `key` to every agent and checkpoint (common random numbers).
    """
    cfg = {**DEFAULT_CONFIG, **config}
    env, env_params = gymnax.make(cfg["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    n_actions = env.action_space(env_params).n
    net = ActorCritic(n_actions, cfg["HIDDEN"], cfg["LAYERNORM"])
    T = int(env_params.max_steps_in_episode) + 1

    def evaluate(params, key):
        k_reset, k_run = jax.random.split(key)
        obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_reset, num_envs), env_params)

        def step(carry, k):
            obs, st, alive, ret, length = carry
            k_a, k_s = jax.random.split(k)
            logits, _ = net.apply(params, obs)
            a = distrax.Categorical(logits=logits).sample(seed=k_a)
            obs, st, r, d, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(k_s, num_envs), st, a, env_params)
            ret = ret + alive * r
            length = length + alive
            alive = alive & ~d
            return (obs, st, alive, ret, length), None

        init = (obs, st, jnp.ones((num_envs,), bool), jnp.zeros((num_envs,)), jnp.zeros((num_envs,), jnp.int32))
        (_, _, alive, ret, length), _ = jax.lax.scan(step, init, jax.random.split(k_run, T))
        return dict(ret=ret, length=length, unfinished=alive)

    return evaluate

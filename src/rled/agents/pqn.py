"""PQN (Gallici et al., ICLR 2025) and a replay-buffer variant (PQN-R) for gymnax/MinAtar, with deficit hooks
and segmentable training (init -> run(n) -> surgery -> run(n)), mirroring rled.agents.ppo.

PQN  (REPLAY=False): LayerNorm Q-network, no target network, no replay buffer. Each update collects
     NUM_STEPS x NUM_ENVS transitions with epsilon-greedy, computes Q(lambda) targets with the pre-update network
     (purejaxql recursion), and trains NUM_EPOCHS x NUM_MINIBATCHES minibatches on that batch, then discards it.
PQN-R (REPLAY=True): same network, exploration, collection and number of gradient steps per update, but the
     transitions are appended to a FIFO replay buffer (BUFFER_SIZE transitions, stored per env so that s' is the
     next slot) and each gradient step samples REPLAY_BATCH transitions uniformly. Targets are 1-step
     r + gamma (1 - d) max_a' Q_theta(s', a') with the *current* network (no target network, as in PQN).
     Replay ratio = grad steps x batch / new transitions per update = 2 by default, equal to PQN's 2 epochs.

Carriers in the runner state (for surgery):
  params    -> runner_state["train_state"].params        opt_state -> runner_state["train_state"].opt_state
  buffer    -> runner_state["buffer"]  (PQN-R only; obs bit-packed uint8)
  exploration schedule and counters are functions of runner_state["update"] (identical across arms at a fork)
  target network: none (PQN and PQN-R both bootstrap from the online network)

Deficit: action_mask with a fixed `allowed_actions` list. While active (per env step for acting, per update for
targets) epsilon-greedy acts only over allowed actions (random draws too) and bootstrap maxima are over allowed
actions, i.e. Q-learning in the restricted MDP. Evaluation (make_eval) uses the full action set, clean
observations and the true reward.
"""
from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

DEFAULT_CONFIG = dict(
    ENV_NAME="SpaceInvaders-MinAtar",
    TOTAL_TIMESTEPS=int(1e7),
    NUM_ENVS=128,
    NUM_STEPS=32,
    NUM_EPOCHS=2,
    NUM_MINIBATCHES=32,
    LR=5e-4,                 # constant by design (purejaxql decays linearly; critical-period sweeps need constant LR)
    MAX_GRAD_NORM=10.0,
    GAMMA=0.99,
    LAMBDA=0.65,
    EPS_START=1.0,
    EPS_FINISH=0.05,
    EPS_DECAY_STEPS=1_000_000,  # purejaxql: 0.1 x 1e7 env steps
    HIDDEN=128,
    LAYERNORM=True,
    REPLAY=False,
    BUFFER_SIZE=100_000,
    REPLAY_BATCH=128,
    SEAM_MASK=False,         # PQN-R: per-slot valid-successor flag; flagged transitions get zero loss weight (Phase 6b)
    DEFICIT="none",
    DEFICIT_START=0,
    DEFICIT_END=0,
    DEFICIT_PARAMS={},
)


class QNet(nn.Module):
    action_dim: int
    hidden: int = 128
    layernorm: bool = True

    @nn.compact
    def __call__(self, x):
        h = x
        for i in range(2):
            h = nn.Dense(self.hidden, name=f"q{i}")(h)
            if self.layernorm:
                h = nn.LayerNorm(name=f"ln{i}")(h)
            h = nn.relu(h)
        return nn.Dense(self.action_dim, name="head")(h)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    q: jnp.ndarray
    info: dict


def masked_max(q, mask):
    return jnp.max(jnp.where(mask, q, -jnp.inf), axis=-1)


def make(config: dict):
    cfg = {**DEFAULT_CONFIG, **config}
    cfg["STEPS_PER_UPDATE"] = spu = cfg["NUM_STEPS"] * cfg["NUM_ENVS"]
    cfg["NUM_UPDATES"] = cfg["TOTAL_TIMESTEPS"] // spu
    cfg["MINIBATCH_SIZE"] = spu // cfg["NUM_MINIBATCHES"]
    cfg["GRAD_STEPS_PER_UPDATE"] = cfg["NUM_EPOCHS"] * cfg["NUM_MINIBATCHES"]
    cfg["BUFFER_LEN"] = cfg["BUFFER_SIZE"] // cfg["NUM_ENVS"]  # slots per env
    cfg["REPLAY_RATIO"] = cfg["GRAD_STEPS_PER_UPDATE"] * (cfg["REPLAY_BATCH"] if cfg["REPLAY"] else cfg["MINIBATCH_SIZE"]) / spu

    env, env_params = gymnax.make(cfg["ENV_NAME"])
    env = LogWrapper(FlattenObservationWrapper(env))
    n_actions = env.action_space(env_params).n
    obs_dim = int(np.prod(env.observation_space(env_params).shape))
    n_packed = (obs_dim + 7) // 8
    net = QNet(n_actions, cfg["HIDDEN"], cfg["LAYERNORM"])
    dp = cfg["DEFICIT_PARAMS"]
    dtype = cfg["DEFICIT"]
    N, L = cfg["NUM_ENVS"], cfg["BUFFER_LEN"]

    def init(rng):
        rng, k_net, k_env = jax.random.split(rng, 3)
        params = net.init(k_net, jnp.zeros((obs_dim,)))
        tx = optax.chain(optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]), optax.radam(cfg["LR"]))
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_env, N), env_params)
        allowed = dp.get("allowed_actions", list(range(n_actions)))
        rs = dict(train_state=ts, env_state=env_state, obs=obs, rng=rng, update=jnp.array(0, jnp.int32),
                  action_mask=jnp.zeros((n_actions,), bool).at[jnp.array(allowed)].set(True),
                  deficit_bounds=jnp.array([cfg["DEFICIT_START"], cfg["DEFICIT_END"]], jnp.int32))
        if cfg["REPLAY"]:
            rs["buffer"] = dict(obs=jnp.zeros((L, N, n_packed), jnp.uint8), action=jnp.zeros((L, N), jnp.uint8),
                                reward=jnp.zeros((L, N), jnp.float32), done=jnp.zeros((L, N), bool),
                                tag=jnp.full((L, N), -1, jnp.int16),  # update index at which each slot was written
                                ptr=jnp.array(0, jnp.int32), size=jnp.array(0, jnp.int32))
            if cfg["SEAM_MASK"]:  # False where slot t's successor (slot t+1, same env) is not its true next obs
                rs["buffer"]["valid_next"] = jnp.ones((L, N), bool)
            # diagnostics only: report the share of the buffer / of sampled batches written in updates [w0, w1)
            rs["tag_window"] = jnp.array([0, 0], jnp.int32)
        return rs

    def _is_active(t, bounds):
        if dtype == "none":
            return jnp.array(False)
        return (t >= bounds[0]) & (t < bounds[1])

    def eps_at(t):
        frac = jnp.clip(t / cfg["EPS_DECAY_STEPS"], 0.0, 1.0)
        return cfg["EPS_START"] + frac * (cfg["EPS_FINISH"] - cfg["EPS_START"])

    def _update(rs, _):
        ts = rs["train_state"]
        t0 = rs["update"] * spu
        full = jnp.ones((n_actions,), bool)
        u_mask = jnp.where(_is_active(t0, rs["deficit_bounds"]) & (dtype == "action_mask"), rs["action_mask"], full)

        def env_step(carry, step_idx):
            env_state, obs, rng = carry
            t = t0 + step_idx * N
            m = jnp.where(_is_active(t, rs["deficit_bounds"]) & (dtype == "action_mask"), rs["action_mask"], full)
            rng, k_r, k_e, k_s = jax.random.split(rng, 4)
            q = net.apply(ts.params, obs)
            greedy = jnp.argmax(jnp.where(m, q, -jnp.inf), -1)
            rand = jax.random.categorical(k_r, jnp.where(m, 0.0, -jnp.inf), shape=(N,))
            explore = jax.random.uniform(k_e, (N,)) < eps_at(t)
            a = jnp.where(explore, rand, greedy)
            obs2, env_state, r, d, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(k_s, N), env_state, a, env_params)
            return (env_state, obs2, rng), Transition(obs, a, r, d, q, info)

        (env_state, obs, rng), traj = jax.lax.scan(env_step, (rs["env_state"], rs["obs"], rs["rng"]), jnp.arange(cfg["NUM_STEPS"]))

        def loss_fn(p, x, a, tgt, w=None):
            q = net.apply(p, x)
            qa = jnp.take_along_axis(q, a[:, None], -1)[:, 0]
            if w is None:
                return 0.5 * jnp.mean((qa - tgt) ** 2), qa.mean()
            return 0.5 * (jnp.sum(w * (qa - tgt) ** 2) / jnp.maximum(jnp.sum(w), 1.0)), qa.mean()

        def sgd(ts, batch):
            (l, qm), g = jax.value_and_grad(loss_fn, has_aux=True)(ts.params, *batch)
            return ts.apply_gradients(grads=g), (l, qm, optax.global_norm(g))

        new_rs = dict(rs)
        extra = {}
        if not cfg["REPLAY"]:
            # Q(lambda) targets, purejaxql recursion, with the pre-update network
            last_q = masked_max(net.apply(ts.params, obs), u_mask) * (1 - traj.done[-1])
            lam_last = traj.reward[-1] + cfg["GAMMA"] * last_q

            def tgt_fn(carry, tr):
                lam, next_q = carry
                boot = tr.reward + cfg["GAMMA"] * (1 - tr.done) * next_q
                lam = boot + cfg["GAMMA"] * cfg["LAMBDA"] * (lam - next_q)
                lam = (1 - tr.done) * lam + tr.done * tr.reward
                return (lam, masked_max(tr.q, u_mask)), lam

            body = jax.tree.map(lambda x: x[:-1], traj)
            # carry next_q for step T-2 is max Q(obs_{T-1}) = traj.q[-1]
            _, tg = jax.lax.scan(tgt_fn, (lam_last, masked_max(traj.q[-1], u_mask)), body, reverse=True)
            targets = jnp.concatenate([tg, lam_last[None]])

            def epoch(carry, _):
                ts, rng = carry
                rng, k = jax.random.split(rng)
                perm = jax.random.permutation(k, spu)
                flat = lambda x: x.reshape((spu,) + x.shape[2:])[perm].reshape((cfg["NUM_MINIBATCHES"], -1) + x.shape[2:])
                ts, out = jax.lax.scan(sgd, ts, (flat(traj.obs), flat(traj.action), flat(targets)))
                return (ts, rng), out

            (ts, rng), (losses, qms, gns) = jax.lax.scan(epoch, (ts, rng), None, cfg["NUM_EPOCHS"])
        else:
            b = rs["buffer"]
            T = cfg["NUM_STEPS"]
            idx = (b["ptr"] + jnp.arange(T)) % L
            pack = lambda o: jnp.packbits(o > 0.5, axis=-1)
            b = dict(obs=b["obs"].at[idx].set(pack(traj.obs)), action=b["action"].at[idx].set(traj.action.astype(jnp.uint8)),
                     reward=b["reward"].at[idx].set(traj.reward), done=b["done"].at[idx].set(traj.done),
                     tag=b["tag"].at[idx].set(rs["update"].astype(jnp.int16)),
                     ptr=(b["ptr"] + T) % L, size=jnp.minimum(b["size"] + T, L),
                     **({"valid_next": rs["buffer"]["valid_next"].at[idx].set(True)} if cfg["SEAM_MASK"] else {}))
            tw = rs["tag_window"]
            in_w = lambda tg: ((tg >= tw[0]) & (tg < tw[1])).astype(jnp.float32)
            unpack = lambda o: jnp.unpackbits(o, axis=-1, count=obs_dim).astype(jnp.float32)
            newest = (b["ptr"] - 1) % L
            # obs for the slot after `newest` is the current env obs (not yet written): patch on read
            cur = pack(obs)

            def replay_step(carry, _):
                ts, rng = carry
                rng, k_t, k_e = jax.random.split(rng, 3)
                j = jax.random.randint(k_t, (cfg["REPLAY_BATCH"],), 0, b["size"])  # 0 = oldest ... size-1 = newest
                ti = (b["ptr"] - b["size"] + j) % L
                ei = jax.random.randint(k_e, (cfg["REPLAY_BATCH"],), 0, N)
                tn = (ti + 1) % L
                o2p = jnp.where((ti == newest)[:, None], cur[ei], b["obs"][tn, ei])
                x, x2 = unpack(b["obs"][ti, ei]), unpack(o2p)
                q2 = masked_max(net.apply(ts.params, x2), u_mask)
                tgt = jax.lax.stop_gradient(b["reward"][ti, ei] + cfg["GAMMA"] * (1 - b["done"][ti, ei]) * q2)
                batch = (x, b["action"][ti, ei].astype(jnp.int32), tgt)
                if cfg["SEAM_MASK"]:
                    batch = batch + (b["valid_next"][ti, ei].astype(jnp.float32),)
                ts, out = sgd(ts, batch)
                return (ts, rng), (*out, in_w(b["tag"][ti, ei]).mean())

            (ts, rng), (losses, qms, gns, sw) = jax.lax.scan(replay_step, (ts, rng), None, cfg["GRAD_STEPS_PER_UPDATE"])
            new_rs["buffer"] = b
            extra = dict(buf_frac_window=in_w(b["tag"]).sum() / jnp.maximum(b["size"] * N, 1), sample_frac_window=sw.mean())

        info = traj.info
        ep_ret = jnp.sum(info["returned_episode_returns"] * info["returned_episode"]) / jnp.maximum(jnp.sum(info["returned_episode"]), 1)
        metrics = dict(step=(rs["update"] + 1) * spu, episode_return=ep_ret, eps=eps_at(t0),
                       deficit_active=_is_active(t0, rs["deficit_bounds"]).astype(jnp.float32),
                       loss=losses.mean(), q_mean=qms.mean(), grad_norm=gns.mean(), **extra)
        new_rs.update(train_state=ts, env_state=env_state, obs=obs, rng=rng, update=rs["update"] + 1)
        return new_rs, metrics

    def run(runner_state, n_updates: int):
        return jax.lax.scan(_update, runner_state, None, n_updates)

    return cfg, init, run


def make_eval(config: dict, num_envs: int = 64, eps: float = 0.0):
    """Deficit-free evaluation: full action set, clean observations, true reward; epsilon-greedy with `eps`
    (0 = greedy, as purejaxql EPS_TEST). First episode per env, capped at max_steps_in_episode. Same key for
    every agent and checkpoint (common random numbers)."""
    cfg = {**DEFAULT_CONFIG, **config}
    env, env_params = gymnax.make(cfg["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    n_actions = env.action_space(env_params).n
    net = QNet(n_actions, cfg["HIDDEN"], cfg["LAYERNORM"])
    T = int(env_params.max_steps_in_episode) + 1

    def evaluate(params, key):
        k_reset, k_run = jax.random.split(key)
        obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_reset, num_envs), env_params)

        def step(carry, k):
            obs, st, alive, ret, length = carry
            k_r, k_e, k_s = jax.random.split(k, 3)
            a = jnp.argmax(net.apply(params, obs), -1)
            a = jnp.where(jax.random.uniform(k_e, (num_envs,)) < eps, jax.random.randint(k_r, (num_envs,), 0, n_actions), a)
            obs, st, r, d, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(k_s, num_envs), st, a, env_params)
            ret = ret + alive * r
            length = length + alive
            alive = alive & ~d
            return (obs, st, alive, ret, length), None

        init = (obs, st, jnp.ones((num_envs,), bool), jnp.zeros((num_envs,)), jnp.zeros((num_envs,), jnp.int32))
        (_, _, alive, ret, length), _ = jax.lax.scan(step, init, jax.random.split(k_run, T))
        return dict(ret=ret, length=length, unfinished=alive)

    return evaluate


def make_collector(config: dict, eps: float = 0.05, steps: int | None = None):
    """Replay-buffer replacement data (Phase 6 purge): a FROZEN policy collects BUFFER_LEN steps in NUM_ENVS fresh,
    unrestricted envs (full action set, epsilon-greedy with `eps`), returning arrays in the PQN-R buffer schema
    (packed obs, uint8 action, reward, done), time-ordered along axis 0. The training env wrappers are the same as in
    `make`; the collection envs and RNG are separate from any agent's env state and RNG."""
    cfg = {**DEFAULT_CONFIG, **config}
    env, env_params = gymnax.make(cfg["ENV_NAME"])
    env = LogWrapper(FlattenObservationWrapper(env))
    n_actions = env.action_space(env_params).n
    net = QNet(n_actions, cfg["HIDDEN"], cfg["LAYERNORM"])
    N = cfg["NUM_ENVS"]
    L = cfg["BUFFER_SIZE"] // N if steps is None else steps  # the first k steps do not depend on `steps`
    pack = lambda o: jnp.packbits(o > 0.5, axis=-1)

    def collect(params, key):
        k_reset, k_run = jax.random.split(key)
        obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_reset, N), env_params)

        def step(carry, _):
            st, obs, rng = carry
            rng, k_r, k_e, k_s = jax.random.split(rng, 4)
            q = net.apply(params, obs)
            rand = jax.random.randint(k_r, (N,), 0, n_actions)
            a = jnp.where(jax.random.uniform(k_e, (N,)) < eps, rand, jnp.argmax(q, -1))
            obs2, st, r, d, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(k_s, N), st, a, env_params)
            ep = (info["returned_episode_returns"] * info["returned_episode"]).sum(), info["returned_episode"].sum()
            return (st, obs2, rng), (pack(obs), a.astype(jnp.uint8), r, d, ep)

        _, (o, a, r, d, (ret_sum, n_ep)) = jax.lax.scan(step, (st, obs, k_run), None, L)
        stats = dict(episodes=n_ep.sum(), mean_return=ret_sum.sum() / jnp.maximum(n_ep.sum(), 1))
        return dict(obs=o, action=a, reward=r, done=d), stats

    return collect

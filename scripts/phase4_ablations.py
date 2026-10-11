"""Phase 4 causal ablations at removal of the late movement restriction (SpaceInvaders, PPO).

One run produces all three ablations from the same forks (same seeds, same clean eval protocol):

Forks (updates; window [w0, w1) = mask):
  shared history 0 -> w0 (no deficit); observation BANK collected with the onset policy at w0
  C : w0 -> w1, no mask                         (clean)
  D : w0 -> w1, movement mask                   (damaged)
  F : w0 -> w1, movement mask + frozen actor hidden layers a0, a1    (drift intervention 1)
  A : w0 -> w1, movement mask + beta * KL(pi_onset || pi_theta)      (drift intervention 2)
  C and D must reproduce the Phase 3 saved states bit-for-bit (checked if --saved-states is given).

Recovery arms (all w1 -> w1 + R, no deficit, env state / obs / RNG from C, deficit-free eval every E):
  Ablation 1 (actor reset vs transplant), learner = D except:
    D_online            : unchanged
    D_actor_reset       : fresh actor weights (new init key per seed); critic + optimizer kept
    D_actor_transplant  : matched clean actor (positive control)
    D_actor_opt_reset   : fresh actor + fresh Adam state (mu = nu = 0, count = 0; the count is shared
                          between actor and critic, so the full Adam state is reset)
  Ablation 2 (actor-experience loop), learner = D; for the first K updates rollouts come from:
    D_online            : the learner itself (on-policy)
    D_beh_clean         : the matched clean actor (frozen at w1)
    D_beh_frozen_self   : D's own actor frozen at w1 (i.e. a fixed pre-intervention dataset distribution:
                          K x 8192 transitions, each batch used for one PPO update with stored log-probs)
    D_beh_xseed_clean   : a clean actor from a different seed (good, but unrelated to the learner)
    D_beh_tempered      : D's own actor with logits / T_s, T_s chosen per seed so that
                          KL(pi_b || pi_D) on the bank matches KL(pi_C || pi_D) (mismatch without quality)
  Ablation 3 (drift mechanism): C_cont, F_cont, A_cont continuations (plus D_online).
Drift metrics vs the onset policy on the fixed bank, at every checkpoint of every fork and arm:
  KL(pi_onset || pi_t), per-action probabilities, entropy, mean value, representation drift
  (1 - linear CKA vs onset) for a0 and a1, and the dead-unit fraction of a1.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import distrax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.wrappers.purerl import FlattenObservationWrapper

from rled.agents.ppo import ACTOR_KEYS, ActorCritic, make, make_eval
from rled.provenance import build_manifest, write_manifest
from rled.surgery import compose_ppo

warnings.filterwarnings("ignore")

METRICS = ("episode_return", "behaviour_kl", "ratio_mean", "ratio_absdev", "clip_frac", "approx_kl", "entropy",
           "anchor_kl")


# ---------------------------------------------------------------- drift metrics on a fixed bank
def actor_fwd(pp, x):
    x = x.astype(jnp.float32)
    h0 = jax.nn.relu(x @ pp["a0"]["kernel"] + pp["a0"]["bias"])
    h1 = jax.nn.relu(h0 @ pp["a1"]["kernel"] + pp["a1"]["bias"])
    return h0, h1, h1 @ pp["Dense_0"]["kernel"] + pp["Dense_0"]["bias"]


def value_fwd(pp, x):
    x = x.astype(jnp.float32)
    c0 = jax.nn.relu(x @ pp["c0"]["kernel"] + pp["c0"]["bias"])
    c1 = jax.nn.relu(c0 @ pp["c1"]["kernel"] + pp["c1"]["bias"])
    return (c1 @ pp["Dense_1"]["kernel"] + pp["Dense_1"]["bias"])[:, 0]


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = jnp.sum((X.T @ Y) ** 2)
    return hsic / (jnp.sqrt(jnp.sum((X.T @ X) ** 2)) * jnp.sqrt(jnp.sum((Y.T @ Y) ** 2)) + 1e-12)


def drift(pp, pp_on, bank):
    h0, h1, lg = actor_fwd(pp, bank)
    o0, o1, lo = actor_fwd(pp_on, bank)
    lp, lq = jax.nn.log_softmax(lg), jax.nn.log_softmax(lo)
    return dict(kl_from_onset=jnp.sum(jnp.exp(lq) * (lq - lp), -1).mean(),
                probs=jnp.exp(lp).mean(0), entropy=-jnp.sum(jnp.exp(lp) * lp, -1).mean(),
                value=value_fwd(pp, bank).mean(), repr_drift_a0=1 - linear_cka(o0, h0),
                repr_drift_a1=1 - linear_cka(o1, h1), dead_a1=(h1.max(0) <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--allowed", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--window", type=int, nargs=2, default=[100, 150])
    ap.add_argument("--recovery", type=int, default=150)
    ap.add_argument("--behaviour-k", type=int, default=10)
    ap.add_argument("--anchor-coef", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--bank-seed", type=int, default=20_000)
    ap.add_argument("--reset-seed-offset", type=int, default=777)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arms (default: all)")
    ap.add_argument("--saved-states", default=None, help="Phase 3 states_w1.pkl for the bit-identity check")
    ap.add_argument("--tag", default="phase4-move-late")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    E, K = a.eval_every, a.behaviour_k
    w0, w1 = a.window
    spu = 64 * 128
    n = len(a.seeds)
    base = dict(ENV_NAME=a.env, HIDDEN=a.hidden, DEFICIT="action_mask", TOTAL_TIMESTEPS=(w1 + a.recovery) * spu,
                DEFICIT_PARAMS={"allowed_actions": a.allowed})
    all_arms = ["C_cont", "D_online", "D_actor_reset", "D_actor_transplant", "D_actor_opt_reset", "D_beh_clean",
                "D_beh_frozen_self", "D_beh_xseed_clean", "D_beh_tempered", "F_cont", "A_cont"]
    arms = a.arms or all_arms
    protocol = dict(window=a.window, recovery=a.recovery, behaviour_k=K, anchor_coef=a.anchor_coef, eval_every=E,
                    eval_envs=a.eval_envs, eval_seed=a.eval_seed, bank_seed=a.bank_seed,
                    reset_seed_offset=a.reset_seed_offset, steps_per_update=spu, allowed=a.allowed, arms=arms,
                    future_randomness="env state, obs, RNG, update counter from clean run C",
                    drift_kl_direction="KL(pi_onset || pi_t), unmasked policies, on the onset bank")
    manifest = build_manifest(f"p4-{a.tag}-ppo-{a.env}", {**base, "protocol": protocol}, a.seeds,
                              allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)
    out_dir = Path("results") / mpath.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_path = Path("results/summary") / f"{mpath.stem}.json"

    cfg, init, run = make(base)
    _, _, run_f = make({**base, "FREEZE_ACTOR_REPR": True})
    _, _, run_a = make({**base, "ANCHOR_COEF": a.anchor_coef})
    init_v = jax.jit(jax.vmap(init))
    run_v = jax.jit(jax.vmap(lambda rs: run(rs, E)))
    run_fv = jax.jit(jax.vmap(lambda rs: run_f(rs, E)))
    run_av = jax.jit(jax.vmap(lambda rs: run_a(rs, E)))
    eval_v = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs), in_axes=(0, None)))
    drift_v = jax.jit(jax.vmap(drift))
    eval_key = jax.random.PRNGKey(a.eval_seed)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    bounds = lambda b0, b1: jnp.tile(jnp.array([b0 * spu, b1 * spu], jnp.int32), (n, 1))
    pp = lambda rs: rs["train_state"].params["params"]
    actor_of = lambda rs: {k: v for k, v in pp(rs).items() if k in ACTOR_KEYS}
    t0 = time.time()

    # bank collection (onset policy, unmasked, 16 envs x 1000 steps, every 8th step kept)
    env, envp = gymnax.make(a.env)
    env = FlattenObservationWrapper(env)
    net = ActorCritic(env.action_space(envp).n, a.hidden, False)

    def collect(params, key):
        k0, k1 = jax.random.split(key)
        obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k0, 16), envp)

        def inner(c, k):
            obs, st = c
            ka, ks = jax.random.split(k)
            a_ = distrax.Categorical(logits=net.apply(params, obs)[0]).sample(seed=ka)
            obs, st, _, _, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(ks, 16), st, a_, envp)
            return (obs, st), None

        def outer(c, k):
            c, _ = jax.lax.scan(inner, c, jax.random.split(k, 8))
            return c, c[0]

        _, hist = jax.lax.scan(outer, (obs, st), jax.random.split(k1, 125))
        return hist.reshape(-1, hist.shape[-1])

    def ev(rs):
        return np.asarray(jax.device_get(eval_v(rs["train_state"].params, eval_key)["ret"])).mean(1)

    def dr(rs):
        d = jax.device_get(drift_v(pp(rs), onset_pp, bank))
        return {k: np.asarray(v) for k, v in d.items()}

    # 1. shared history
    rs = init_v(keys)
    rs["deficit_bounds"] = bounds(0, 0)
    for _ in range(w0 // E):
        rs, _m = run_v(rs)
    onset_pp = pp(rs)
    bank = jax.jit(jax.vmap(collect, in_axes=(0, None)))(rs["train_state"].params, jax.random.PRNGKey(a.bank_seed))
    np.savez_compressed(out_dir / "bank.npz", bank=np.asarray(bank))
    # check the manual forward matches the network
    lg_net = jax.vmap(lambda p, x: net.apply(p, x)[0])(rs["train_state"].params, bank[:, :8])
    lg_man = jax.vmap(lambda p, x: actor_fwd(p, x)[2])(onset_pp, bank[:, :8])
    assert np.allclose(lg_net, lg_man, atol=1e-5), "manual actor forward does not match network"
    print(f"shared history + bank {tuple(bank.shape)} done, elapsed={time.time()-t0:.0f}s", flush=True)

    # 2. forks
    rs = dict(rs, anchor_actor={k: v for k, v in onset_pp.items() if k in ACTOR_KEYS})
    forks = {"C": (dict(rs, deficit_bounds=bounds(0, 0)), run_v), "D": (dict(rs, deficit_bounds=bounds(w0, w1)), run_v),
             "F": (dict(rs, deficit_bounds=bounds(w0, w1)), run_fv), "A": (dict(rs, deficit_bounds=bounds(w0, w1)), run_av)}
    fork_log, fork_state = {}, {}
    for name, (r, fn) in forks.items():
        log = dict(eval=[ev(r)], drift=[dr(r)], train=[])
        for _ in range((w1 - w0) // E):
            r, m = fn(r)
            log["eval"].append(ev(r))
            log["drift"].append(dr(r))
            log["train"].append({k: np.asarray(m[k]) for k in METRICS})
        fork_state[name] = dict(r, deficit_bounds=bounds(0, 0))
        fork_log[name] = log
        print(f"fork {name}: eval at onset {np.round(log['eval'][0],1)} removal {np.round(log['eval'][-1],1)} "
              f"KL_onset at removal {np.round(log['drift'][-1]['kl_from_onset'],3)} elapsed={time.time()-t0:.0f}s",
              flush=True)
    C, D, F, A = (fork_state[k] for k in "CDFA")

    check = {}
    if a.saved_states:
        saved = pickle.load(open(a.saved_states, "rb"))
        same = lambda x, y: all(np.array_equal(np.asarray(p), np.asarray(q))
                                for p, q in zip(jax.tree.leaves(x), jax.tree.leaves(y)))
        check = dict(C_params_identical=same(C["train_state"].params, saved["C"]["params"]),
                     D_params_identical=same(D["train_state"].params, saved["D"]["params"]),
                     C_opt_identical=same(C["train_state"].opt_state, saved["C"]["opt_state"]),
                     D_opt_identical=same(D["train_state"].opt_state, saved["D"]["opt_state"]))
        print("bit-identity vs Phase 3 saved states:", check, flush=True)

    # tempered behaviour: match KL(pi_b || pi_D) to KL(pi_C || pi_D) on the bank, per seed
    lD = np.asarray(jax.vmap(lambda p, x: actor_fwd(p, x)[2])(pp(D), bank))
    lC = np.asarray(jax.vmap(lambda p, x: actor_fwd(p, x)[2])(pp(C), bank))
    lsm = lambda z: z - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)) - z.max(-1, keepdims=True)
    kl = lambda lp, lq: float(np.mean(np.sum(np.exp(lp) * (lp - lq), -1)))
    temps, kl_target, kl_tempered = [], [], []
    for i in range(n):
        tgt = kl(lsm(lC[i]), lsm(lD[i]))
        f = lambda T: kl(lsm(lD[i] / T), lsm(lD[i]))
        lo, hi = 1.0, 1e3
        if f(hi) < tgt:
            T = hi
        else:
            for _ in range(60):
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if f(mid) < tgt else (lo, mid)
            T = np.sqrt(lo * hi)
        temps.append(float(T)); kl_target.append(tgt); kl_tempered.append(f(T))
    print(f"tempered behaviour: T={np.round(temps,2)} KL target={np.round(kl_target,3)} "
          f"achieved={np.round(kl_tempered,3)}", flush=True)

    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds, bit_identity_check=check,
                   tempered=dict(T=temps, kl_target_clean_vs_D=kl_target, kl_achieved=kl_tempered),
                   fork_updates=list(range(w0, w1 + 1, E)), eval_updates=list(range(w1, w1 + a.recovery + 1, E)),
                   forks={}, arms={})
    tolist = lambda x: np.round(np.asarray(x, float), 4).tolist()
    for name, log in fork_log.items():
        summary["forks"][name] = dict(
            eval_return_mean=tolist(np.stack(log["eval"], 1)),
            drift={k: tolist(np.stack([d[k] for d in log["drift"]], 1)) for k in log["drift"][0]},
            train={k: tolist(np.concatenate([t[k] for t in log["train"]], 1)) for k in METRICS})

    # 3. recovery arms
    fresh = init_v(jnp.stack([jax.random.PRNGKey(s + a.reset_seed_offset) for s in a.seeds]))
    fresh_actor = actor_of(fresh)
    base_D = compose_ppo(C, D, "DDD")

    def with_actor(r, actor):
        p = {**pp(r), **actor}
        return dict(r, train_state=r["train_state"].replace(params={"params": p}))

    def with_behaviour(r, actor, temp=None):
        out = dict(r, behaviour_actor=actor, behaviour_until=jnp.full((n,), w1 + K, jnp.int32))
        if temp is not None:
            out["behaviour_temp"] = jnp.asarray(temp, jnp.float32)
        return out

    builders = {
        "C_cont": lambda: compose_ppo(C, C, "CCC"),
        "D_online": lambda: base_D,
        "D_actor_reset": lambda: with_actor(base_D, fresh_actor),
        "D_actor_transplant": lambda: compose_ppo(C, D, "CDD"),
        "D_actor_opt_reset": lambda: (lambda r: dict(r, train_state=r["train_state"].replace(
            opt_state=fresh["train_state"].opt_state)))(with_actor(base_D, fresh_actor)),
        "D_beh_clean": lambda: with_behaviour(base_D, actor_of(C)),
        "D_beh_frozen_self": lambda: with_behaviour(base_D, actor_of(D)),
        "D_beh_xseed_clean": lambda: with_behaviour(base_D, jax.tree.map(lambda x: jnp.roll(x, 1, axis=0), actor_of(C))),
        "D_beh_tempered": lambda: with_behaviour(base_D, actor_of(D), temps),
        "F_cont": lambda: compose_ppo(C, F, "DDD"),
        "A_cont": lambda: compose_ppo(C, A, "DDD"),
    }
    for arm in arms:
        r = builders[arm]()
        evals, drifts, train = [ev(r)], [dr(r)], []
        for _ in range(a.recovery // E):
            r, m = run_v(r)
            evals.append(ev(r))
            drifts.append(dr(r))
            train.append({k: np.asarray(m[k]) for k in METRICS})
        summary["arms"][arm] = dict(
            eval_return_mean=tolist(np.stack(evals, 1)),
            drift={k: tolist(np.stack([d[k] for d in drifts], 1)) for k in drifts[0]},
            train={k: tolist(np.concatenate([t[k] for t in train], 1)) for k in METRICS})
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary))
        e = np.stack(evals, 1)
        bk = np.concatenate([t["behaviour_kl"] for t in train], 1)[:, :K].mean()
        cf = np.concatenate([t["clip_frac"] for t in train], 1)[:, :K].mean()
        print(f"arm {arm:>18}: eval +0 {e[:,0].mean():5.1f} +50 {e[:,min(5, e.shape[1]-1)].mean():5.1f} +{a.recovery} {e[:,-1].mean():5.1f} "
              f"AUC {e[:,1:].mean():5.1f} | first-{K} behaviour KL {bk:.3f} clip {cf:.3f} elapsed={time.time()-t0:.0f}s",
              flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

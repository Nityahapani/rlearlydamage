"""Point-of-no-return duration sweep (PPO, Phase 7).

Question: as a late action restriction gets longer, when does the damage stop being recoverable
within a fixed budget, and which state variable, measured at removal, predicts that?

Design (all windows start at the same onset w0, so they are NESTED: a single masked trajectory
passes through the exact removal state of every duration, and every clean reference is a single
clean trajectory):
  shared history 0 -> w0 (no deficit); probe BANK collected with the onset policy at w0
  C : w0 -> w0 + max(d) + R, no mask            (clean reference trajectory)
  D : w0 -> w0 + max(d), restriction mask        (damaged trajectory; snapshot at w0 + d for each d)
  for each duration d (removal at w1 = w0 + d), recovery arms run w1 -> w1 + R, no deficit,
  with env state / obs / RNG / update counter from C at w1 (future randomness matched, as in Phase 4):
    D_online     = compose_ppo(C_w1, D_w1, "DDD")   damaged agent's own recovery
    D_transplant = compose_ppo(C_w1, D_w1, "CDD")   clean actor (from C at w1) + damaged critic and Adam state
Outcome references (computed in analysis, not here):
  delay-adjusted (pause-equivalent):  D_online(w1 + k) - C(w0 + k)   (< 0 = damage beyond lost time)
  raw (clean continuation):           D_online(w1 + k) - C(w1 + k)
Predictor probes (rled.probes, candidate set fixed before any Phase 7 data) are computed online at every
eval checkpoint for C and D through the mask, and for every recovery arm. Params of C and D are saved
only at the removal points (plus onset and the bank) for later offline analysis.

Deficit-free evaluation (full action set, true reward, fixed eval key) every `eval_every` updates.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import distrax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.wrappers.purerl import FlattenObservationWrapper

from rled.agents.ppo import ActorCritic, make, make_eval
from rled.provenance import build_manifest, write_manifest
from rled.probes import PROBE_KEYS, probe
from rled.surgery import compose_ppo

warnings.filterwarnings("ignore")


def flat_params(pp):
    """{'a0/kernel': array[n_seeds, ...], ...} from a vmapped params dict."""
    return {f"{k}/{kk}": np.asarray(v) for k, sub in pp.items() for kk, v in sub.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--allowed", type=int, nargs="+", default=[0, 3], help="[0,3] movement restriction; [0,1,2] fire")
    ap.add_argument("--onset", type=int, default=100)
    ap.add_argument("--durations", type=int, nargs="+", default=[50, 100, 200, 300])
    ap.add_argument("--recovery", type=int, default=250)
    ap.add_argument("--arms", nargs="+", default=["D_online", "D_transplant"])
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--bank-seed", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--tag", default="ponr-pilot-move")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    E, w0, R = a.eval_every, a.onset, a.recovery
    durs = sorted(a.durations)
    dmax = durs[-1]
    assert all(d % E == 0 for d in durs) and R % E == 0 and w0 % E == 0
    spu = 64 * 128
    n = len(a.seeds)
    base = dict(ENV_NAME=a.env, HIDDEN=a.hidden, DEFICIT="action_mask",
                TOTAL_TIMESTEPS=(w0 + dmax + R) * spu, DEFICIT_PARAMS={"allowed_actions": a.allowed})
    protocol = dict(onset=w0, durations=durs, recovery=R, arms=a.arms, eval_every=E, probes=list(PROBE_KEYS),
                    eval_envs=a.eval_envs, eval_seed=a.eval_seed, bank_seed=a.bank_seed, steps_per_update=spu,
                    allowed=a.allowed, nested_windows=True,
                    future_randomness="env state, obs, RNG, update counter from clean run C at removal",
                    transplant="actor (a0, a1, Dense_0) from C at removal; critic + Adam state from D")
    manifest = build_manifest(f"p7-{a.tag}-ppo-{a.env}", {**base, "protocol": protocol}, a.seeds,
                              allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)
    out_dir = Path("results") / mpath.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_path = Path("results/summary") / f"{mpath.stem}.json"

    cfg, init, run = make(base)
    init_v = jax.jit(jax.vmap(init))
    run_v = jax.jit(jax.vmap(lambda rs: run(rs, E)))
    eval_v = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs), in_axes=(0, None)))
    eval_key = jax.random.PRNGKey(a.eval_seed)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    bounds = lambda b0, b1: jnp.tile(jnp.array([b0 * spu, b1 * spu], jnp.int32), (n, 1))
    pp = lambda rs: rs["train_state"].params["params"]
    t0 = time.time()

    env, envp = gymnax.make(a.env)
    env = FlattenObservationWrapper(env)
    net = ActorCritic(env.action_space(envp).n, a.hidden, False)

    def collect(params, key):  # identical to Phase 4: 16 envs x 1000 steps, every 8th kept -> 2000 obs
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

    n_act = env.action_space(envp).n
    masked = tuple(i for i in range(n_act) if i not in a.allowed)
    probe_v = jax.jit(jax.vmap(lambda p, q, b: probe(p, q, b, masked)))

    def pr(rs):
        d = jax.device_get(probe_v(pp(rs), onset_pp, bank))
        return {k: np.asarray(d[k]) for k in PROBE_KEYS}

    def stack_probes(lst):
        return {k: tolist(np.stack([x[k] for x in lst], 1)) for k in PROBE_KEYS}

    tolist = lambda x: np.round(np.asarray(x, float), 4).tolist()
    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds, onset=w0, durations=durs, recovery=R)

    def save():
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary))

    # 1. shared history + bank
    rs = init_v(keys)
    rs["deficit_bounds"] = bounds(0, 0)
    for _ in range(w0 // E):
        rs, _m = run_v(rs)
    onset = rs
    onset_pp = pp(onset)
    bank = jax.jit(jax.vmap(collect, in_axes=(0, None)))(rs["train_state"].params, jax.random.PRNGKey(a.bank_seed))
    np.savez_compressed(out_dir / "bank.npz", bank=np.asarray(bank))
    np.savez_compressed(out_dir / f"onset_u{w0}.npz", **flat_params(pp(onset)))
    print(f"shared history + bank {tuple(bank.shape)} done, elapsed={time.time()-t0:.0f}s", flush=True)

    removal_states = {"C": {}, "D": {}}

    def trajectory(name, r, n_updates, keep_at):
        evals, upd, probes = [ev(r)], [w0], [pr(r)]
        for i in range(n_updates // E):
            r, _m = run_v(r)
            u = w0 + (i + 1) * E
            evals.append(ev(r))
            upd.append(u)
            if u - w0 <= dmax:
                probes.append(pr(r))
            if u - w0 in keep_at:
                removal_states[name][u - w0] = dict(r, deficit_bounds=bounds(0, 0))
                np.savez_compressed(out_dir / f"{name}_u{u}.npz", **flat_params(pp(r)))
        return np.stack(evals, 1), upd, stack_probes(probes)

    # 2. clean reference trajectory C (also provides env/RNG for every removal point)
    ce, cu, cp = trajectory("C", dict(onset, deficit_bounds=bounds(0, 0)), dmax + R, set(durs))
    summary["C"] = dict(updates=cu, eval_return_mean=tolist(ce), probe_updates=cu[:dmax // E + 1], probes=cp)
    save()
    print(f"C trajectory done: eval at onset {np.round(ce[:,0],1)}; at w0+d "
          f"{ {d: round(float(ce[:, d // E].mean()), 1) for d in durs} } elapsed={time.time()-t0:.0f}s", flush=True)

    # 3. damaged trajectory D (one nested mask w0 -> w0 + dmax)
    de, du, dp_ = trajectory("D", dict(onset, deficit_bounds=bounds(w0, w0 + dmax)), dmax, set(durs))
    summary["D_mask"] = dict(updates=du, eval_return_mean=tolist(de), probes=dp_)
    save()
    print(f"D trajectory done: unmasked eval at w0+d { {d: round(float(de[:, d // E].mean()), 1) for d in durs} } "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    print("removal evals per seed:", {d: dict(D=np.round(de[:, d // E], 4).tolist(), C=np.round(ce[:, d // E], 4).tolist())
                                      for d in durs}, flush=True)

    # 4. recovery arms per duration
    summary["arms"] = {}
    for d in durs:
        C1, D1 = removal_states["C"][d], removal_states["D"][d]
        builders = {"D_online": lambda: compose_ppo(C1, D1, "DDD"), "D_transplant": lambda: compose_ppo(C1, D1, "CDD")}
        summary["arms"][str(d)] = {}
        for arm in a.arms:
            r = builders[arm]()
            evals, probes = [ev(r)], [pr(r)]
            for _ in range(R // E):
                r, _m = run_v(r)
                evals.append(ev(r))
                probes.append(pr(r))
            e = np.stack(evals, 1)
            summary["arms"][str(d)][arm] = dict(updates=list(range(w0 + d, w0 + d + R + 1, E)), eval_return_mean=tolist(e),
                                                probes=stack_probes(probes))
            save()
            L = min(5, R // E)
            ref_delay = ce[:, 1:R // E + 1]                     # C(w0 + k), k = E..R
            ref_raw = ce[:, d // E + 1:d // E + R // E + 1]     # C(w0 + d + k)
            print(f"d={d:3d} {arm:>13}: +0 {e[:,0].mean():5.1f} +{R} {e[:,-1].mean():5.1f} AUC {e[:,1:].mean():5.1f} | "
                  f"delay-adj AUC {(e[:,1:]-ref_delay).mean():+5.1f} late {(e[:,-L:]-ref_delay[:,-L:]).mean():+5.1f} | "
                  f"raw AUC {(e[:,1:]-ref_raw).mean():+5.1f} elapsed={time.time()-t0:.0f}s", flush=True)
    if a.tag.startswith("p13-D-"):
        # Measurement archival only: no fitting, model comparison or primary interim analysis.
        # Preserve frozen feature definitions in a small versionable JSON even if large checkpoints are lost.
        jax.clear_caches()
        from p13_prediction import extract
        summary["prospective_features"] = extract([summ_path])
        summary["prospective_feature_schema"] = "prereg/p13_frozen_predictors.json"
        save()
        print("Prospective feature rows archived (values withheld)", flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

"""State surgery on PPO at deficit removal: which carrier holds the late-mask damage?

Pipeline (per seed, all seeds vmapped):
  1. Train a shared history to update `w0` (no deficit).
  2. Fork: clean run C trains w0 -> w1 with no deficit; damaged run D trains w0 -> w1 with the fixed
     action mask. Sanity check: eval at w1 must reproduce the recovery-test control / masked run exactly.
  3. Save both runner states at w1 (results/<stem>/states_w1.pkl, git-ignored).
  4. Surgery arms, all continuing w1 -> w1 + R with no deficit and the same future randomness
     (env state, obs, RNG taken from C):
       2^3 factorial over carriers (actor weights, critic weights, optimizer state); code "DCC" =
       damaged actor, clean critic, clean optimizer. CCC is the control's own continuation, DDD the
       damaged run's own continuation (placed in C's env state).
       Data arms (on-policy "data" = behaviour policy): CCC+bD collects its first K post-surgery
       rollouts with D's frozen actor; DDD+bC with C's frozen actor (importance-weighted PPO updates).
  5. Evaluation every E updates with the deficit-free protocol of recovery_test.py (same eval key).

Usage: python scripts/surgery_ppo.py [--seeds 0 1 2 3] [--recovery 150]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rled.agents.ppo import ACTOR_KEYS, make, make_eval
from rled.provenance import build_manifest, write_manifest
from rled.surgery import PPO_CARRIERS, all_codes, compose_ppo

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--allowed", type=int, nargs="+", default=[0, 3], help="movement restriction: noop, fire")
    ap.add_argument("--window", type=int, nargs=2, default=[100, 150], help="[w0, w1) in updates")
    ap.add_argument("--recovery", type=int, default=150, help="updates after surgery")
    ap.add_argument("--behaviour-k", type=int, default=10, help="updates of behaviour swap in data arms")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--reference", default=None, help="recovery-test summary to check reproduction against")
    ap.add_argument("--tag", default="surgery-move-late")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of arms (default: 2^3 factorial + CCC+bD, DDD+bC); extra: CCC+bDr, DDD+bCr (ratio-neutral)")
    ap.add_argument("--deficit", default="action_mask", choices=["action_mask", "reward_flip"])
    ap.add_argument("--flip-prob", type=float, default=1.0, help="reward_flip: probability a reward is negated")
    ap.add_argument("--clean-source", default="fork", choices=["fork", "onset"],
                    help="source of 'C' components: the clean fork at w1 (default) or the shared onset state at w0 "
                         "(pause counterfactual; env/obs/RNG/counter still from the clean fork at w1)")
    ap.add_argument("--eval-fork", action="store_true", help="also evaluate C and D every E updates during [w0, w1]")
    a = ap.parse_args()

    E = a.eval_every
    w0, w1 = a.window
    assert w0 % E == 0 and w1 % E == 0 and a.recovery % E == 0
    spu = 64 * 128
    total_updates = w1 + a.recovery
    dparams = {"allowed_actions": a.allowed} if a.deficit == "action_mask" else {"prob": a.flip_prob}
    base = dict(ENV_NAME=a.env, HIDDEN=a.hidden, DEFICIT=a.deficit, TOTAL_TIMESTEPS=total_updates * spu,
                DEFICIT_PARAMS=dparams)
    arms = a.arms if a.arms else all_codes(PPO_CARRIERS) + ["CCC+bD", "DDD+bC"]
    protocol = dict(window=a.window, recovery=a.recovery, behaviour_k=a.behaviour_k, eval_every=E,
                    eval_envs=a.eval_envs, eval_seed=a.eval_seed, steps_per_update=spu, carriers=list(PPO_CARRIERS),
                    arms=arms, allowed=a.allowed, deficit=a.deficit, deficit_params=dparams,
                    future_randomness="env state, obs, RNG, update counter from clean run C",
                    clean_source=a.clean_source)
    name = f"p3-{a.tag}-ppo-{a.env}"
    manifest = build_manifest(name, {**base, "protocol": protocol}, a.seeds, allow_dirty=a.allow_dirty)
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
    n = len(a.seeds)
    bounds = lambda b0, b1: jnp.tile(jnp.array([b0 * spu, b1 * spu], jnp.int32), (n, 1))
    ev = lambda rs: np.asarray(jax.device_get(eval_v(rs["train_state"].params, eval_key)["ret"])).mean(1)
    t0 = time.time()

    # 1-2. shared history and fork
    rs = init_v(keys)
    rs["deficit_bounds"] = bounds(0, 0)
    for _ in range(w0 // E):
        rs, _m = run_v(rs)
    rs_w0 = rs  # shared onset state (pause counterfactual for --clean-source onset)
    rs_c, rs_d = rs, dict(rs, deficit_bounds=bounds(w0, w1))
    fork_eval = dict(updates=[w0], C=[ev(rs_c).tolist()], D=[ev(rs_d).tolist()]) if a.eval_fork else None
    for i in range((w1 - w0) // E):
        rs_c, _m = run_v(rs_c)
        rs_d, _m = run_v(rs_d)
        if a.eval_fork and i < (w1 - w0) // E - 1:
            fork_eval["updates"].append(w0 + (i + 1) * E)
            fork_eval["C"].append(ev(rs_c).tolist())
            fork_eval["D"].append(ev(rs_d).tolist())
    rs_d = dict(rs_d, deficit_bounds=bounds(0, 0))  # deficit removed from here on
    e_c, e_d = ev(rs_c), ev(rs_d)
    print(f"fork done: eval at removal C={np.round(e_c,2)} D={np.round(e_d,2)} elapsed={time.time()-t0:.0f}s", flush=True)
    check = {}
    if a.reference:
        ref = json.loads(Path(a.reference).read_text())
        i = ref["eval_updates"].index(w1)
        rc = np.array(ref["configs"]["control"]["eval_return_mean"])[:, i]
        rd = np.array(ref["configs"]["move_late"]["eval_return_mean"])[:, i]
        check = dict(control_ref=rc.tolist(), damaged_ref=rd.tolist(), control_now=np.round(e_c, 3).tolist(),
                     damaged_now=np.round(e_d, 3).tolist(),
                     max_abs_diff=float(max(np.abs(rc - e_c).max(), np.abs(rd - e_d).max())))
        print("reproduction check vs recovery test:", check, flush=True)

    # 3. save states at removal
    with open(out_dir / "states_w1.pkl", "wb") as f:
        arrays = lambda r: dict({k: v for k, v in r.items() if k != "train_state"},
                                params=r["train_state"].params, opt_state=r["train_state"].opt_state,
                                step=r["train_state"].step)
        pickle.dump(jax.device_get(dict(C=arrays(rs_c), D=arrays(rs_d))), f)

    # 4-5. surgery arms
    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds, reproduction_check=check,
                   eval_updates=list(range(w1, total_updates + 1, E)), arms={}, fork_eval=fork_eval)
    actor_of = lambda r: {k: v for k, v in r["train_state"].params["params"].items() if k in ACTOR_KEYS}
    rs_ref = rs_c
    if a.clean_source == "onset":
        assert not any("+b" in x for x in arms), "data arms are defined only for the clean fork"
        rs_ref = dict(rs_c, train_state=rs_c["train_state"].replace(params=rs_w0["train_state"].params,
                                                                    opt_state=rs_w0["train_state"].opt_state))
        print(f"onset reference: eval of onset params {np.round(ev(rs_ref), 2)}", flush=True)
    for arm in arms:
        code = arm[:3]
        rs = compose_ppo(rs_ref, rs_d, code)
        if "+b" in arm:
            other = rs_d if (arm.endswith("bD") or arm.endswith("bDr")) else rs_c
            rs = dict(rs, behaviour_actor=actor_of(other), behaviour_until=jnp.full((n,), w1 + a.behaviour_k, jnp.int32))
            if arm.endswith("r"):  # ratio-neutral: learner log-probs stored, PPO ratio starts at 1
                rs = dict(rs, behaviour_ratio_learner=jnp.ones((n,), bool))
        evals, train, diag = [ev(rs)], [], []
        for _ in range(a.recovery // E):
            rs, m = run_v(rs)
            train.append(jax.device_get(m["episode_return"]))
            diag.append({k: np.asarray(jax.device_get(m[k])) for k in ("clip_frac", "ratio_absdev", "approx_kl", "behaviour_kl")})
            evals.append(ev(rs))
        evals = np.stack(evals, 1)  # seeds x checkpoints
        train = np.concatenate(train, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr_ds = np.nanmean(train.reshape(n, -1, E), 2)
        np.savez_compressed(out_dir / f"{arm}.npz", eval=evals, train_return=train)
        summary["arms"][arm] = dict(eval_return_mean=np.round(evals, 3).tolist(),
                                    train_return_by_block=[[None if np.isnan(v) else round(float(v), 3) for v in r]
                                                           for r in tr_ds])
        dg = {k: np.concatenate([d[k] for d in diag], 1) for k in diag[0]}  # seeds x updates
        summary["arms"][arm]["diag_first_k"] = {k: round(float(v[:, :a.behaviour_k].mean()), 4) for k, v in dg.items()}
        summary["arms"][arm]["diag_next_k"] = {k: round(float(v[:, a.behaviour_k:2 * a.behaviour_k].mean()), 4) for k, v in dg.items()}
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary, indent=1))
        print(f"arm {arm:>7}: eval at +0 {np.round(evals[:,0],1)} +50 {np.round(evals[:,min(5, evals.shape[1]-1)],1)} "
              f"+{a.recovery} {np.round(evals[:,-1],1)} elapsed={time.time()-t0:.0f}s", flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

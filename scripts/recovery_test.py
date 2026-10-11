"""Recovery test for fixed action-mask deficits, with a deficit-free evaluation protocol.

Protocol (all times in PPO updates; 1 update = NUM_ENVS * NUM_STEPS = 8192 env steps):
  * Every agent (control and deficit) trains for --updates updates with the same seeds.
  * Every --eval-every updates (and at update 0), each agent's current params are evaluated by
    `make_eval`: full action set (no training-time mask), clean observations, true env reward,
    --eval-envs episodes, actions sampled from the policy, and the SAME eval key for every agent and
    checkpoint (common random numbers). Mask boundaries are multiples of --eval-every, so there is an
    evaluation exactly at removal.
  * The training-time return (masked, as experienced) is also logged per update.

Interventions (SpaceInvaders action indices: 0 noop, 1 left, 2 right, 3 fire):
  movement restriction : allowed [0, 3]    (no left/right)
  fire restriction     : allowed [0, 1, 2] (no fire; scoring essentially impossible). A *different*
                         intervention, not a matched-severity replicate.

Usage:
  python scripts/recovery_test.py --configs control move_early move_late
Outputs: runs/<stem>.json, results/<stem>/<config>.npz (ignored), results/summary/<stem>.json (committed).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rled.agents.ppo import make, make_eval
from rled.provenance import build_manifest, write_manifest

warnings.filterwarnings("ignore")

INTERVENTIONS = {"move": [0, 3], "fire": [0, 1, 2]}


def parse_config(name: str, early: tuple[int, int], late: tuple[int, int]):
    if name == "control":
        return dict(name=name, allowed=INTERVENTIONS["move"], window=(0, 0))
    kind, when = name.split("_")
    return dict(name=name, allowed=INTERVENTIONS[kind], window=early if when == "early" else late)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--configs", nargs="+", default=["control", "move_early", "move_late"])
    ap.add_argument("--early", type=int, nargs=2, default=[0, 50], help="mask window [start, end) in updates")
    ap.add_argument("--late", type=int, nargs=2, default=[100, 150])
    ap.add_argument("--updates", type=int, default=440)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--tag", default="recovery-move")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    E = a.eval_every
    for w in (a.early, a.late):
        assert w[0] % E == 0 and w[1] % E == 0, "mask boundaries must be multiples of --eval-every"
    assert a.updates % E == 0
    configs = [parse_config(c, tuple(a.early), tuple(a.late)) for c in a.configs]

    base = dict(ENV_NAME=a.env, HIDDEN=a.hidden, DEFICIT="action_mask")
    spu = 64 * 128  # default NUM_ENVS * NUM_STEPS
    base["TOTAL_TIMESTEPS"] = a.updates * spu
    name = f"p2-{a.tag}-ppo-{a.env}"
    protocol = dict(updates=a.updates, steps_per_update=spu, eval_every=E, eval_envs=a.eval_envs,
                    eval_seed=a.eval_seed, eval_policy="stochastic, unmasked, clean obs, true reward",
                    early=a.early, late=a.late, interventions=INTERVENTIONS,
                    configs={c["name"]: dict(allowed=c["allowed"], window=list(c["window"])) for c in configs})
    manifest = build_manifest(name, {**base, "protocol": protocol}, a.seeds, allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)
    out_dir = Path("results") / mpath.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_path = Path("results/summary") / f"{mpath.stem}.json"

    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    eval_key = jax.random.PRNGKey(a.eval_seed)
    eval_updates = list(range(0, a.updates + 1, E))
    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds, eval_updates=eval_updates,
                   eval_steps=[u * spu for u in eval_updates], configs={})
    compiled = {}
    t0 = time.time()
    for c in configs:
        key = tuple(c["allowed"])
        if key not in compiled:
            cfg, init, run = make({**base, "DEFICIT_PARAMS": {"allowed_actions": c["allowed"]}})
            ev = make_eval(cfg, num_envs=a.eval_envs)
            compiled[key] = (cfg, jax.jit(jax.vmap(init)), jax.jit(jax.vmap(lambda rs: run(rs, E))),
                             jax.jit(jax.vmap(ev, in_axes=(0, None))))
        cfg, init_v, run_v, eval_v = compiled[key]
        rs = init_v(keys)
        w0, w1 = c["window"]
        rs["deficit_bounds"] = jnp.tile(jnp.array([w0 * spu, w1 * spu], jnp.int32), (len(a.seeds), 1))

        def do_eval(rs):
            o = jax.device_get(eval_v(rs["train_state"].params, eval_key))
            return o["ret"], o["length"], o["unfinished"]

        ev_ret, ev_len, ev_unf, train = [], [], [], []
        r, l, u = do_eval(rs); ev_ret.append(r); ev_len.append(l); ev_unf.append(u)
        for i in range(a.updates // E):
            rs, m = run_v(rs)
            train.append(jax.device_get(m))
            r, l, u = do_eval(rs); ev_ret.append(r); ev_len.append(l); ev_unf.append(u)
            if (i + 1) % 5 == 0:
                print(f"  {c['name']:>11} update {(i+1)*E:>4}/{a.updates} eval/seed={np.round(r.mean(1),1)} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        ev_ret = np.stack(ev_ret, 1)  # seeds x evals x episodes
        ev_len = np.stack(ev_len, 1)
        ev_unf = np.stack(ev_unf, 1)
        train = {k: np.concatenate([t[k] for t in train], axis=1) for k in train[0]}
        np.savez_compressed(out_dir / f"{c['name']}.npz", eval_ret=ev_ret, eval_len=ev_len, eval_unfinished=ev_unf,
                            **{f"train_{k}": v for k, v in train.items()})
        k = E  # training curve downsampled to eval resolution
        tr = train["episode_return"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr_ds = np.nanmean(tr[:, : tr.shape[1] // k * k].reshape(tr.shape[0], -1, k), axis=2)
        r3 = lambda x: [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in x]
        summary["configs"][c["name"]] = dict(
            allowed=c["allowed"], window_updates=[w0, w1], window_steps=[w0 * spu, w1 * spu],
            deficit_updates=int(train["deficit_active"][0].sum()),
            eval_return_mean=r3(ev_ret.mean(2)),          # seeds x evals
            eval_return_sem=r3(ev_ret.std(2) / np.sqrt(ev_ret.shape[2])),
            eval_length_mean=r3(ev_len.mean(2)),
            eval_unfinished_frac=r3(ev_unf.mean(2)),
            train_return_by_block=r3(tr_ds),               # seeds x (updates/E): mean over updates (u-E, u]
        )
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary, indent=1))  # incremental: partial progress survives
        print(f"done {c['name']}: final eval/seed={np.round(ev_ret[:, -1].mean(1), 1)} elapsed={time.time()-t0:.0f}s",
              flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

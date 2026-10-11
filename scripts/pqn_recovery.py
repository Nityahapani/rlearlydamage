"""Phase 5 recovery replication with PQN / PQN-R on MinAtar (fixed action restriction, deficit-free eval).

Arms (same seeds; the deficit window is a traced array, so one compile serves all arms of a variant):
  control : no restriction
  early   : restriction on updates [e0, e0 + dur)
  late    : restriction on updates [l0, l0 + dur)
Evaluation every --eval-every updates: full action set, clean observations, true reward, greedy (eps 0),
--eval-envs episodes, same key for every arm and checkpoint (common random numbers).

Matching to PPO (recorded in the manifest under `exposure`): windows are matched in *environment steps*
(PPO: 50 updates x 8192 = 409,600 steps -> PQN: 100 updates x 4096), and the late onset is chosen by meaning
(see LOG: the clean control's competence at onset and a finished exploration schedule), not by update count.
Gradient steps, sample reuse and the exploration schedule differ and are recorded explicitly.
Usage:
  python scripts/pqn_recovery.py --variant pqn  --arms control --updates 1000
  python scripts/pqn_recovery.py --variant pqn  --arms early late --late-onset 300 --updates 1000
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

from rled.agents.pqn import make, make_eval
from rled.provenance import build_manifest, write_manifest

warnings.filterwarnings("ignore")
METRICS = ("episode_return", "eps", "loss", "q_mean", "grad_norm", "deficit_active")
PPO_REF = dict(steps_per_update=8192, grad_steps_per_update=32, minibatch=1024, epochs=4, window_updates=50,
               window_env_steps=409_600, early=[0, 50], late=[100, 150], total_env_steps=3_604_480,
               post_removal_env_steps_late=2_375_680, exploration="stochastic policy, entropy bonus 0.01")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["pqn", "pqnr"], required=True)
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--allowed", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--arms", nargs="+", default=["control"])
    ap.add_argument("--duration", type=int, default=100, help="window length in updates (100 x 4096 = PPO's 409,600 steps)")
    ap.add_argument("--early-onset", type=int, default=0)
    ap.add_argument("--late-onset", type=int, default=None)
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if "late" in a.arms and a.late_onset is None:
        ap.error("--late-onset is required for the late arm")
    E = a.eval_every
    assert a.updates % E == 0

    base = dict(ENV_NAME=a.env, REPLAY=a.variant == "pqnr", DEFICIT="action_mask",
                DEFICIT_PARAMS={"allowed_actions": a.allowed}, DEFICIT_START=0, DEFICIT_END=0)
    cfg, init, run = make(base)
    spu = cfg["STEPS_PER_UPDATE"]
    windows = {"control": (0, 0), "early": (a.early_onset, a.early_onset + a.duration)}
    if a.late_onset is not None:
        windows["late"] = (a.late_onset, a.late_onset + a.duration)
    exposure = dict(
        pqn=dict(steps_per_update=spu, grad_steps_per_update=cfg["GRAD_STEPS_PER_UPDATE"],
                 batch=cfg["REPLAY_BATCH"] if cfg["REPLAY"] else cfg["MINIBATCH_SIZE"], replay_ratio=cfg["REPLAY_RATIO"],
                 buffer_transitions=cfg["BUFFER_SIZE"] if cfg["REPLAY"] else 0,
                 buffer_turnover_updates=cfg["BUFFER_SIZE"] / spu if cfg["REPLAY"] else 0,
                 eps_schedule=f"linear {cfg['EPS_START']} -> {cfg['EPS_FINISH']} over {cfg['EPS_DECAY_STEPS']} env steps "
                              f"({cfg['EPS_DECAY_STEPS'] / spu:.0f} updates), then constant",
                 lr=f"{cfg['LR']} constant (RAdam, grad clip {cfg['MAX_GRAD_NORM']})",
                 window_env_steps=a.duration * spu, window_grad_steps=a.duration * cfg["GRAD_STEPS_PER_UPDATE"],
                 window_sample_gradients=a.duration * cfg["GRAD_STEPS_PER_UPDATE"] *
                 (cfg["REPLAY_BATCH"] if cfg["REPLAY"] else cfg["MINIBATCH_SIZE"])),
        ppo_reference=dict(PPO_REF, window_grad_steps=50 * 32, window_sample_gradients=409_600 * 4))
    protocol = dict(variant=a.variant, windows_updates=windows,
                    windows_env_steps={k: [w[0] * spu, w[1] * spu] for k, w in windows.items()},
                    eps_at_window={k: [float(max(cfg["EPS_FINISH"], 1 - (1 - cfg["EPS_FINISH"]) * w[0] * spu / cfg["EPS_DECAY_STEPS"])),
                                       float(max(cfg["EPS_FINISH"], 1 - (1 - cfg["EPS_FINISH"]) * w[1] * spu / cfg["EPS_DECAY_STEPS"]))]
                                   for k, w in windows.items()},
                    updates=a.updates, eval_every=E, eval_envs=a.eval_envs, eval_seed=a.eval_seed, eval_eps=0.0,
                    allowed=a.allowed, arms=a.arms, exposure=exposure)
    tag = a.tag or f"{a.variant}-{'-'.join(a.arms)}"
    manifest = build_manifest(f"p5-recovery-{tag}-{a.env}", {**cfg, "protocol": protocol}, a.seeds, allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)
    print(json.dumps(exposure, indent=1), flush=True)
    summ_path = Path("results/summary") / f"{mpath.stem}.json"

    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    n = len(a.seeds)
    init_v = jax.jit(jax.vmap(init))
    run_v = jax.jit(jax.vmap(lambda rs: run(rs, E)))
    eval_v = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs), in_axes=(0, None)))
    ek = jax.random.PRNGKey(a.eval_seed)
    ev = lambda rs: np.asarray(jax.device_get(eval_v(rs["train_state"].params, ek)["ret"])).mean(1)
    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds,
                   eval_updates=list(range(0, a.updates + 1, E)), arms={})
    t0 = time.time()
    for arm in a.arms:
        w = windows[arm]
        rs = init_v(keys)
        rs["deficit_bounds"] = jnp.tile(jnp.array([w[0] * spu, w[1] * spu], jnp.int32), (n, 1))
        evals, train = [ev(rs)], []
        for i in range(a.updates // E):
            rs, m = run_v(rs)
            evals.append(ev(rs))
            train.append({k: np.asarray(m[k]) for k in METRICS})
            if (i + 1) % 10 == 0:
                print(f"  {arm} u{(i + 1) * E}: eval {np.round(evals[-1], 1)} train-ret "
                      f"{np.round(train[-1]['episode_return'][:, -1], 1)} eps {train[-1]['eps'][0, -1]:.3f} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
        r4 = lambda x: np.round(np.asarray(x, float), 4).tolist()
        summary["arms"][arm] = dict(window_updates=list(w), eval_return_mean=r4(np.stack(evals, 1)),
                                    train={k: r4(np.concatenate([t[k] for t in train], 1)) for k in METRICS})
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary))
        print(f"arm {arm} done, final eval {np.round(evals[-1], 1)} elapsed={time.time() - t0:.0f}s", flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

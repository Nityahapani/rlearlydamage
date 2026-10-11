"""Phase 5 replay-buffer surgery (PQN-R): network parameters x replay buffer, 2 x 2, at removal of the late restriction.

Forks: shared history 0 -> l0 (no deficit); C: l0 -> l1 unrestricted; D: l0 -> l1 restricted (l1 = l0 + dur).
Saved at removal (results/<stem>/states_removal.pkl): params, optimizer state, replay buffer (packed obs, actions,
rewards, dones, write pointer, size), env state, current obs, RNG, update counter -- for C and D.

Arms (code = network source, buffer source):
  CC, DC, CD, DD.
Prespecified assignment rule (never varies with the buffer treatment):
  * network carrier = parameters + optimizer (RAdam) state, taken together from the network source
    (the optimizer moments are a function of the parameter trajectory; PPO Phase 3 found ~0 optimizer effect);
  * buffer carrier  = all buffer arrays incl. write pointer and size (identical in C and D at removal);
  * everything else from C: env state, current obs, RNG, update counter (=> identical exploration schedule;
    epsilon is already at its floor), deficit bounds (none);
  * target network: none (PQN-R bootstraps from the online network), so nothing to assign.
Thus CC is exactly the clean control's continuation (checked against the recovery run if --recovery-summary is given).
Every arm is evaluated deficit-free (full action set, greedy, clean obs, true reward, common random numbers).
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

from rled.agents.pqn import make, make_eval
from rled.provenance import build_manifest, write_manifest

warnings.filterwarnings("ignore")
METRICS = ("episode_return", "eps", "loss", "q_mean", "grad_norm", "deficit_active")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["pqnr"], default="pqnr")
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--allowed", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--late-onset", type=int, required=True)
    ap.add_argument("--duration", type=int, default=100)
    ap.add_argument("--recovery", type=int, default=580)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=["CC", "DC", "CD", "DD"])
    ap.add_argument("--recovery-summary", default=None, help="p5 recovery summary with control + late arms (consistency check)")
    ap.add_argument("--tag", default="surgery-late")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    E = a.eval_every
    l0, l1 = a.late_onset, a.late_onset + a.duration
    assert l0 % E == 0 and a.duration % E == 0 and a.recovery % E == 0

    cfg, init, run = make(dict(ENV_NAME=a.env, REPLAY=True, DEFICIT="action_mask",
                               DEFICIT_PARAMS={"allowed_actions": a.allowed}, DEFICIT_START=0, DEFICIT_END=0))
    spu = cfg["STEPS_PER_UPDATE"]
    protocol = dict(variant=a.variant, window_updates=[l0, l1], recovery=a.recovery, eval_every=E, eval_envs=a.eval_envs,
                    eval_seed=a.eval_seed, eval_eps=0.0, allowed=a.allowed, arms=a.arms,
                    assignment_rule="network = params + optimizer state (together); buffer = all buffer arrays; "
                                    "env state, obs, RNG, update counter, exploration schedule, deficit bounds from C; "
                                    "no target network",
                    buffer_turnover_updates=cfg["BUFFER_SIZE"] / spu)
    manifest = build_manifest(f"p5-{a.tag}-{a.variant}-{a.env}", {**cfg, "protocol": protocol}, a.seeds, allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)
    out_dir = Path("results") / mpath.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_path = Path("results/summary") / f"{mpath.stem}.json"

    n = len(a.seeds)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    run_v = jax.jit(jax.vmap(lambda rs: run(rs, E)))
    eval_v = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs), in_axes=(0, None)))
    ek = jax.random.PRNGKey(a.eval_seed)
    ev = lambda rs: np.asarray(jax.device_get(eval_v(rs["train_state"].params, ek)["ret"])).mean(1)
    bounds = lambda b0, b1: jnp.tile(jnp.array([b0 * spu, b1 * spu], jnp.int32), (n, 1))
    t0 = time.time()

    rs = jax.jit(jax.vmap(init))(keys)
    rs["deficit_bounds"] = bounds(0, 0)
    for _ in range(l0 // E):
        rs, _ = run_v(rs)
    forks, fork_eval = {}, {}
    for name, b in (("C", bounds(0, 0)), ("D", bounds(l0, l1))):
        r = dict(rs, deficit_bounds=b)
        e = [ev(r)]
        for _ in range(a.duration // E):
            r, _ = run_v(r)
            e.append(ev(r))
        forks[name], fork_eval[name] = r, np.stack(e, 1)
        print(f"fork {name}: eval at onset {np.round(e[0], 1)} removal {np.round(e[-1], 1)} elapsed={time.time() - t0:.0f}s", flush=True)
    C, D = forks["C"], forks["D"]
    keep = ("env_state", "obs", "rng", "update", "buffer")
    states = {k: jax.device_get(dict(params=f["train_state"].params, opt_state=f["train_state"].opt_state,
                                     **{x: f[x] for x in keep})) for k, f in forks.items()}
    with open(out_dir / "states_removal.pkl", "wb") as fh:
        pickle.dump(states, fh)
    same_buf_meta = bool(np.array_equal(states["C"]["buffer"]["ptr"], states["D"]["buffer"]["ptr"]) and
                         np.array_equal(states["C"]["buffer"]["size"], states["D"]["buffer"]["size"]))
    # how much of D's buffer was written under the restriction (slots written during [l0, l1))
    in_window = min(a.duration * cfg["NUM_STEPS"], cfg["BUFFER_LEN"]) / cfg["BUFFER_LEN"]

    check = {}
    if a.recovery_summary:
        rsum = json.loads(Path(a.recovery_summary).read_text())
        U = rsum["eval_updates"]
        i0, i1 = U.index(l0), U.index(l1)
        rc = np.array(rsum["arms"]["control"]["eval_return_mean"])[:, i0:i1 + 1]
        rl = np.array(rsum["arms"]["late"]["eval_return_mean"])[:, i0:i1 + 1]
        check = dict(C_fork_matches_recovery_control=bool(np.allclose(rc, fork_eval["C"], atol=1e-3)),
                     D_fork_matches_recovery_late=bool(np.allclose(rl, fork_eval["D"], atol=1e-3)))
        print("consistency vs recovery run:", check, flush=True)

    def compose(code):
        net, buf = forks[code[0]], forks[code[1]]
        ts = C["train_state"].replace(params=net["train_state"].params, opt_state=net["train_state"].opt_state)
        return dict(C, train_state=ts, buffer=buf["buffer"], deficit_bounds=bounds(0, 0))

    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds, consistency_check=check,
                   buffer_meta_identical=same_buf_meta, frac_D_buffer_written_in_window=in_window,
                   fork_updates=list(range(l0, l1 + 1, E)), eval_updates=list(range(l1, l1 + a.recovery + 1, E)),
                   forks={k: np.round(v, 4).tolist() for k, v in fork_eval.items()}, arms={})
    r4 = lambda x: np.round(np.asarray(x, float), 4).tolist()
    for code in a.arms:
        r = compose(code)
        evals, train = [ev(r)], []
        for _ in range(a.recovery // E):
            r, m = run_v(r)
            evals.append(ev(r))
            train.append({k: np.asarray(m[k]) for k in METRICS})
        summary["arms"][code] = dict(eval_return_mean=r4(np.stack(evals, 1)),
                                     train={k: r4(np.concatenate([t[k] for t in train], 1)) for k in METRICS})
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary))
        e = np.stack(evals, 1)
        print(f"arm {code}: +0 {e[:, 0].mean():5.1f} +{E} {e[:, 1].mean():5.1f} +50 {e[:, min(5, e.shape[1] - 1)].mean():5.1f} "
              f"end {e[:, -1].mean():5.1f} AUC {e[:, 1:].mean():5.1f} elapsed={time.time() - t0:.0f}s", flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

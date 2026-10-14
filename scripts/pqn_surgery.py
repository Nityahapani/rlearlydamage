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
EVAL_EPS2 = 0.05  # second eval: epsilon-greedy at the training floor (greedy eval is brittle)
METRICS = ("episode_return", "eps", "loss", "q_mean", "grad_norm", "deficit_active", "buf_frac_window", "sample_frac_window")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["pqnr"], default="pqnr")
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--allowed", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--deficit", default="action_mask", choices=["action_mask", "reward_flip"],
                    help="reward_flip (Phase 12): stored reward sign-flipped during the window (p = 1)")
    ap.add_argument("--late-onset", type=int, required=True)
    ap.add_argument("--buffer-size", type=int, default=100_000)
    ap.add_argument("--save-states", choices=["auto", "full", "nobuffer", "none"], default="auto",
                    help="auto = full for buffers <= 200k transitions, else params/optimizer/counters only")
    ap.add_argument("--duration", type=int, default=100)
    ap.add_argument("--recovery", type=int, default=580)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=["CC", "DC", "CD", "DD"])
    ap.add_argument("--recovery-summary", nargs="*", default=None, help="p5 recovery summaries containing the control and late arms (consistency check)")
    ap.add_argument("--tag", default="surgery-late")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    E = a.eval_every
    l0, l1 = a.late_onset, a.late_onset + a.duration
    assert l0 % E == 0 and a.duration % E == 0 and a.recovery % E == 0

    label = "pqnr" if a.buffer_size == 100_000 else f"pqnr-{a.buffer_size // 1000}k"
    cfg, init, run = make(dict(ENV_NAME=a.env, REPLAY=True, BUFFER_SIZE=a.buffer_size, DEFICIT=a.deficit,
                               DEFICIT_PARAMS={"allowed_actions": a.allowed}, DEFICIT_START=0, DEFICIT_END=0))
    spu = cfg["STEPS_PER_UPDATE"]
    protocol = dict(variant=label, window_updates=[l0, l1], recovery=a.recovery, eval_every=E, eval_envs=a.eval_envs,
                    eval_seed=a.eval_seed, eval_eps=[0.0, EVAL_EPS2], allowed=a.allowed, arms=a.arms, deficit=a.deficit,
                    assignment_rule="network = params + optimizer state (together); buffer = all buffer arrays; "
                                    "env state, obs, RNG, update counter, exploration schedule, deficit bounds from C; "
                                    "no target network",
                    buffer_turnover_updates=cfg["BUFFER_SIZE"] / spu, buffer_size=cfg["BUFFER_SIZE"],
                    overwrite_diagnostics="tag_window = [l0, l1): buf_frac_window = share of buffer slots written during the "
                                          "restriction window; sample_frac_window = share of sampled replay transitions from it")
    manifest = build_manifest(f"p5-{a.tag}-{label}-{a.env}", {**cfg, "protocol": protocol}, a.seeds, allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)
    out_dir = Path("results") / mpath.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_path = Path("results/summary") / f"{mpath.stem}.json"

    n = len(a.seeds)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    run_v = jax.jit(jax.vmap(lambda rs: run(rs, E)), donate_argnums=0)
    fresh = lambda r: jax.tree.map(jnp.copy, r)  # donation invalidates inputs: always run on a private copy
    eval_g = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs, eps=0.0), in_axes=(0, None)))
    eval_e = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs, eps=EVAL_EPS2), in_axes=(0, None)))
    ek = jax.random.PRNGKey(a.eval_seed)
    ev2 = lambda rs: (np.asarray(jax.device_get(eval_g(rs["train_state"].params, ek)["ret"])).mean(1),
                      np.asarray(jax.device_get(eval_e(rs["train_state"].params, ek)["ret"])).mean(1))
    bounds = lambda b0, b1: jnp.tile(jnp.array([b0 * spu, b1 * spu], jnp.int32), (n, 1))
    t0 = time.time()

    rs = jax.jit(jax.vmap(init))(keys)
    rs["deficit_bounds"] = bounds(0, 0)
    rs["tag_window"] = jnp.tile(jnp.array([l0, l1], jnp.int32), (n, 1))  # diagnostics only
    for _ in range(l0 // E):
        rs, _ = run_v(rs)
    forks, fork_eval = {}, {}
    for name, b in (("C", bounds(0, 0)), ("D", bounds(l0, l1))):
        r = fresh(dict(rs, deficit_bounds=b))
        g, x = ev2(r)
        e, ee = [g], [x]
        for _ in range(a.duration // E):
            r, _ = run_v(r)
            g, x = ev2(r)
            e.append(g); ee.append(x)
        forks[name], fork_eval[name], fork_eval[name + "_eps05"] = r, np.stack(e, 1), np.stack(ee, 1)
        print(f"fork {name}: eval at onset {np.round(e[0], 1)} removal {np.round(e[-1], 1)} elapsed={time.time() - t0:.0f}s", flush=True)
    del rs
    C, D = forks["C"], forks["D"]
    mode = a.save_states if a.save_states != "auto" else ("full" if cfg["BUFFER_SIZE"] <= 200_000 else "nobuffer")
    if mode != "none":
        keep = ("env_state", "obs", "rng", "update") + (("buffer",) if mode == "full" else ())
        states = {k: jax.device_get(dict(params=f["train_state"].params, opt_state=f["train_state"].opt_state,
                                         **{x: f[x] for x in keep})) for k, f in forks.items()}
        with open(out_dir / "states_removal.pkl", "wb") as fh:
            pickle.dump(states, fh)
        del states
    same_buf_meta = bool(np.array_equal(np.asarray(C["buffer"]["ptr"]), np.asarray(D["buffer"]["ptr"])) and
                         np.array_equal(np.asarray(C["buffer"]["size"]), np.asarray(D["buffer"]["size"])))
    # how much of D's buffer was written under the restriction (slots written during [l0, l1))
    tg = np.asarray(D["buffer"]["tag"])
    in_window = float(((tg >= l0) & (tg < l1)).sum() / max((tg >= 0).sum(), 1))  # share of valid D-buffer slots

    check = {}
    if a.recovery_summary:
        arms_by = {}
        for f in a.recovery_summary:
            d = json.loads(Path(f).read_text())
            for arm, v in d["arms"].items():  # pool seed groups across files
                sd0, U0, M0 = arms_by.get(arm, ([], d["eval_updates"], None))
                M1 = np.array(v["eval_eps05_return_mean"])
                arms_by[arm] = (sd0 + d["seeds"], U0, M1 if M0 is None else np.concatenate([M0, M1]))
        for arm, fk, key in (("control", "C_eps05", "C_fork_matches_recovery_control"), ("late", "D_eps05", "D_fork_matches_recovery_late")):
            if arm not in arms_by:
                continue
            sd, U, M = arms_by[arm]
            both = [s_ for s_ in a.seeds if s_ in sd]
            if not both:
                continue
            i0, i1 = U.index(l0), U.index(l1)
            mine = fork_eval[fk][[a.seeds.index(s_) for s_ in both]]
            theirs = M[[sd.index(s_) for s_ in both], i0:i1 + 1]
            check[key] = dict(seeds=both, identical=bool(np.allclose(mine, theirs, atol=1e-3)))
        print("consistency vs recovery run:", check, flush=True)

    def compose(code):
        if len(code) == 3:  # (params, optimizer state, buffer) -- separates the network carrier
            par, opt, buf = forks[code[0]], forks[code[1]], forks[code[2]]
        else:               # (network = params + optimizer state, buffer) -- prespecified Phase 5 rule
            par, opt, buf = forks[code[0]], forks[code[0]], forks[code[1]]
        ts = C["train_state"].replace(params=par["train_state"].params, opt_state=opt["train_state"].opt_state)
        return dict(C, train_state=ts, buffer=buf["buffer"], deficit_bounds=bounds(0, 0))

    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds, consistency_check=check,
                   buffer_meta_identical=same_buf_meta, frac_D_buffer_written_in_window=in_window,
                   fork_updates=list(range(l0, l1 + 1, E)), eval_updates=list(range(l1, l1 + a.recovery + 1, E)),
                   forks={k: np.round(v, 4).tolist() for k, v in fork_eval.items()}, arms={})
    r4 = lambda x: np.round(np.asarray(x, float), 4).tolist()
    for code in a.arms:
        r = fresh(compose(code))
        g, x = ev2(r)
        evals, evals_e, train = [g], [x], []
        for _ in range(a.recovery // E):
            r, m = run_v(r)
            g, x = ev2(r)
            evals.append(g); evals_e.append(x)
            train.append({k: np.asarray(m[k]) for k in METRICS})
        summary["arms"][code] = dict(eval_return_mean=r4(np.stack(evals, 1)), eval_eps05_return_mean=r4(np.stack(evals_e, 1)),
                                     train={k: r4(np.concatenate([t[k] for t in train], 1)) for k in METRICS})
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary))
        e = np.stack(evals_e, 1)
        print(f"arm {code} (eps .05 eval): +0 {e[:, 0].mean():5.1f} +{E} {e[:, 1].mean():5.1f} +50 {e[:, min(5, e.shape[1] - 1)].mean():5.1f} "
              f"end {e[:, -1].mean():5.1f} AUC {e[:, 1:].mean():5.1f} elapsed={time.time() - t0:.0f}s", flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

"""Phase 6 replay-purge experiment (PQN-R, 1M buffer): does replay data leave a scar after it is removed?

History per seed (identical to the Phase 6 1M surgery): shared 0 -> l0 unrestricted; fork C: l0 -> l1 unrestricted;
fork D: l0 -> l1 movement-restricted ({noop, fire}); removal at l1 (= u350 by the prespecified onset rule).

Arms at removal (recovery R updates, eval every E; deficit-free eval as in Phase 5):
  A_retain      damaged agent continues with its own 1M buffer (restriction-era data overwritten by normal collection)
  B_purge       damaged agent; at l1 the ENTIRE buffer is replaced by the replacement buffer (below)
  C_delay50     damaged agent; keeps its buffer for `--delay` (50) updates (identical to A_retain up to l1+50),
                then the entire buffer is replaced by the SAME replacement buffer as B
  D_clean       clean agent continues with its own buffer (= the 1M control run; checked if summaries are given)
  E_clean_purge (secondary, added) clean agent; at l1 its buffer is replaced by the same replacement buffer.
                Measures the effect of the purge procedure / replacement data itself on a clean network.
Everything except the buffer is the arm's own fork state: params, optimizer (RAdam) state, env state, current obs,
RNG, update counter (A/B/C from fork D, D/E from fork C).

Replacement buffer (prespecified, one per seed, used identically by B, C and E):
  * real transitions, not relabelled old data: a FROZEN clean actor (fork C's parameters at l1) acts epsilon-greedy
    with eps = 0.05 (the training floor) over the FULL action set in 128 fresh, unrestricted envs (training wrappers),
    key = fold_in(PRNGKey(seed), 2026) -- separate from every agent's env state / RNG;
  * BUFFER_LEN x 128 steps (fills all 1M slots), time-ordered; written with ptr = 0, size = full, so FIFO overwrites
    the oldest replacement data first; slot tags = the update at which the purge happens (outside [l0, l1)).
  * Splice note: after any buffer swap, the last replacement row's successor is the agent's next written row, i.e.
    128 of 1,000,000 transitions (0.013%) have the agent's obs as successor (as in all Phase 5/6 surgery arms).
Restriction-era share: share of valid slots tagged in [l0, l1) at every eval checkpoint (0 for clean buffers and
right after a purge).
Usage: python scripts/pqn_purge.py --seeds 0 --tag purge [--control-summary <1M control summaries>]
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

from rled.agents.pqn import make, make_collector, make_eval
from rled.provenance import build_manifest, write_manifest

warnings.filterwarnings("ignore")
EVAL_EPS2 = 0.05
COLLECT_EPS = 0.05
COLLECT_KEY_SALT = 2026
METRICS = ("episode_return", "loss", "q_mean", "grad_norm", "buf_frac_window", "sample_frac_window")
ARMS = ("A_retain", "B_purge", "C_delay50", "D_clean", "E_clean_purge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--allowed", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--late-onset", type=int, default=250)
    ap.add_argument("--duration", type=int, default=100)
    ap.add_argument("--recovery", type=int, default=650)
    ap.add_argument("--delay", type=int, default=50)
    ap.add_argument("--buffer-size", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--control-summary", nargs="*", default=None, help="1M control summaries (D_clean consistency check)")
    ap.add_argument("--surgery-summary", nargs="*", default=None, help="Phase 6 1M surgery summaries (fork consistency check)")
    ap.add_argument("--tag", default="purge")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    E = a.eval_every
    l0, l1 = a.late_onset, a.late_onset + a.duration
    assert l0 % E == 0 and a.duration % E == 0 and a.recovery % E == 0 and a.delay % E == 0 and a.delay < a.recovery

    label = f"pqnr-{a.buffer_size // 1000}k"
    base = dict(ENV_NAME=a.env, REPLAY=True, BUFFER_SIZE=a.buffer_size, DEFICIT="action_mask",
                DEFICIT_PARAMS={"allowed_actions": a.allowed}, DEFICIT_START=0, DEFICIT_END=0)
    cfg, init, run = make(base)
    spu, N, L = cfg["STEPS_PER_UPDATE"], cfg["NUM_ENVS"], cfg["BUFFER_LEN"]
    protocol = dict(variant=label, window_updates=[l0, l1], recovery=a.recovery, delay=a.delay, eval_every=E,
                    eval_envs=a.eval_envs, eval_seed=a.eval_seed, eval_eps=[0.0, EVAL_EPS2], allowed=a.allowed, arms=list(ARMS),
                    replacement=dict(policy="frozen clean actor (fork C params at removal)", eps=COLLECT_EPS,
                                     actions="full action set", envs=f"{N} fresh unrestricted envs, key fold_in(PRNGKey(seed), {COLLECT_KEY_SALT})",
                                     steps=f"{L} x {N} = {L * N} transitions (fills the buffer)", layout="time-ordered, ptr=0, size=full",
                                     tags="update at which the purge happens", shared="identical replacement buffer for B, C, E"),
                    state_rule="all non-buffer state (params, optimizer, env state, obs, RNG, update counter) from the arm's own fork",
                    splice_note="after a swap, 128 of 1M transitions have the agent's next obs as successor (0.013%)",
                    buffer_size=cfg["BUFFER_SIZE"], buffer_turnover_updates=cfg["BUFFER_SIZE"] / spu)
    manifest = build_manifest(f"p6-{a.tag}-{label}-{a.env}", {**cfg, "protocol": protocol}, a.seeds, allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)
    summ_path = Path("results/summary") / f"{mpath.stem}.json"

    n = len(a.seeds)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    run_v = jax.jit(jax.vmap(lambda rs: run(rs, E)), donate_argnums=0)
    fresh = lambda r: jax.tree.map(jnp.copy, r)
    eval_g = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs, eps=0.0), in_axes=(0, None)))
    eval_e = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs, eps=EVAL_EPS2), in_axes=(0, None)))
    ek = jax.random.PRNGKey(a.eval_seed)
    ev2 = lambda rs: (np.asarray(jax.device_get(eval_g(rs["train_state"].params, ek)["ret"])).mean(1),
                      np.asarray(jax.device_get(eval_e(rs["train_state"].params, ek)["ret"])).mean(1))
    bounds = lambda b0, b1: jnp.tile(jnp.array([b0 * spu, b1 * spu], jnp.int32), (n, 1))
    tw = jnp.tile(jnp.array([l0, l1], jnp.int32), (n, 1))
    rshare = lambda r: np.asarray(jax.device_get(
        ((r["buffer"]["tag"] >= l0) & (r["buffer"]["tag"] < l1)).sum((1, 2)) / jnp.maximum(r["buffer"]["size"] * N, 1)))
    t0 = time.time()

    # ---- history and forks (identical to pqn_surgery.py)
    rs = jax.jit(jax.vmap(init))(keys)
    rs["deficit_bounds"] = bounds(0, 0)
    rs["tag_window"] = tw
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
        print(f"fork {name}: eps.05 eval at onset {np.round(ee[0], 1)} removal {np.round(ee[-1], 1)} elapsed={time.time() - t0:.0f}s", flush=True)
    del rs
    # removal: the deficit is over for every arm
    C = dict(forks.pop("C"), deficit_bounds=bounds(0, 0))
    D = dict(forks.pop("D"), deficit_bounds=bounds(0, 0))

    # ---- replacement buffer (frozen clean actor, fresh unrestricted envs)
    collect = jax.jit(jax.vmap(make_collector(base, eps=COLLECT_EPS)))
    ckeys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(s), COLLECT_KEY_SALT) for s in a.seeds])
    repl, cstats = collect(C["train_state"].params, ckeys)
    cstats = {k: np.asarray(v).round(3).tolist() for k, v in cstats.items()}
    print(f"replacement buffer collected: {cstats} elapsed={time.time() - t0:.0f}s", flush=True)

    def purge(r, at_update):
        b = dict(obs=jnp.copy(repl["obs"]), action=jnp.copy(repl["action"]), reward=jnp.copy(repl["reward"]),
                 done=jnp.copy(repl["done"]), tag=jnp.full((n, L, N), at_update, jnp.int16),
                 ptr=jnp.zeros((n,), jnp.int32), size=jnp.full((n,), L, jnp.int32))
        return dict(jax.tree.map(jnp.copy, {k: v for k, v in r.items() if k != "buffer"}), buffer=b)  # private copy, old buffer dropped

    check = {}
    r4 = lambda x: np.round(np.asarray(x, float), 4).tolist()
    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds, consistency_check=check, collection_stats=cstats,
                   restricted_share_at_removal=dict(C=r4(rshare(C)), D=r4(rshare(D))),
                   fork_updates=list(range(l0, l1 + 1, E)), eval_updates=list(range(l1, l1 + a.recovery + 1, E)),
                   forks={k: np.round(v, 4).tolist() for k, v in fork_eval.items()}, arms={})

    def run_arm(name, r, n_chunks, prefix=None, snapshot_at=None):
        """Run n_chunks of E updates with eval after each; `prefix` = (evals, evals_e, shares, train) already done."""
        evals, evals_e, shares, train = prefix if prefix else ([], [], [], [])
        if not prefix:
            g, x = ev2(r)
            evals.append(g); evals_e.append(x); shares.append(rshare(r))
        snap = None
        for i in range(n_chunks):
            r, m = run_v(r)
            g, x = ev2(r)
            evals.append(g); evals_e.append(x); shares.append(rshare(r))
            train.append({k: np.asarray(m[k]) for k in METRICS})
            if snapshot_at is not None and i + 1 == snapshot_at:
                snap = (fresh(r), (list(evals), list(evals_e), list(shares), list(train)))
        return r, (evals, evals_e, shares, train), snap

    def record(name, hist):
        evals, evals_e, shares, train = hist
        summary["arms"][name] = dict(eval_return_mean=r4(np.stack(evals, 1)), eval_eps05_return_mean=r4(np.stack(evals_e, 1)),
                                     restricted_share=r4(np.stack(shares, 1)),
                                     train={k: r4(np.concatenate([t[k] for t in train], 1)) for k in METRICS})
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary))
        e = np.stack(evals_e, 1)
        print(f"arm {name:13s} (eps .05): +0 {e[:, 0].mean():5.1f} +10 {e[:, 1].mean():5.1f} +50 {e[:, 5].mean():5.1f} "
              f"+100 {e[:, 10].mean():5.1f} end {e[:, -1].mean():5.1f} AUC {e[:, 1:].mean():5.1f} "
              f"share +0/+50/+60 {np.mean(shares[0]):.2f}/{np.mean(shares[5]):.2f}/{np.mean(shares[6]):.2f} elapsed={time.time() - t0:.0f}s", flush=True)

    K = a.recovery // E
    # clean-network arms first (frees C)
    _, h, _ = run_arm("E_clean_purge", purge(C, l1), K); record("E_clean_purge", h)
    _, h, _ = run_arm("D_clean", C, K); record("D_clean", h)
    del C
    _, h, _ = run_arm("B_purge", purge(D, l1), K); record("B_purge", h)
    _, h, snap = run_arm("A_retain", D, K, snapshot_at=a.delay // E); record("A_retain", h)
    del D
    r_c, pre = snap
    _, h, _ = run_arm("C_delay50", purge(r_c, l1 + a.delay), K - a.delay // E, prefix=pre); record("C_delay50", h)

    # ---- consistency checks
    if a.control_summary:
        sd, M, U = [], [], None
        for f in a.control_summary:
            d = json.loads(Path(f).read_text())
            sd += d["seeds"]; M.append(np.array(d["arms"]["control"]["eval_eps05_return_mean"])); U = d["eval_updates"]
        M = np.concatenate(M)
        both = [s for s in a.seeds if s in sd]
        if both:
            mine = np.array(summary["arms"]["D_clean"]["eval_eps05_return_mean"])[[a.seeds.index(s) for s in both]]
            theirs = M[[sd.index(s) for s in both]][:, [U.index(u) for u in summary["eval_updates"]]]
            check["D_clean_matches_1M_control_run"] = dict(seeds=both, identical=bool(np.allclose(mine, theirs, atol=1e-3)))
    if a.surgery_summary:
        for f in a.surgery_summary:
            d = json.loads(Path(f).read_text())
            for s in d["seeds"]:
                if s in a.seeds:
                    i, j = a.seeds.index(s), d["seeds"].index(s)
                    ok = all(np.allclose(fork_eval[k][i], np.array(d["forks"][k])[j], atol=1e-3) for k in ("C_eps05", "D_eps05"))
                    check.setdefault("forks_match_phase6_surgery", {})[str(s)] = bool(ok)
    summary["consistency_check"] = check
    summ_path.write_text(json.dumps(summary))
    print("consistency:", check, flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

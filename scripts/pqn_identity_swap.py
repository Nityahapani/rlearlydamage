"""Phase 6d matched-identity purge (PQN-R, 1M buffer). Pre-registered in LOG.md before any real run.

Question: holding slots, ages, replacement pool, exposure, retained residence, seams and learner state fixed, does it
matter WHICH transitions are removed: restriction-era transitions (R) or matched unrestricted transitions from the same
epoch (U)?

History per seed is identical to Phases 6/6b/6c: shared 0 -> l0 unrestricted; clean fork C and restricted fork D
(movement restriction, allowed {noop, fire}) over [l0, l1) = [250, 350). Both forks write the SAME buffer rows over
[l0, l1) (the newest K = 3200 rows, "P1"); all older rows are identical in C and D (shared history).
  R = D's P1 rows: restriction-era transitions (restricted learner + restricted action set, eps = 0.05)
  U = C's P1 rows: same rows, same update window, same eps (0.05), same env instances and identical env state at u250,
      same collection procedure (on-line training collection); differs only in the absence of the restriction (and
      in the clean learner that the restriction-free history produced)
Common pre-purge buffer Z (one per seed): D's buffer with the P1 rows of one half of the env columns taken from C.
  Column half H_R holds R, the other half H_U holds U (all other rows: D = C). Counterbalanced by seed parity:
  even seeds H_R = columns 0-63, odd seeds H_R = columns 64-127. Within every column the time series is that
  column's genuine history (C and D share every column's state at u250), so Z has no internal seams.
Arms (both start from the restricted fork D at removal: params, RAdam state, env state, current obs, RNG, update
counter; buffer ptr and size unchanged):
  purge_R   P1 rows of H_R overwritten with the replacement pool; U retained
  purge_U   P1 rows of H_U overwritten with the SAME pool trajectories; R retained
Replaced: K x 64 = 204,800 slots (20.5% of the buffer) per arm, in the same rows (identical ages and FIFO residence),
differing only in env column (uniform replay sampling does not depend on the column). Pool: the Phase 6b pool
(frozen clean actor = fork C params at removal, eps 0.05, full action set, key fold_in(PRNGKey(seed), 2026), 128 envs x
K steps); its env columns 0-63 are used in both arms (identical trajectories, identical sampled exposure).
Seams (identical in both arms): rows P2-end (= row before P1) and P1-end (newest row; its successor is patched to the
current D env obs) are flagged valid_next = False in ALL 128 columns (2 x 128 slots, zero loss weight until
overwritten). This covers every invalid link in either arm (P2 -> pool, pool -> current obs, U -> current obs) and
discards the same valid links in both.
Common random numbers: both arms continue the same RNG, so replay draws the same (row, column) indices in both arms.
Diagnostics: R keeps its tags (l0..l1-1, the learner's restriction window -> logged per-update sampled R fraction);
U gets tag 20000 + update (20250..20349); pool tag 32000. Shares of R / U / pool / P1 logged at every checkpoint.
Usage: python scripts/pqn_identity_swap.py --seeds 0 --tag idswap-s0 [--p6b-summary <Phase 6b summary of that seed>]
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
REPL_TAG = 32000
U_TAG_OFFSET = 20000
METRICS = ("episode_return", "loss", "q_mean", "grad_norm", "buf_frac_window", "sample_frac_window")
ARMS = ("purge_R", "purge_U")
FIELDS = ("obs", "action", "reward", "done", "tag", "valid_next")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--allowed", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--late-onset", type=int, default=250)
    ap.add_argument("--duration", type=int, default=100)
    ap.add_argument("--recovery", type=int, default=650)
    ap.add_argument("--buffer-size", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--p6b-summary", nargs="*", default=None, help="Phase 6b summaries (fork / pool consistency)")
    ap.add_argument("--tag", default="idswap")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    assert len(a.seeds) == 1, "one seed per process (memory; column assignment is per seed)"
    seed = a.seeds[0]
    E = a.eval_every
    l0, l1 = a.late_onset, a.late_onset + a.duration
    assert l0 % E == 0 and a.duration % E == 0 and a.recovery % E == 0

    label = f"pqnr-{a.buffer_size // 1000}k"
    base = dict(ENV_NAME=a.env, REPLAY=True, BUFFER_SIZE=a.buffer_size, SEAM_MASK=True, DEFICIT="action_mask",
                DEFICIT_PARAMS={"allowed_actions": a.allowed}, DEFICIT_START=0, DEFICIT_END=0)
    cfg, init, run = make(base)
    spu, N, L, T = cfg["STEPS_PER_UPDATE"], cfg["NUM_ENVS"], cfg["BUFFER_LEN"], cfg["NUM_STEPS"]
    K = a.duration * T
    assert K < L and N % 2 == 0
    H = N // 2
    cols_R = np.arange(H) if seed % 2 == 0 else np.arange(H, N)
    cols_U = np.setdiff1d(np.arange(N), cols_R)
    protocol = dict(variant=label, window_updates=[l0, l1], recovery=a.recovery, eval_every=E, eval_envs=a.eval_envs,
                    eval_seed=a.eval_seed, eval_eps=[0.0, EVAL_EPS2], allowed=a.allowed, arms=list(ARMS),
                    start_state="both arms: restricted fork D at removal (params, optimizer, env state, obs, RNG, update)",
                    Z=f"D buffer; P1 rows (newest {K}, written u{l0}-{l1 - 1}) of columns H_U taken from clean fork C",
                    cols_R=[int(cols_R[0]), int(cols_R[-1])], cols_U=[int(cols_U[0]), int(cols_U[-1])],
                    counterbalance="even seeds H_R = cols 0-63, odd seeds H_R = cols 64-127",
                    purge=dict(purge_R="P1 rows of H_R <- pool cols 0-63", purge_U="P1 rows of H_U <- pool cols 0-63",
                               slots=K * H, share=K * H / (L * N)),
                    pool=dict(policy="frozen clean actor (fork C params at removal)", eps=COLLECT_EPS, actions="full action set",
                              envs=f"{N} fresh unrestricted envs, key fold_in(PRNGKey(seed), {COLLECT_KEY_SALT}); cols 0-{H - 1} used",
                              steps=K, frozen_before_branching=True),
                    seam_rule="rows P2-end and P1-end flagged valid_next=False in all columns, both arms",
                    tags=dict(R="l0..l1-1 (learner tag window)", U=f"{U_TAG_OFFSET}+update", pool=REPL_TAG),
                    buffer_size=cfg["BUFFER_SIZE"], buffer_turnover_updates=cfg["BUFFER_SIZE"] / spu)
    manifest = build_manifest(f"p6d-{a.tag}-{label}-{a.env}", {**cfg, "protocol": protocol}, a.seeds, allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)
    summ_path = Path("results/summary") / f"{mpath.stem}.json"

    n = 1
    keys = jnp.stack([jax.random.PRNGKey(seed)])
    run_v = jax.jit(jax.vmap(lambda rs: run(rs, E)), donate_argnums=0)
    fresh = lambda r: jax.tree.map(jnp.copy, r)
    eval_g = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs, eps=0.0), in_axes=(0, None)))
    eval_e = jax.jit(jax.vmap(make_eval(cfg, num_envs=a.eval_envs, eps=EVAL_EPS2), in_axes=(0, None)))
    ek = jax.random.PRNGKey(a.eval_seed)
    ev2 = lambda rs: (np.asarray(jax.device_get(eval_g(rs["train_state"].params, ek)["ret"])).mean(1),
                      np.asarray(jax.device_get(eval_e(rs["train_state"].params, ek)["ret"])).mean(1))
    bounds = lambda b0, b1: jnp.tile(jnp.array([b0 * spu, b1 * spu], jnp.int32), (n, 1))

    def shares(r):
        tg = r["buffer"]["tag"]; valid = jnp.maximum(r["buffer"]["size"] * N, 1)
        f = lambda m: np.asarray(jax.device_get(m.sum((1, 2)) / valid))
        return (f((tg >= l0) & (tg < l1)), f((tg >= U_TAG_OFFSET + l0) & (tg < U_TAG_OFFSET + l1)), f(tg == REPL_TAG))

    t0 = time.time()
    rs = jax.jit(jax.vmap(init))(keys)
    rs["deficit_bounds"] = bounds(0, 0)
    rs["tag_window"] = jnp.tile(jnp.array([l0, l1], jnp.int32), (n, 1))
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
    ptr = int(np.asarray(forks["D"]["buffer"]["ptr"])[0])
    assert ptr == int(np.asarray(forks["C"]["buffer"]["ptr"])[0]), "forks must share the write schedule"
    assert int(np.asarray(forks["D"]["buffer"]["size"])[0]) == L, "buffer must be full at removal"
    P1 = (ptr - K + np.arange(K)) % L
    P2_end, P1_end = (ptr - K - 1) % L, (ptr - 1) % L
    older = np.setdiff1d(np.arange(L), P1)
    Cb = {f: np.asarray(forks["C"]["buffer"][f]) for f in ("obs", "action", "reward", "done", "tag")}
    Db = {f: np.asarray(forks["D"]["buffer"][f]) for f in ("obs", "action", "reward", "done", "tag")}
    # shared history: every row outside P1 is identical in C and D; P1 rows are the window rows in both
    for f in Cb:
        assert (Cb[f][:, older] == Db[f][:, older]).all(), f"rows outside P1 must be identical in C and D ({f})"
    assert ((Db["tag"][:, P1] >= l0) & (Db["tag"][:, P1] < l1)).all() and (Cb["tag"][:, P1] == Db["tag"][:, P1]).all()
    assert ((Db["tag"] >= l0) & (Db["tag"] < l1)).sum() == K * N, "restriction-era rows must be exactly P1"
    Cnet = jax.tree.map(jnp.copy, (forks["C"]["train_state"].params, forks["C"]["train_state"].opt_state))
    U_rows = {f: jnp.asarray(Cb[f][:, P1][:, :, cols_U]) for f in Cb}
    U_rows["tag"] = (U_rows["tag"] + U_TAG_OFFSET).astype(jnp.int16)
    del forks["C"], Cb
    D = dict(forks.pop("D"), deficit_bounds=bounds(0, 0))

    # ---- frozen replacement pool (identical to Phase 6b; collected before any arm branches)
    collect = jax.jit(jax.vmap(make_collector(base, eps=COLLECT_EPS, steps=K)))
    ckeys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(seed), COLLECT_KEY_SALT)])
    pool, cstats = collect(Cnet[0], ckeys)
    del Cnet
    cstats = {k: np.asarray(v).round(3).tolist() for k, v in cstats.items()}
    pool = {k: v[:, :, :H] for k, v in pool.items()}  # cols 0-63 of the Phase 6b pool, used in both arms
    print(f"replacement pool collected ({K} steps x {N} envs, cols 0-{H - 1} used): {cstats} elapsed={time.time() - t0:.0f}s", flush=True)

    # ---- common pre-purge buffer Z
    iP1, iU, iR = jnp.asarray(P1), jnp.asarray(cols_U), jnp.asarray(cols_R)
    b = D["buffer"]
    Z = {f: b[f] for f in FIELDS}
    for f in ("obs", "action", "reward", "done", "tag"):
        Z[f] = Z[f].at[:, iP1[:, None], iU[None, :]].set(U_rows[f])
    Z["valid_next"] = jnp.ones_like(b["valid_next"]).at[:, jnp.asarray([P2_end, P1_end])].set(False)
    del U_rows

    def purge(cols):
        ic = jnp.asarray(cols)
        nb = {f: jnp.copy(Z[f]) for f in FIELDS}
        for f in ("obs", "action", "reward", "done"):
            nb[f] = nb[f].at[:, iP1[:, None], ic[None, :]].set(pool[f])
        nb["tag"] = nb["tag"].at[:, iP1[:, None], ic[None, :]].set(jnp.int16(REPL_TAG))
        nb.update(ptr=jnp.copy(b["ptr"]), size=jnp.copy(b["size"]))
        return dict(jax.tree.map(jnp.copy, {k: v for k, v in D.items() if k != "buffer"}), buffer=nb)

    # ---- construction checks (numpy)
    Zn = {f: np.asarray(Z[f]) for f in FIELDS}
    for f in ("obs", "action", "reward", "done", "tag"):
        assert (Zn[f][:, older] == Db[f][:, older]).all() and (Zn[f][:, P1][:, :, cols_R] == Db[f][:, P1][:, :, cols_R]).all()
    assert ((Zn["tag"][:, P1][:, :, cols_U] >= U_TAG_OFFSET + l0) & (Zn["tag"][:, P1][:, :, cols_U] < U_TAG_OFFSET + l1)).all()
    assert int((~Zn["valid_next"]).sum()) == 2 * N
    del Db

    def check_arm(r, purged, kept):
        bb = {f: np.asarray(r["buffer"][f]) for f in FIELDS}
        po = {f: np.asarray(pool[f]) for f in ("obs", "action", "reward", "done")}
        for f in ("obs", "action", "reward", "done"):
            assert (bb[f][:, P1][:, :, purged] == po[f]).all()
        for f in FIELDS:
            zf = np.asarray(Z[f])
            assert (bb[f][:, older] == zf[:, older]).all() and (bb[f][:, P1][:, :, kept] == zf[:, P1][:, :, kept]).all()
        assert (bb["tag"][:, P1][:, :, purged] == REPL_TAG).all() and int((~bb["valid_next"]).sum()) == 2 * N
        assert int(np.asarray(r["buffer"]["ptr"])[0]) == ptr

    r4 = lambda x: np.round(np.asarray(x, float), 4).tolist()
    check = {}
    zs = shares(dict(D, buffer={**Z, "size": b["size"]}))
    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds, consistency_check=check, pool_stats=cstats,
                   blocks=dict(ptr=ptr, P1_rows_written=[int(np.asarray(b["tag"])[0, P1].min()), int(np.asarray(b["tag"])[0, P1].max())],
                               K_rows=K, cols_R=protocol["cols_R"], cols_U=protocol["cols_U"]),
                   Z_shares=dict(R=r4(zs[0]), U=r4(zs[1]), pool=r4(zs[2])),
                   fork_updates=list(range(l0, l1 + 1, E)), eval_updates=list(range(l1, l1 + a.recovery + 1, E)),
                   forks={k: np.round(v, 4).tolist() for k, v in fork_eval.items()}, arms={})
    del Zn
    print(f"Z: R share {zs[0][0]:.3f}, U share {zs[1][0]:.3f}; P1 rows {summary['blocks']['P1_rows_written']}; cols_R {protocol['cols_R']}", flush=True)

    def run_arm(name, r):
        g, x = ev2(r); s = shares(r)
        ev, eve, S, train = [g], [x], [s], []
        flags0 = int(np.asarray((~r["buffer"]["valid_next"]).sum()))
        for _ in range(a.recovery // E):
            r, m = run_v(r)
            g, x = ev2(r); s = shares(r)
            ev.append(g); eve.append(x); S.append(s)
            train.append({k: np.asarray(m[k]) for k in METRICS})
        summary["arms"][name] = dict(eval_return_mean=r4(np.stack(ev, 1)), eval_eps05_return_mean=r4(np.stack(eve, 1)),
                                     restricted_share=r4(np.stack([s[0] for s in S], 1)), U_share=r4(np.stack([s[1] for s in S], 1)),
                                     replacement_share=r4(np.stack([s[2] for s in S], 1)), seam_flags_at_start=flags0,
                                     train={k: r4(np.concatenate([t[k] for t in train], 1)) for k in METRICS})
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary))
        e = np.stack(eve, 1); c = lambda i: min(i, e.shape[1] - 1)
        sr = np.stack([s[0] for s in S], 1)[0]; su = np.stack([s[1] for s in S], 1)[0]; sp = np.stack([s[2] for s in S], 1)[0]
        # blind monitoring: no post-removal returns are printed (the primary contrast is not looked at before n = 32)
        print(f"arm {name:8s} done | R/U/pool share +0 {sr[0]:.3f}/{su[0]:.3f}/{sp[0]:.3f} +200 {sr[c(20)]:.3f}/{su[c(20)]:.3f}/{sp[c(20)]:.3f} "
              f"seams {flags0} elapsed={time.time() - t0:.0f}s", flush=True)

    for name, purged, kept in (("purge_R", cols_R, cols_U), ("purge_U", cols_U, cols_R)):
        r = purge(purged); check_arm(r, purged, kept)
        run_arm(name, r)
    del pool, D, Z

    if a.p6b_summary:
        for f in a.p6b_summary:
            d = json.loads(Path(f).read_text())
            if seed in d["seeds"]:
                j = d["seeds"].index(seed)
                check["forks_match_p6b"] = {str(seed): bool(all(np.allclose(fork_eval[k][0], np.array(d["forks"][k])[j], atol=1e-3)
                                                                for k in ("C_eps05", "D_eps05")))}
                check["pool_stats_match_p6b"] = {str(seed): bool(all(np.allclose(np.array(cstats[k])[0], np.array(d["pool_stats"][k])[j], atol=1e-3)
                                                                     for k in cstats))}
    summary["consistency_check"] = check
    summ_path.write_text(json.dumps(summary))
    print("consistency:", check, flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

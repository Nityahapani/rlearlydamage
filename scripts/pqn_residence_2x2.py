"""Phase 6c identity x residence 2x2 (PQN-R, 1M buffer). Pre-registered in LOG.md before any real run.

Completes Phase 6b (scripts/pqn_targeted_purge.py) into a 2x2 factorial. History, forks, removal state (restricted fork D
at u350), frozen replacement pool, eval protocol and seeds are IDENTICAL to Phase 6b (same code path, same keys).
Buffer positions at removal (FIFO, oldest -> newest): [rest][P2 = rows written u150-249][P1 = rows written u250-349].
Factor 1, data identity: which 41% block is removed (restriction-era R = u250-349, or pre-onset P = u150-249).
Factor 2, replacement residence: the pool occupies P1 (newest; overwritten +144..+244 = long) or P2 (+44..+144 = short).
The retained block occupies the other position. Cells (contents of P2 / P1):
  B_targeted        P2 = P (u150-249),  P1 = pool      R removed, long    (Phase 6b arm; reused)
  Bp_targeted_short P2 = pool,          P1 = P         R removed, short   (NEW: P block moved to the newest rows)
  Cp_sham_long      P2 = R (u250-349),  P1 = pool      P removed, long    (NEW: R block moved to the older rows)
  C_sham            P2 = pool,          P1 = R         P removed, short   (Phase 6b arm; reused)
B vs Cp: same replacement position AND same retained-block position; they differ only in whether the retained block is
restriction-era or pre-onset data (residence-matched identity test). Bp vs C: same, at short replacement residence.
Moved blocks keep their per-env time order and all fields (obs, action, reward, done, tag, valid_next); ptr and size
are unchanged. Successor seams in the new cells (3 links per env, valid_next = False, zero loss weight until
overwritten): row before P2 (rest -> new P2 content), last row of P2 (P2 -> new P1 content), last row of P1 (newest
row; its successor is patched to the current env obs, valid only for the original u349 data). B and C (Phase 6b) have
2 flags per env; the new cells 3 per env (0.038% of slots).
Arms run here: Bp_targeted_short, Cp_sham_long (+ B_targeted with --check-b, compared against Phase 6b for bit identity).
Consistency (vs the Phase 6b summary of the same seed): forks C/D eval traces, pool collection stats, and B_targeted.
Usage: python scripts/pqn_residence_2x2.py --seeds 0 --tag res2x2-s0 --p6b-summary results/summary/*p6b-tpurge-s0-* [--check-b]
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
METRICS = ("episode_return", "loss", "q_mean", "grad_norm", "buf_frac_window", "sample_frac_window")
ARMS = ("Bp_targeted_short", "Cp_sham_long")


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
    ap.add_argument("--p6b-summary", nargs="*", default=None, help="Phase 6b summaries (fork / pool / B consistency)")
    ap.add_argument("--check-b", action="store_true", help="also rerun B_targeted and compare with Phase 6b")
    ap.add_argument("--tag", default="res2x2")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    E = a.eval_every
    l0, l1 = a.late_onset, a.late_onset + a.duration
    assert l0 % E == 0 and a.duration % E == 0 and a.recovery % E == 0

    label = f"pqnr-{a.buffer_size // 1000}k"
    base = dict(ENV_NAME=a.env, REPLAY=True, BUFFER_SIZE=a.buffer_size, SEAM_MASK=True, DEFICIT="action_mask",
                DEFICIT_PARAMS={"allowed_actions": a.allowed}, DEFICIT_START=0, DEFICIT_END=0)
    cfg, init, run = make(base)
    spu, N, L, T = cfg["STEPS_PER_UPDATE"], cfg["NUM_ENVS"], cfg["BUFFER_LEN"], cfg["NUM_STEPS"]
    K = a.duration * T  # rows written during the restriction window
    assert 2 * K < L, "restriction block + sham block must fit in the buffer"
    protocol = dict(variant=label, window_updates=[l0, l1], recovery=a.recovery, eval_every=E, eval_envs=a.eval_envs,
                    eval_seed=a.eval_seed, eval_eps=[0.0, EVAL_EPS2], allowed=a.allowed,
                    arms=list(ARMS) + (["B_targeted"] if a.check_b else []),
                    start_state="all arms: restricted fork D at removal; D_cleannet swaps in params + optimizer state of clean fork C",
                    layout_P2_P1=dict(B_targeted=["P", "pool"], Bp_targeted_short=["pool", "P"],
                                      Cp_sham_long=["R", "pool"], C_sham=["pool", "R"],
                                      P=f"rows written u{l0 - a.duration}-{l0 - 1}", R=f"rows written u{l0}-{l1 - 1}",
                                      transitions_per_block=K * N, share=K / L),
                    pool=dict(policy="frozen clean actor (fork C params at removal)", eps=COLLECT_EPS, actions="full action set",
                              envs=f"{N} fresh unrestricted envs, key fold_in(PRNGKey(seed), {COLLECT_KEY_SALT})",
                              steps=K, frozen_before_branching=True, shared="identical pool in B, C, D"),
                    seam_rule="new cells: rows (P2 start - 1), P2 end, P1 end per env -> valid_next=False, zero loss weight; B_targeted as Phase 6b (2 rows)",
                    replacement_tag=REPL_TAG, buffer_size=cfg["BUFFER_SIZE"], buffer_turnover_updates=cfg["BUFFER_SIZE"] / spu)
    manifest = build_manifest(f"p6c-{a.tag}-{label}-{a.env}", {**cfg, "protocol": protocol}, a.seeds, allow_dirty=a.allow_dirty)
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

    def shares(r):
        tg = r["buffer"]["tag"]; valid = jnp.maximum(r["buffer"]["size"] * N, 1)
        rs_ = ((tg >= l0) & (tg < l1)).sum((1, 2)) / valid
        rp_ = (tg == REPL_TAG).sum((1, 2)) / valid
        return np.asarray(jax.device_get(rs_)), np.asarray(jax.device_get(rp_))

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
    Cnet = jax.tree.map(jnp.copy, (forks["C"]["train_state"].params, forks["C"]["train_state"].opt_state))
    del forks["C"]
    D = dict(forks.pop("D"), deficit_bounds=bounds(0, 0))

    # ---- frozen replacement pool (collected once, before any learner arm branches)
    collect = jax.jit(jax.vmap(make_collector(base, eps=COLLECT_EPS, steps=K)))
    ckeys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(s), COLLECT_KEY_SALT) for s in a.seeds])
    pool, cstats = collect(Cnet[0], ckeys)
    cstats = {k: np.asarray(v).round(3).tolist() for k, v in cstats.items()}
    print(f"replacement pool collected ({K} steps x {N} envs): {cstats} elapsed={time.time() - t0:.0f}s", flush=True)

    # ---- row blocks (the write pointer is identical across seeds: same schedule)
    ptr = np.asarray(D["buffer"]["ptr"]); assert (ptr == ptr[0]).all(); ptr = int(ptr[0])
    assert int(np.asarray(D["buffer"]["size"])[0]) == L, "buffer must be full at removal"
    blk_B = (ptr - K + np.arange(K)) % L
    blk_C = (ptr - 2 * K + np.arange(K)) % L
    tg = np.asarray(D["buffer"]["tag"])
    inw = (tg >= l0) & (tg < l1)
    assert inw[:, blk_B].all() and inw.sum() == n * K * N, "restriction-era rows must be exactly the newest K rows"
    assert ((tg[:, blk_C] >= l0 - a.duration) & (tg[:, blk_C] < l0)).all(), "sham block must be the adjacent older block"
    age = lambda blk: [int(tg[0, blk].min()), int(tg[0, blk].max())]
    print(f"blocks: B rows written u{age(blk_B)}, C rows written u{age(blk_C)}; ptr={ptr}, L={L}", flush=True)

    def splice(r, blk):
        b = r["buffer"]; idx = jnp.asarray(blk)
        seam = jnp.asarray([(blk[0] - 1) % L, blk[-1]])
        nb = dict(obs=b["obs"].at[:, idx].set(pool["obs"]), action=b["action"].at[:, idx].set(pool["action"]),
                  reward=b["reward"].at[:, idx].set(pool["reward"]), done=b["done"].at[:, idx].set(pool["done"]),
                  tag=b["tag"].at[:, idx].set(jnp.int16(REPL_TAG)),
                  valid_next=b["valid_next"].at[:, idx].set(True).at[:, seam].set(False),
                  ptr=jnp.copy(b["ptr"]), size=jnp.copy(b["size"]))
        return dict(jax.tree.map(jnp.copy, {k: v for k, v in r.items() if k != "buffer"}), buffer=nb)

    def layout(r, p2, p1):
        """Rewrite positions P2 (= blk_C) and P1 (= blk_B) with contents in {"pool", "P", "R"} taken from the ORIGINAL
        removal buffer (P = original P2 rows, R = original P1 rows); everything else (rest rows, ptr, size) unchanged."""
        b = r["buffer"]; i2, i1 = jnp.asarray(blk_C), jnp.asarray(blk_B)
        src_rows = {"P": i2, "R": i1}
        def get(field, what):
            if what == "pool":
                return {"tag": jnp.full((n, K, N), REPL_TAG, jnp.int16), "valid_next": jnp.ones((n, K, N), bool)}.get(field, pool.get(field))
            return b[field][:, src_rows[what]]
        seam = jnp.asarray([(blk_C[0] - 1) % L, blk_C[-1], blk_B[-1]])
        nb = {f: b[f].at[:, i2].set(get(f, p2)).at[:, i1].set(get(f, p1)) for f in ("obs", "action", "reward", "done", "tag", "valid_next")}
        nb["valid_next"] = nb["valid_next"].at[:, seam].set(False)
        nb.update(ptr=jnp.copy(b["ptr"]), size=jnp.copy(b["size"]))
        return dict(jax.tree.map(jnp.copy, {k: v for k, v in r.items() if k != "buffer"}), buffer=nb)

    def check_layout(r, p2, p1):
        tg_ = np.asarray(r["buffer"]["tag"]); ob = np.asarray(r["buffer"]["obs"]); ob0 = np.asarray(D["buffer"]["obs"])
        po = np.asarray(pool["obs"])
        for blk, what in ((blk_C, p2), (blk_B, p1)):
            if what == "pool":
                assert (tg_[:, blk] == REPL_TAG).all() and (ob[:, blk] == po).all()
            else:
                srcb = blk_C if what == "P" else blk_B
                assert (tg_[:, blk] == tg[:, srcb]).all() and (ob[:, blk] == ob0[:, srcb]).all()
        rest = np.setdiff1d(np.arange(L), np.concatenate([blk_B, blk_C]))
        assert (tg_[:, rest] == tg[:, rest]).all() and (ob[:, rest] == ob0[:, rest]).all()
        assert int((~np.asarray(r["buffer"]["valid_next"])).sum()) == 3 * N * n

    def with_clean_net(r):
        ts = r["train_state"].replace(params=jax.tree.map(jnp.copy, Cnet[0]), opt_state=jax.tree.map(jnp.copy, Cnet[1]))
        return dict(r, train_state=ts)

    r4 = lambda x: np.round(np.asarray(x, float), 4).tolist()
    check = {}
    summary = dict(manifest=mpath.name, protocol=protocol, seeds=a.seeds, consistency_check=check, pool_stats=cstats,
                   blocks=dict(ptr=ptr, B_rows_written=age(blk_B), C_rows_written=age(blk_C), K_rows=K),
                   fork_updates=list(range(l0, l1 + 1, E)), eval_updates=list(range(l1, l1 + a.recovery + 1, E)),
                   forks={k: np.round(v, 4).tolist() for k, v in fork_eval.items()}, arms={})

    def run_arm(name, r):
        g, x = ev2(r); s_r, s_p = shares(r)
        ev, eve, sr, sp, train = [g], [x], [s_r], [s_p], []
        flags0 = int(np.asarray((~r["buffer"]["valid_next"]).sum()))
        for _ in range(a.recovery // E):
            r, m = run_v(r)
            g, x = ev2(r); s_r, s_p = shares(r)
            ev.append(g); eve.append(x); sr.append(s_r); sp.append(s_p)
            train.append({k: np.asarray(m[k]) for k in METRICS})
        summary["arms"][name] = dict(eval_return_mean=r4(np.stack(ev, 1)), eval_eps05_return_mean=r4(np.stack(eve, 1)),
                                     restricted_share=r4(np.stack(sr, 1)), replacement_share=r4(np.stack(sp, 1)),
                                     seam_flags_at_start=flags0,
                                     train={k: r4(np.concatenate([t[k] for t in train], 1)) for k in METRICS})
        summary["wallclock_s"] = round(time.time() - t0, 1)
        summ_path.write_text(json.dumps(summary))
        e = np.stack(eve, 1); S_r = np.stack(sr, 1).mean(0); S_p = np.stack(sp, 1).mean(0)
        c = lambda i: min(i, e.shape[1] - 1)  # checkpoint index, clipped for short smoke runs
        print(f"arm {name:10s} (eps .05): +0 {e[:, 0].mean():5.1f} +10 {e[:, 1].mean():5.1f} +100 {e[:, c(10)].mean():5.1f} "
              f"end {e[:, -1].mean():5.1f} AUC {e[:, 1:].mean():5.1f} | restricted +0/+100/+200 {S_r[0]:.2f}/{S_r[c(10)]:.2f}/{S_r[c(20)]:.2f} "
              f"repl +0/+100/+200 {S_p[0]:.2f}/{S_p[c(10)]:.2f}/{S_p[c(20)]:.2f} seams {flags0} elapsed={time.time() - t0:.0f}s", flush=True)

    del Cnet
    for name, (p2, p1) in (("Bp_targeted_short", ("pool", "P")), ("Cp_sham_long", ("R", "pool"))):
        r = layout(D, p2, p1); check_layout(r, p2, p1)
        run_arm(name, r)
    if a.check_b:
        run_arm("B_targeted", splice(D, blk_B))
    del pool, D

    if a.p6b_summary:
        for f in a.p6b_summary:
            d = json.loads(Path(f).read_text())
            for s in d["seeds"]:
                if s in a.seeds:
                    i, j = a.seeds.index(s), d["seeds"].index(s)
                    check.setdefault("forks_match_p6b", {})[str(s)] = bool(all(
                        np.allclose(fork_eval[k][i], np.array(d["forks"][k])[j], atol=1e-3) for k in ("C_eps05", "D_eps05")))
                    check.setdefault("pool_stats_match_p6b", {})[str(s)] = bool(all(
                        np.allclose(np.array(cstats[k])[i], np.array(d["pool_stats"][k])[j], atol=1e-3) for k in cstats))
                    check.setdefault("blocks_match_p6b", {})[str(s)] = summary["blocks"] == d["blocks"]
                    if a.check_b:
                        check.setdefault("B_targeted_matches_p6b", {})[str(s)] = bool(all(
                            np.allclose(np.array(summary["arms"]["B_targeted"][m])[i], np.array(d["arms"]["B_targeted"][m])[j], atol=1e-3)
                            for m in ("eval_eps05_return_mean", "eval_return_mean", "replacement_share")))
    summary["consistency_check"] = check
    summ_path.write_text(json.dumps(summary))
    print("consistency:", check, flush=True)
    print("summary:", summ_path, flush=True)


if __name__ == "__main__":
    main()

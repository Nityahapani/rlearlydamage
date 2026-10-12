"""Phase 6b targeted vs sham replay purge (PQN-R, 1M buffer). Pre-registered in LOG.md before any real run.

History per seed (identical to the Phase 6 1M surgery / purge runs): shared 0 -> l0 unrestricted; fork C: l0 -> l1
unrestricted; fork D: l0 -> l1 movement-restricted ({noop, fire}); removal at l1 = u350.

Every arm starts from the SAME saved removal state: the restricted fork D (params, RAdam state, env state, current obs,
RNG, update counter, buffer). Only the listed component is changed:
  A_retain     nothing (continue normal training with the original buffer)
  B_targeted   the K restriction-era buffer rows (written in updates [l0, l1); the newest K rows) are overwritten with
               the frozen replacement pool; all other rows are preserved
  C_sham       the K non-restriction-era rows closest in age to the restriction-era block (the adjacent block, written
               in updates [l0 - dur, l0)) are overwritten with the SAME pool; restriction-era rows are left intact
  D_cleannet   as B_targeted (identical buffer), but params + optimizer state from the clean fork C
K = dur x NUM_STEPS rows x NUM_ENVS envs = 3200 x 128 = 409,600 transitions (41% of the buffer).

Replacement pool (frozen before any learner arm branches; one per seed; identical in B, C, D):
  frozen clean actor (fork C params at l1), eps-greedy eps = 0.05 over the full action set, 128 fresh unrestricted
  envs (training wrappers), key fold_in(PRNGKey(seed), 2026) (= the first 3200 steps of the Phase 6 purge pool),
  K/128 = 3200 time-ordered steps per env, written in time order into the replaced rows.
Controlled: buffer size, pointer and size (unchanged), slot schema; replaced slots get tag 32000 (diagnostics only).
Successor continuity: the buffer stores s' as the next row of the same env. A spliced block creates two invalid
successor links per env (the row before the block and the block's last row); these 2 x 128 transitions (0.026%) are
flagged (valid_next = False, SEAM_MASK) and get zero loss weight until overwritten. The same code path (SEAM_MASK on)
is used by every arm; with no flags it is bit-identical to the unmasked agent (checked).
Age-matching limit: the restriction-era rows are the newest K rows, so no non-restriction-era row has the same age.
The sham uses the closest possible block (ages 100-199 updates vs 0-99); under FIFO its replacement data is
overwritten ~100 updates earlier (+44..+144 vs +144..+244 after removal). Logged at every eval checkpoint:
restriction-era share and replacement share of each buffer.
Usage: python scripts/pqn_targeted_purge.py --seeds 0 --tag tpurge-s0 [--purge-summary ...] [--surgery-summary ...]
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
ARMS = ("D_cleannet", "B_targeted", "C_sham", "A_retain")  # run order (frees memory)


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
    ap.add_argument("--purge-summary", nargs="*", default=None, help="Phase 6 purge summaries (A_retain consistency)")
    ap.add_argument("--surgery-summary", nargs="*", default=None, help="Phase 6 1M surgery summaries (fork consistency)")
    ap.add_argument("--tag", default="tpurge")
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
                    eval_seed=a.eval_seed, eval_eps=[0.0, EVAL_EPS2], allowed=a.allowed, arms=list(ARMS),
                    start_state="all arms: restricted fork D at removal; D_cleannet swaps in params + optimizer state of clean fork C",
                    replaced_rows=dict(B_targeted=f"newest {K} rows (written u{l0}-{l1 - 1})",
                                       C_sham=f"adjacent older {K} rows (written u{l0 - a.duration}-{l0 - 1})",
                                       transitions=K * N, share=K / L),
                    pool=dict(policy="frozen clean actor (fork C params at removal)", eps=COLLECT_EPS, actions="full action set",
                              envs=f"{N} fresh unrestricted envs, key fold_in(PRNGKey(seed), {COLLECT_KEY_SALT})",
                              steps=K, frozen_before_branching=True, shared="identical pool in B, C, D"),
                    seam_rule="2 rows x NUM_ENVS successor links invalidated per spliced block -> valid_next=False, zero loss weight",
                    replacement_tag=REPL_TAG, buffer_size=cfg["BUFFER_SIZE"], buffer_turnover_updates=cfg["BUFFER_SIZE"] / spu)
    manifest = build_manifest(f"p6b-{a.tag}-{label}-{a.env}", {**cfg, "protocol": protocol}, a.seeds, allow_dirty=a.allow_dirty)
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
        print(f"arm {name:10s} (eps .05): +0 {e[:, 0].mean():5.1f} +10 {e[:, 1].mean():5.1f} +100 {e[:, 10].mean():5.1f} "
              f"end {e[:, -1].mean():5.1f} AUC {e[:, 1:].mean():5.1f} | restricted +0/+100/+200 {S_r[0]:.2f}/{S_r[10]:.2f}/{S_r[20]:.2f} "
              f"repl +0/+100/+200 {S_p[0]:.2f}/{S_p[10]:.2f}/{S_p[20]:.2f} seams {flags0} elapsed={time.time() - t0:.0f}s", flush=True)

    run_arm("D_cleannet", with_clean_net(splice(D, blk_B)))
    del Cnet
    run_arm("B_targeted", splice(D, blk_B))
    run_arm("C_sham", splice(D, blk_C))
    del pool
    run_arm("A_retain", D)
    del D

    if a.purge_summary:
        for f in a.purge_summary:
            d = json.loads(Path(f).read_text())
            for s in d["seeds"]:
                if s in a.seeds:
                    i, j = a.seeds.index(s), d["seeds"].index(s)
                    ok = np.allclose(np.array(summary["arms"]["A_retain"]["eval_eps05_return_mean"])[i],
                                     np.array(d["arms"]["A_retain"]["eval_eps05_return_mean"])[j], atol=1e-3)
                    check.setdefault("A_retain_matches_phase6_purge", {})[str(s)] = bool(ok)
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

"""Phase 2 critical-period sweep: one deficit type, several (onset, duration) windows, vmapped seeds.

The deficit window lives in the runner state, so the training step compiles once for the whole sweep.
Each window is trained from scratch with the same seeds, so it can be compared with a no-deficit control
that uses the same config and seeds.

Usage:
  python scripts/sweep_windows.py --env SpaceInvaders-MinAtar --deficit obs_noise \
      --onsets 0 0.1 0.2 0.3 0.4 --duration 0.2 --steps 2000000 --seeds 0 1 2 3 [--control]

Outputs:
  runs/<stem>.json                 manifest (committed)
  results/<stem>/curves.npz        full per-update curves, windows x seeds x updates (git-ignored)
  results/summary/<stem>.json      small summary (committed)
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
import yaml

from rled.agents.ppo import make
from rled.deficits import DeficitWindow
from rled.provenance import build_manifest, write_manifest

warnings.filterwarnings("ignore")

RECOVERY_FRAC = 0.4  # "equal recovery" metric: return measured this long after the deficit is removed


def summarize_window(er: np.ndarray, steps: np.ndarray, end_step: int, total: int) -> dict:
    """er: seeds x updates. Returns final return and the return at a fixed time after removal."""
    n = er.shape[1]
    last = max(1, n // 10)
    final = np.nanmean(er[:, -last:], axis=1)
    t_eq = end_step + int(RECOVERY_FRAC * total)
    i = int(np.searchsorted(steps, t_eq))
    lo, hi = max(0, i - last // 2), min(n, i + last // 2 + 1)
    eq = np.nanmean(er[:, lo:hi], axis=1) if i < n else np.full(er.shape[0], np.nan)
    r3 = lambda x: [None if np.isnan(v) else round(float(v), 3) for v in x]
    return dict(final_return=r3(final), equal_recovery_return=r3(eq),
                equal_recovery_step=t_eq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="SpaceInvaders-MinAtar")
    ap.add_argument("--deficit", required=True, choices=["obs_noise", "action_mask", "reward_delay", "reward_flip"])
    ap.add_argument("--params", default=None, help="YAML dict overriding configs/deficits/<deficit>.yaml params")
    ap.add_argument("--onsets", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--duration", type=float, default=0.2)
    ap.add_argument("--control", action="store_true", help="also run a no-deficit control (empty window)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    dparams = yaml.safe_load(Path(f"configs/deficits/{a.deficit}.yaml").read_text())["deficit"]["params"]
    if a.params:
        dparams.update(yaml.safe_load(a.params))
    overrides = dict(ENV_NAME=a.env, TOTAL_TIMESTEPS=a.steps, HIDDEN=a.hidden, DEFICIT=a.deficit,
                     DEFICIT_PARAMS=dparams)
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    cfg, init, run = make(overrides)

    windows = [DeficitWindow(a.deficit, o, a.duration, dparams) for o in a.onsets]
    labels = [f"on{w.onset_frac:g}_dur{w.duration_frac:g}" for w in windows]
    bounds = [w.bounds(a.steps) for w in windows]
    if a.control:
        labels.append("control")
        bounds.append((0, 0))

    name = f"p2-{a.tag}-{a.deficit}-ppo-{a.env}"
    sweep = dict(deficit=a.deficit, params=dparams, onsets=a.onsets, duration=a.duration,
                 windows=dict(zip(labels, [list(b) for b in bounds])), recovery_frac=RECOVERY_FRAC)
    manifest = build_manifest(name, {**cfg, "sweep": sweep}, a.seeds, allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)

    per_chunk = cfg["NUM_UPDATES"] // a.chunks
    f = jax.jit(jax.vmap(lambda rs: run(rs, per_chunk)))
    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    t0 = time.time()
    all_curves, summary = {}, {}
    for lab, (b0, b1) in zip(labels, bounds):
        rs = jax.vmap(init)(keys)
        rs["deficit_bounds"] = jnp.tile(jnp.array([b0, b1], jnp.int32), (len(a.seeds), 1))
        chunks = []
        for c in range(a.chunks):
            rs, m = f(rs)
            chunks.append(jax.device_get(m))
        cur = {k: np.concatenate([cc[k] for cc in chunks], axis=1) for k in chunks[0]}
        all_curves[lab] = cur
        steps = cur["step"][0]
        end_step = b1 if b1 > b0 else int(a.onsets[0] * a.steps + a.duration * a.steps)
        summary[lab] = dict(bounds=[b0, b1], deficit_updates=int(cur["deficit_active"][0].sum()),
                            **summarize_window(cur["episode_return"], steps, end_step, a.steps))
        print(f"{lab:>16} bounds={b0:>9,}-{b1:<9,} final/seed={summary[lab]['final_return']} "
              f"eqrec/seed={summary[lab]['equal_recovery_return']} elapsed={time.time()-t0:.0f}s", flush=True)

    out = Path("results") / mpath.stem
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "curves.npz",
                        **{f"{lab}__{k}": v for lab, cur in all_curves.items() for k, v in cur.items()})
    k = 10
    small = {}
    for lab, cur in all_curves.items():
        er = cur["episode_return"]
        w = er[:, : er.shape[1] // k * k].reshape(er.shape[0], -1, k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small[lab] = [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in np.nanmean(w, axis=2)]
    steps_ds = all_curves[labels[0]]["step"][0, k - 1 :: k][: len(small[labels[0]][0])].tolist()
    summ = Path("results/summary") / f"{mpath.stem}.json"
    summ.write_text(json.dumps(dict(manifest=mpath.name, sweep=sweep, seeds=a.seeds, windows=summary,
                                    curve_steps=steps_ds, curves=small,
                                    wallclock_s=round(time.time() - t0, 1)), indent=1))
    print("summary:", summ, flush=True)


if __name__ == "__main__":
    main()

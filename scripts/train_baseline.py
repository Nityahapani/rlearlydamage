"""Phase 1 baseline: PPO with no deficit, several seeds vmapped together; saves learning curves.

Usage: python scripts/train_baseline.py --env Breakout-MinAtar --seeds 0 1 2 3 --steps 5000000
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import jax
import yaml
import numpy as np

from rled.agents.ppo import make
from rled.provenance import build_manifest, write_manifest

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Breakout-MinAtar")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra PPO config overrides, e.g. GAMMA=0.999 LAYERNORM=True")
    a = ap.parse_args()

    overrides = dict(ENV_NAME=a.env, TOTAL_TIMESTEPS=a.steps, HIDDEN=a.hidden)
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    cfg, init, run = make(overrides)
    name = f"p1-{a.tag}-ppo-{a.env}"
    manifest = build_manifest(name, {k: v for k, v in cfg.items()}, a.seeds, allow_dirty=a.allow_dirty)
    mpath = write_manifest(manifest)
    print("manifest:", mpath, flush=True)

    n_total = cfg["NUM_UPDATES"]
    per_chunk = n_total // a.chunks
    f = jax.jit(jax.vmap(lambda rs: run(rs, per_chunk)))
    rs = jax.vmap(init)(jax.numpy.stack([jax.random.PRNGKey(s) for s in a.seeds]))
    curves = []
    t0 = time.time()
    for c in range(a.chunks):
        rs, m = f(rs)
        m = jax.device_get(m)
        curves.append(m)
        ret = np.nanmean(m["episode_return"][:, -10:], axis=1)
        print(f"chunk {c+1}/{a.chunks} step={int(m['step'][0,-1]):,} "
              f"return/seed={np.round(ret,1)} elapsed={time.time()-t0:.0f}s", flush=True)
    curves = {k: np.concatenate([cc[k] for cc in curves], axis=1) for k in curves[0]}

    out = Path("results") / mpath.stem
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "curves.npz", **curves)
    # small committed summary: return curve averaged over windows of 10 updates
    summ = Path("results/summary") / f"{mpath.stem}.json"
    er = curves["episode_return"]
    k = 10
    w = er[:, : er.shape[1] // k * k].reshape(er.shape[0], -1, k)
    summ.write_text(json.dumps({
        "manifest": mpath.name,
        "steps": curves["step"][0, k - 1 :: k][: w.shape[1]].tolist(),
        "return_per_seed": np.round(np.nanmean(w, axis=2), 3).tolist(),
        "final_return_per_seed": np.round(np.nanmean(er[:, -20:], axis=1), 3).tolist(),
        "wallclock_s": round(time.time() - t0, 1),
    }, indent=1))
    print("summary:", summ, flush=True)


if __name__ == "__main__":
    main()

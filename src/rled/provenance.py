"""Run manifests: git commit, config, seed, and environment for every run."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

TRACKED_PACKAGES = ("jax", "jaxlib", "flax", "optax", "gymnax", "numpy", "rliable")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def git_state() -> dict:
    try:
        return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def package_versions() -> dict:
    out = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def build_manifest(name: str, config: dict, seeds: list[int], allow_dirty: bool = False) -> dict:
    g = git_state()
    if g["dirty"] and not allow_dirty:
        raise RuntimeError("working tree is dirty; commit first or pass allow_dirty=True")
    return {
        "name": name,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": g,
        "allow_dirty": allow_dirty,
        "seeds": list(seeds),
        "config": config,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def write_manifest(manifest: dict, runs_dir: str | Path = "runs") -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    day = manifest["created_utc"][:10].replace("-", "")
    sha = (manifest["git"]["commit"] or "nogit")[:7]
    path = runs_dir / f"{day}-{manifest['name']}-{sha}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path

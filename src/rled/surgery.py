"""State surgery: build hybrid trainer states from a damaged run and a clean run.

A trainer state has three carriers:
  - "params":    network parameters (weights)
  - "opt_state": optimizer state (Adam m, v, step count)
  - "data":      replay buffer (off-policy) or behaviour-policy params (on-policy)

At the deficit-removal step we checkpoint both runs (same seed, same step) and create all
2^3 = 8 combinations. Example: code "DCC" = damaged params, clean optimizer, clean data.
"""
from __future__ import annotations

import copy
from itertools import product

CARRIERS = ("params", "opt_state", "data")


def all_codes() -> list[str]:
    """['CCC', 'CCD', ..., 'DDD'], one letter per carrier in CARRIERS order."""
    return ["".join(c) for c in product("CD", repeat=len(CARRIERS))]


def compose(clean: dict, damaged: dict, code: str) -> dict:
    """Return a new trainer state taking each carrier from `clean` (C) or `damaged` (D).

    Any extra keys (e.g. env state, RNG) are copied from `clean`, so the post-surgery future
    is held fixed and only the three carriers differ.
    """
    if len(code) != len(CARRIERS) or set(code) - {"C", "D"}:
        raise ValueError(f"bad surgery code {code!r}")
    for k in CARRIERS:
        if k not in clean or k not in damaged:
            raise KeyError(f"both states must contain carrier {k!r}")
    out = copy.deepcopy(clean)
    for carrier, src in zip(CARRIERS, code):
        out[carrier] = copy.deepcopy(damaged[carrier] if src == "D" else clean[carrier])
    return out


def main_effects(scores: dict[str, float]) -> dict[str, float]:
    """Main effect of damaging each carrier, averaged over the other carriers (2^3 factorial).

    effect[c] = mean(score | c damaged) - mean(score | c clean). Negative means harm.
    """
    if set(scores) != set(all_codes()):
        raise ValueError("need scores for all 8 surgery codes")
    eff = {}
    for i, c in enumerate(CARRIERS):
        dmg = [v for k, v in scores.items() if k[i] == "D"]
        cln = [v for k, v in scores.items() if k[i] == "C"]
        eff[c] = sum(dmg) / len(dmg) - sum(cln) / len(cln)
    return eff

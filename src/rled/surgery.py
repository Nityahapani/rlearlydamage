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


def all_codes(carriers=CARRIERS) -> list[str]:
    """['CCC', 'CCD', ..., 'DDD'], one letter per carrier in `carriers` order."""
    return ["".join(c) for c in product("CD", repeat=len(carriers))]


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


def main_effects(scores: dict[str, float], carriers=CARRIERS) -> dict[str, float]:
    """Main effect of damaging each carrier, averaged over the other carriers (2^3 factorial).

    effect[c] = mean(score | c damaged) - mean(score | c clean). Negative means harm.
    """
    if set(scores) != set(all_codes(carriers)):
        raise ValueError(f"need scores for all {2 ** len(carriers)} surgery codes")
    eff = {}
    for i, c in enumerate(carriers):
        dmg = [v for k, v in scores.items() if k[i] == "D"]
        cln = [v for k, v in scores.items() if k[i] == "C"]
        eff[c] = sum(dmg) / len(dmg) - sum(cln) / len(cln)
    return eff


# --- PPO adapter -------------------------------------------------------------------------------
# On-policy PPO has no replay buffer, so the factorial carriers are actor weights, critic weights and
# optimizer state. The "data" carrier is tested separately via a behaviour-policy swap (see ppo.py).
PPO_CARRIERS = ("actor", "critic", "opt")


def compose_ppo(clean: dict, damaged: dict, code: str) -> dict:
    """Hybrid PPO runner state. code[i] in {C, D} selects the source of PPO_CARRIERS[i].

    Everything else (env state, observations, RNG, update counter, deficit bounds) is taken from
    `clean`, so the post-surgery future randomness is identical across arms. Works on vmapped states.
    """
    from rled.agents.ppo import ACTOR_KEYS, CRITIC_KEYS

    if len(code) != 3 or set(code) - {"C", "D"}:
        raise ValueError(f"bad surgery code {code!r}")
    src = {c: (damaged if x == "D" else clean) for c, x in zip(PPO_CARRIERS, code)}
    ts_c = clean["train_state"]
    p = {k: (src["actor"] if k in ACTOR_KEYS else src["critic"])["train_state"].params["params"][k]
         for k in ts_c.params["params"]}
    assert set(p) == set(ACTOR_KEYS) | set(CRITIC_KEYS)
    ts = ts_c.replace(params={"params": p}, opt_state=src["opt"]["train_state"].opt_state)
    return dict(clean, train_state=ts)

"""Deficit windows.

A deficit is active for env steps t with onset <= t < onset + duration (Achille-style sliding window).
The transforms here are framework-agnostic definitions. Agents apply them inside their jitted step
using `is_active` as a traced boolean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

DEFICIT_TYPES = ("none", "obs_noise", "action_mask", "reward_delay", "reward_flip")


@dataclass(frozen=True)
class DeficitWindow:
    type: str = "none"
    onset_frac: float = 0.0
    duration_frac: float = 0.0
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in DEFICIT_TYPES:
            raise ValueError(f"unknown deficit type {self.type!r}; expected one of {DEFICIT_TYPES}")
        if not (0.0 <= self.onset_frac <= 1.0 and 0.0 <= self.duration_frac <= 1.0):
            raise ValueError("onset_frac and duration_frac must be in [0, 1]")
        if self.onset_frac + self.duration_frac > 1.0 + 1e-9:
            raise ValueError("deficit window extends past the end of training")

    def bounds(self, total_steps: int) -> tuple[int, int]:
        start = int(round(self.onset_frac * total_steps))
        end = int(round((self.onset_frac + self.duration_frac) * total_steps))
        return start, end

    def is_active(self, step: int, total_steps: int) -> bool:
        if self.type == "none" or self.duration_frac == 0.0:
            return False
        start, end = self.bounds(total_steps)
        return start <= step < end

    @property
    def removal_frac(self) -> float:
        """Point at which the deficit is removed, which is where state surgery happens."""
        return self.onset_frac + self.duration_frac


def window_grid(onsets, durations, dtype: str, params: dict | None = None) -> list[DeficitWindow]:
    """All valid (onset, duration) windows for a sensitivity sweep."""
    out = []
    for o, d in product(onsets, durations):
        if o + d <= 1.0 + 1e-9:
            out.append(DeficitWindow(dtype, o, d, params or {}))
    return out

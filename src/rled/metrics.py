"""Plasticity diagnostics computed on activations / features (numpy, framework-agnostic)."""
from __future__ import annotations

import numpy as np


def dormant_ratio(activations: np.ndarray, tau: float = 0.025) -> float:
    """Fraction of units whose normalised mean |activation| <= tau (Sokar et al., 2023).

    activations: [batch, units] for one layer.
    """
    a = np.abs(np.asarray(activations)).mean(axis=0)
    score = a / (a.mean() + 1e-8)
    return float((score <= tau).mean())


def effective_rank(features: np.ndarray, delta: float = 0.01) -> int:
    """srank_delta (Kumar et al., 2021): the smallest k whose top-k singular values hold (1 - delta) of the mass."""
    s = np.linalg.svd(np.asarray(features), compute_uv=False)
    c = np.cumsum(s) / (s.sum() + 1e-12)
    return int(np.searchsorted(c, 1.0 - delta) + 1)

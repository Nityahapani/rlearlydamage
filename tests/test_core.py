import numpy as np
import pytest

from rled.deficits import DeficitWindow, window_grid
from rled.metrics import dormant_ratio, effective_rank
from rled.surgery import all_codes, compose, main_effects


def test_window_activity():
    w = DeficitWindow("obs_noise", 0.1, 0.2)
    assert w.bounds(1000) == (100, 300)
    assert not w.is_active(99, 1000) and w.is_active(100, 1000) and not w.is_active(300, 1000)
    assert DeficitWindow().is_active(5, 1000) is False


def test_window_validation():
    with pytest.raises(ValueError):
        DeficitWindow("obs_noise", 0.9, 0.2)
    with pytest.raises(ValueError):
        DeficitWindow("bogus", 0.0, 0.1)


def test_grid_drops_invalid():
    g = window_grid([0.0, 0.9], [0.05, 0.2], "obs_noise")
    assert len(g) == 3


def test_surgery_compose():
    clean = {"params": "Pc", "opt_state": "Oc", "data": "Bc", "rng": 1}
    dmg = {"params": "Pd", "opt_state": "Od", "data": "Bd", "rng": 2}
    s = compose(clean, dmg, "DCD")
    assert s == {"params": "Pd", "opt_state": "Oc", "data": "Bd", "rng": 1}
    assert len(all_codes()) == 8


def test_main_effects():
    scores = {k: 10.0 - 5.0 * (k[0] == "D") for k in all_codes()}  # only params matter
    e = main_effects(scores)
    assert e["params"] == -5.0 and e["opt_state"] == 0.0 and e["data"] == 0.0


def test_metrics():
    a = np.ones((32, 10)); a[:, :3] = 0.0
    assert dormant_ratio(a) == pytest.approx(0.3)
    f = np.zeros((50, 8)); f[:, 0] = 1.0
    assert effective_rank(f) == 1

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

warnings.filterwarnings("ignore", category=FutureWarning)
ppo = pytest.importorskip("rled.agents.ppo")

BASE = dict(ENV_NAME="SpaceInvaders-MinAtar", NUM_ENVS=4, NUM_STEPS=8, NUM_MINIBATCHES=2, UPDATE_EPOCHS=1,
            TOTAL_TIMESTEPS=4 * 8 * 4, HIDDEN=16, DEFICIT="action_mask",
            DEFICIT_PARAMS={"allowed_actions": [0, 1]}, DEFICIT_START=0, DEFICIT_END=10**9)


def _leaves(p, keys):
    return [np.asarray(x) for k in keys for x in jax.tree.leaves(p["params"][k])]


def _run(extra, n=3):
    cfg, init, run = ppo.make({**BASE, **extra})
    rs0 = init(jax.random.PRNGKey(0))
    rs, m = jax.jit(lambda r: run(r, n))(rs0)
    return rs0, rs, m


def test_freeze_keeps_actor_hidden_layers_fixed_inside_window():
    rs0, rs, _ = _run(dict(FREEZE_ACTOR_REPR=True))
    p0, p1 = rs0["train_state"].params, rs["train_state"].params
    hidden = [k for k in ppo.ACTOR_KEYS if k in ("a0", "a1")]
    assert hidden, ppo.ACTOR_KEYS
    assert all(np.array_equal(a, b) for a, b in zip(_leaves(p0, hidden), _leaves(p1, hidden)))
    others = [k for k in p0["params"] if k not in hidden]
    assert not all(np.array_equal(a, b) for a, b in zip(_leaves(p0, others), _leaves(p1, others)))


def test_freeze_inactive_outside_window():
    rs0, rs, _ = _run(dict(FREEZE_ACTOR_REPR=True, DEFICIT_END=1))
    p0, p1 = rs0["train_state"].params, rs["train_state"].params
    assert not all(np.array_equal(a, b) for a, b in zip(_leaves(p0, ["a0", "a1"]), _leaves(p1, ["a0", "a1"])))


def test_anchor_is_logged_and_reduces_drift():
    rs0, rs_free, m_free = _run(dict(LR=1e-2), n=6)
    _, rs_anch, m_anch = _run(dict(LR=1e-2, ANCHOR_COEF=10.0), n=6)
    assert float(m_free["anchor_kl"].sum()) == 0.0  # off by default: not computed
    assert float(m_anch["anchor_kl"][1:].mean()) > 0.0
    cfg, _, _ = ppo.make(BASE)
    net = ppo.ActorCritic(4, BASE["HIDDEN"], cfg["LAYERNORM"])
    obs = rs0["obs"]

    def kl_from_init(p):
        l0 = jax.nn.log_softmax(net.apply(rs0["train_state"].params, obs)[0])
        l1 = jax.nn.log_softmax(net.apply(p, obs)[0])
        return float(jnp.sum(jnp.exp(l0) * (l0 - l1), -1).mean())

    assert kl_from_init(rs_anch["train_state"].params) < kl_from_init(rs_free["train_state"].params)


def test_new_metrics_present_and_finite():
    _, _, m = _run({})
    for k in ("behaviour_kl", "ratio_mean", "ratio_absdev", "clip_frac", "approx_kl", "entropy", "anchor_kl"):
        assert k in m and bool(jnp.all(jnp.isfinite(m[k]))), k
    assert float(jnp.abs(m["behaviour_kl"]).max()) < 1e-6  # on-policy: behaviour == learner

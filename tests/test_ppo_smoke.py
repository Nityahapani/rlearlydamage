import warnings

import jax
import pytest

warnings.filterwarnings("ignore", category=FutureWarning)
ppo = pytest.importorskip("rled.agents.ppo")


@pytest.mark.parametrize("deficit,params", [
    ("none", {}), ("obs_noise", {"sigma": 1.0}), ("action_mask", {"allowed_frac": 0.5}),
    ("reward_flip", {"prob": 1.0}), ("reward_delay", {"delay": 4}),
])
def test_ppo_runs_with_each_deficit(deficit, params):
    cfg, init, run = ppo.make(dict(NUM_ENVS=4, NUM_STEPS=8, NUM_MINIBATCHES=2, UPDATE_EPOCHS=1,
                                   TOTAL_TIMESTEPS=4 * 8 * 4, HIDDEN=16, DEFICIT=deficit,
                                   DEFICIT_START=0, DEFICIT_END=64, DEFICIT_PARAMS=params))
    rs = init(jax.random.PRNGKey(0))
    rs, m = jax.jit(lambda r: run(r, 3))(rs)
    assert int(rs["update"]) == 3
    expected = [0.0, 0.0, 0.0] if deficit == "none" else [1.0, 1.0, 0.0]
    assert m["deficit_active"].tolist() == expected


def test_fixed_action_mask_is_seed_independent_and_eval_is_unmasked():
    import jax
    import jax.numpy as jnp
    from rled.agents.ppo import make, make_eval

    cfg, init, run = make(dict(ENV_NAME="SpaceInvaders-MinAtar", NUM_ENVS=4, NUM_STEPS=8, NUM_MINIBATCHES=2,
                               TOTAL_TIMESTEPS=4 * 8 * 3, HIDDEN=16, DEFICIT="action_mask",
                               DEFICIT_PARAMS={"allowed_actions": [0, 3]}, DEFICIT_START=0, DEFICIT_END=10**9))
    masks = [init(jax.random.PRNGKey(s))["action_mask"].tolist() for s in range(3)]
    assert all(m == [True, False, False, True] for m in masks)
    rs = init(jax.random.PRNGKey(0))
    ev = make_eval(cfg, num_envs=4)
    out = jax.jit(ev)(rs["train_state"].params, jax.random.PRNGKey(123))
    assert out["ret"].shape == (4,) and bool(jnp.all(out["length"] > 0))
    out2 = jax.jit(ev)(rs["train_state"].params, jax.random.PRNGKey(123))
    assert jnp.array_equal(out["ret"], out2["ret"])  # common random numbers: deterministic given key

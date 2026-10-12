import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

warnings.filterwarnings("ignore", category=FutureWarning)
pqn = pytest.importorskip("rled.agents.pqn")

TINY = dict(ENV_NAME="SpaceInvaders-MinAtar", NUM_ENVS=4, NUM_STEPS=8, NUM_MINIBATCHES=2, NUM_EPOCHS=1, HIDDEN=16,
            BUFFER_SIZE=64, REPLAY_BATCH=16, DEFICIT="action_mask", DEFICIT_PARAMS={"allowed_actions": [0, 3]},
            DEFICIT_START=0, DEFICIT_END=2 * 32)


@pytest.mark.parametrize("replay", [False, True])
def test_pqn_runs_window_and_counters(replay):
    cfg, init, run = pqn.make({**TINY, "REPLAY": replay})
    rs = init(jax.random.PRNGKey(0))
    assert rs["action_mask"].tolist() == [True, False, False, True]
    rs, m = jax.jit(lambda r: run(r, 4))(rs)
    assert m["deficit_active"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert bool(jnp.all(jnp.isfinite(m["loss"])))
    assert float(m["eps"][0]) == 1.0 and float(m["eps"][1]) < 1.0
    assert cfg["REPLAY_RATIO"] == pytest.approx(1.0)  # 1 epoch here
    if replay:
        b = rs["buffer"]
        assert int(b["size"]) == cfg["BUFFER_LEN"] == 16 and int(b["ptr"]) == (4 * 8) % 16
        assert b["obs"].dtype == jnp.uint8


def test_packbits_roundtrip_is_exact_for_minatar_obs():
    cfg, init, _ = pqn.make({**TINY, "REPLAY": True})
    rs = init(jax.random.PRNGKey(1))
    o = rs["obs"]
    assert set(np.unique(np.asarray(o)).tolist()) <= {0.0, 1.0}
    back = jnp.unpackbits(jnp.packbits(o > 0.5, axis=-1), axis=-1, count=o.shape[-1]).astype(jnp.float32)
    assert jnp.array_equal(back, o)


def test_greedy_eval_unmasked_and_deterministic():
    cfg, init, _ = pqn.make({**TINY, "REPLAY": False})
    rs = init(jax.random.PRNGKey(0))
    ev = jax.jit(pqn.make_eval(cfg, num_envs=4))
    a, b = ev(rs["train_state"].params, jax.random.PRNGKey(5)), ev(rs["train_state"].params, jax.random.PRNGKey(5))
    assert jnp.array_equal(a["ret"], b["ret"]) and bool(jnp.all(a["length"] > 0))


def test_seam_mask_zero_weight():
    """Flagged slots (valid_next False) must not influence learning: two buffers that differ only in the rewards of
    flagged slots give bit-identical parameters after an update."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rled.agents.pqn import make
    cfg, init, run = make(dict(REPLAY=True, NUM_ENVS=16, NUM_STEPS=8, NUM_MINIBATCHES=4, BUFFER_SIZE=512,
                               REPLAY_BATCH=32, SEAM_MASK=True))
    rs = init(jax.random.PRNGKey(0))
    rs, _ = jax.jit(lambda r: run(r, 4))(rs)  # 32 of 32 slots filled -> wraps; ptr = 0
    b = rs["buffer"]
    rows = jnp.array([10, 11, 12])  # not written by the next update (it writes rows ptr..ptr+7)
    assert int(b["ptr"]) not in (10, 11, 12) and int(b["size"]) == cfg["BUFFER_LEN"]
    flag = b["valid_next"].at[rows].set(False)
    r1 = dict(rs, buffer=dict(b, valid_next=flag, reward=b["reward"].at[rows].set(0.0)))
    r2 = dict(rs, buffer=dict(b, valid_next=flag, reward=b["reward"].at[rows].set(1e6)))
    step = jax.jit(lambda r: run(r, 1))
    p1 = jax.tree.leaves(step(r1)[0]["train_state"].params)
    p2 = jax.tree.leaves(step(r2)[0]["train_state"].params)
    assert all(np.array_equal(x, y) for x, y in zip(p1, p2))
    # sanity: without the flag the reward change does matter
    r3 = dict(rs, buffer=dict(b, reward=b["reward"].at[rows].set(1e6)))
    p3 = jax.tree.leaves(step(r3)[0]["train_state"].params)
    assert not all(np.array_equal(x, y) for x, y in zip(p1, p3))

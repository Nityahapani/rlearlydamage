import jax
import jax.numpy as jnp
import numpy as np

from rled.agents.ppo import ActorCritic
from rled.probes import PROBE_KEYS, probe


def _params(seed, obs_dim=20, n_act=6):
    net = ActorCritic(n_act, 32, False)
    return net.init(jax.random.PRNGKey(seed), jnp.zeros((obs_dim,)))["params"], net


def test_probe_self_identity_and_manual_forward():
    pp, net = _params(0)
    bank = jax.random.bernoulli(jax.random.PRNGKey(5), 0.3, (400, 20)).astype(jnp.float32)
    out = probe(pp, pp, bank, (1, 2))
    assert set(out) == set(PROBE_KEYS)
    assert abs(float(out["kl_onset"])) < 1e-6 and abs(float(out["kl_masked_cond"])) < 1e-6
    assert float(out["masked_logit_corr"]) > 0.999
    assert float(out["decode_r2_h1"]) > 0.99          # onset logits are linear in onset a1 features
    assert abs(float(out["repr_drift_a1"])) < 1e-5 and float(out["value_corr"]) > 0.999
    assert float(out["actor_rel_change"]) == 0.0 and float(out["value_nmse"]) < 1e-10
    # manual forward matches the flax module
    from rled.probes import actor_fwd, value_fwd
    lg, v = net.apply({"params": pp}, bank)
    assert np.allclose(actor_fwd(pp, bank)[2], lg, atol=1e-5)
    assert np.allclose(value_fwd(pp, bank), v[..., 0] if v.ndim == 2 else v, atol=1e-5)


def test_probe_detects_change_and_single_masked_action():
    p0, _ = _params(0)
    p1, _ = _params(1)
    bank = jax.random.bernoulli(jax.random.PRNGKey(5), 0.3, (400, 20)).astype(jnp.float32)
    out = probe(p1, p0, bank, (3,))
    assert float(out["kl_onset"]) > 0 and float(out["actor_rel_change"]) > 0.5
    assert float(out["kl_masked_cond"]) == 0.0
    assert float(out["decode_r2_h1"]) < 0.99

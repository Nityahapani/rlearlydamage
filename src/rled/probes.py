"""State-variable probes for the point-of-no-return study (Phase 7).

All probes compare a PPO actor-critic `pp` with the onset parameters `pp_on` on a fixed bank of
observations collected by the onset policy (unmasked). Every probe is a scalar per seed.
The candidate set is fixed here BEFORE any Phase 7 data exist; the pilot is exploratory,
Stage 1 pre-registers a subset.

Candidate predictors
  policy level
    kl_onset          KL(pi_onset || pi_t), full action set (the Phase 4 drift metric)
    kl_masked_cond    KL between onset and current policy restricted to the masked actions
                      (renormalised among them); 0 when |masked| == 1
    masked_logit_corr Pearson corr over (state, masked action) of the state-dependent part of the
                      masked-action logits (logit_a - mean_a' logit_a', then minus its bank mean),
                      current vs onset: does state-conditional preference for masked actions survive?
    p_masked          mean probability mass on masked actions (unmasked current policy)
    entropy           mean entropy of the unmasked current policy
  representation level
    decode_r2_h1      held-out R^2 of a ridge readout from current a1 features to onset logits
                      (state-centred): is the competent policy still linearly decodable?
    decode_r2_h0      same from a0 features
    repr_drift_a0/a1  1 - linear CKA vs onset features
    srank_a1          effective rank (exp of entropy of normalised singular values) of a1 features
    dead_a1           fraction of a1 units inactive on the whole bank
  critic level
    value_corr        Pearson corr of V_t vs V_onset on the bank
    value_nmse        mean (V_t - V_onset)^2 / var(V_onset)
  weights
    actor_rel_change  ||theta_actor - theta_actor_onset|| / ||theta_actor_onset||  (a0, a1, Dense_0)
    critic_rel_change same for c0, c1, Dense_1
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

ACTOR = ("a0", "a1", "Dense_0")
CRITIC = ("c0", "c1", "Dense_1")


def actor_fwd(pp, x):
    x = x.astype(jnp.float32)
    h0 = jax.nn.relu(x @ pp["a0"]["kernel"] + pp["a0"]["bias"])
    h1 = jax.nn.relu(h0 @ pp["a1"]["kernel"] + pp["a1"]["bias"])
    return h0, h1, h1 @ pp["Dense_0"]["kernel"] + pp["Dense_0"]["bias"]


def value_fwd(pp, x):
    x = x.astype(jnp.float32)
    c0 = jax.nn.relu(x @ pp["c0"]["kernel"] + pp["c0"]["bias"])
    c1 = jax.nn.relu(c0 @ pp["c1"]["kernel"] + pp["c1"]["bias"])
    return (c1 @ pp["Dense_1"]["kernel"] + pp["Dense_1"]["bias"])[:, 0]


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = jnp.sum((X.T @ Y) ** 2)
    return hsic / (jnp.sqrt(jnp.sum((X.T @ X) ** 2)) * jnp.sqrt(jnp.sum((Y.T @ Y) ** 2)) + 1e-12)


def ridge_r2(X, Y, ridge=1e-3):
    """Fit on the first half of the bank, R^2 on the second half (bank order is time-ordered
    within 16 parallel envs, so halves are temporally separated). Y columns pooled."""
    n = X.shape[0] // 2
    Xtr, Xte, Ytr, Yte = X[:n], X[n:], Y[:n], Y[n:]
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    G = Xc.T @ Xc
    lam = ridge * jnp.trace(G) / G.shape[0] + 1e-8
    W = jnp.linalg.solve(G + lam * jnp.eye(G.shape[0]), Xc.T @ Yc)
    pred = (Xte - mx) @ W + my
    sse = jnp.sum((Yte - pred) ** 2)
    sst = jnp.sum((Yte - Yte.mean(0)) ** 2)
    return 1.0 - sse / (sst + 1e-12)


def srank(H):
    s = jnp.linalg.svd(H - H.mean(0), compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return jnp.exp(-jnp.sum(jnp.where(p > 0, p * jnp.log(p + 1e-30), 0.0)))


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return jnp.sum(a * b) / (jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b)) + 1e-12)


def _norm(tree):
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(tree)))


def probe(pp, pp_on, bank, masked):
    """masked: static tuple of masked action indices (e.g. (1, 2) for movement in SpaceInvaders)."""
    h0, h1, lg = actor_fwd(pp, bank)
    o0, o1, lo = actor_fwd(pp_on, bank)
    lp, lq = jax.nn.log_softmax(lg), jax.nn.log_softmax(lo)
    out = dict(kl_onset=jnp.sum(jnp.exp(lq) * (lq - lp), -1).mean(),
               entropy=-jnp.sum(jnp.exp(lp) * lp, -1).mean())
    m = jnp.array(masked)
    out["p_masked"] = jnp.exp(lp)[:, m].sum(-1).mean()
    if len(masked) > 1:
        cp, cq = jax.nn.log_softmax(lg[:, m]), jax.nn.log_softmax(lo[:, m])
        out["kl_masked_cond"] = jnp.sum(jnp.exp(cq) * (cq - cp), -1).mean()
    else:
        out["kl_masked_cond"] = jnp.array(0.0)
    zc = (lg - lg.mean(-1, keepdims=True))[:, m]
    zo = (lo - lo.mean(-1, keepdims=True))[:, m]
    out["masked_logit_corr"] = _pearson(zc - zc.mean(0), zo - zo.mean(0))
    y = lo - lo.mean(-1, keepdims=True)  # onset logits, state-centred (softmax-invariant)
    out["decode_r2_h1"] = ridge_r2(h1, y)
    out["decode_r2_h0"] = ridge_r2(h0, y)
    out["repr_drift_a0"] = 1 - linear_cka(o0, h0)
    out["repr_drift_a1"] = 1 - linear_cka(o1, h1)
    out["srank_a1"] = srank(h1)
    out["dead_a1"] = (h1.max(0) <= 0).mean()
    v, vo = value_fwd(pp, bank), value_fwd(pp_on, bank)
    out["value_corr"] = _pearson(v, vo)
    out["value_nmse"] = jnp.mean((v - vo) ** 2) / (jnp.var(vo) + 1e-12)
    sub = lambda t, ks: {k: t[k] for k in ks}
    diff = lambda ks: jax.tree.map(lambda x, y: x - y, sub(pp, ks), sub(pp_on, ks))
    out["actor_rel_change"] = _norm(diff(ACTOR)) / (_norm(sub(pp_on, ACTOR)) + 1e-12)
    out["critic_rel_change"] = _norm(diff(CRITIC)) / (_norm(sub(pp_on, CRITIC)) + 1e-12)
    return out


PROBE_KEYS = ("kl_onset", "entropy", "p_masked", "kl_masked_cond", "masked_logit_corr", "decode_r2_h1",
              "decode_r2_h0", "repr_drift_a0", "repr_drift_a1", "srank_a1", "dead_a1", "value_corr",
              "value_nmse", "actor_rel_change", "critic_rel_change")

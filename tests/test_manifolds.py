from __future__ import annotations

import jax
import jax.numpy as jnp

from diffgeo.manifold import ExtrinsicMaskedTorus, IntrinsicMaskedTorus


def test_intrinsic_projection_idempotent() -> None:
    m = IntrinsicMaskedTorus(dim=18)
    x = jnp.linspace(-20.0, 20.0, 18, dtype=jnp.float32)[None, :]
    p1 = m.projection(x)
    p2 = m.projection(p1)
    assert jnp.max(jnp.abs(p1 - p2)) < 1e-6


def test_intrinsic_log_exp_local_consistency() -> None:
    m = IntrinsicMaskedTorus(dim=18)
    rng = jax.random.PRNGKey(0)
    base = m.random_uniform(rng, (2, 18))
    v = 1e-3 * jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    y = m.exp(v, base)
    v_rec = m.log(y, base)
    assert jnp.max(jnp.abs(v - v_rec)) < 2e-3


def test_extrinsic_projection_unit_pairs() -> None:
    m = ExtrinsicMaskedTorus(dim=18)
    x = jnp.linspace(-3.0, 5.0, 36, dtype=jnp.float32)[None, :]
    p = m.projection(x)
    pairs = p.reshape(1, 18, 2)
    norms = jnp.linalg.norm(pairs, axis=-1)
    assert jnp.max(jnp.abs(norms - 1.0)) < 1e-5


def test_extrinsic_to_tangent_orthogonal_to_base() -> None:
    m = ExtrinsicMaskedTorus(dim=18)
    base = m.random_uniform(jax.random.PRNGKey(2), (2, 36))
    vec = jax.random.normal(jax.random.PRNGKey(3), (2, 36))
    t = m.to_tangent(vec, base)
    t_pairs = t.reshape(2, 18, 2)
    b_pairs = base.reshape(2, 18, 2)
    inner = jnp.sum(t_pairs * b_pairs, axis=-1)
    assert jnp.max(jnp.abs(inner)) < 1e-5


def test_extrinsic_exp_stays_on_manifold() -> None:
    m = ExtrinsicMaskedTorus(dim=18)
    base = m.random_uniform(jax.random.PRNGKey(4), (2, 36))
    v = 1e-2 * jax.random.normal(jax.random.PRNGKey(5), (2, 36))
    y = m.exp(v, base)
    norms = jnp.linalg.norm(y.reshape(2, 18, 2), axis=-1)
    assert jnp.max(jnp.abs(norms - 1.0)) < 1e-5

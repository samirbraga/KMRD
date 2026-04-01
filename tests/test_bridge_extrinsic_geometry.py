from __future__ import annotations

import jax
import jax.numpy as jnp

from diffgeo.manifold import ExtrinsicMaskedTorus
from RDM.beta_schedule import LinearBetaSchedule
from RDM.sde_lib import DiffusionMixture
from RDM.training import intrinsic_to_cossin


def _make_varlen_intrinsic_mask() -> jnp.ndarray:
    # lengths 3 and 4 residues (geo dims 12 and 18)
    m = jnp.zeros((2, 18), dtype=jnp.float32)
    m = m.at[0, :12].set(1.0)
    m = m.at[1, :18].set(1.0)
    return m


def test_extrinsic_log_exp_local_consistency_with_variable_masks() -> None:
    m_intr = _make_varlen_intrinsic_mask()
    x_intr = jax.random.uniform(jax.random.PRNGKey(0), (2, 18), minval=-jnp.pi, maxval=jnp.pi)
    x, m = intrinsic_to_cossin(x_intr, m_intr)
    m_ext = jnp.repeat(m, 2, axis=-1)
    manifold = ExtrinsicMaskedTorus(dim=18)

    base = manifold.projection(x, mask=m)
    v = 1e-3 * jax.random.normal(jax.random.PRNGKey(1), base.shape)
    y = manifold.exp(v, base, mask=m)
    v_rec = manifold.log(y, base, mask=m)

    y_pairs = y.reshape(2, 18, 2)
    m_pairs = m.astype(bool)
    assert jnp.allclose(y_pairs[..., 0][~m_pairs], 1.0, atol=1e-6)
    assert jnp.allclose(y_pairs[..., 1][~m_pairs], 0.0, atol=1e-6)
    assert jnp.max(jnp.abs((v - v_rec) * m_ext)) < 5e-3


def test_bridge_drift_shape_and_mask_invariance() -> None:
    m_intr = _make_varlen_intrinsic_mask()
    x_intr = jax.random.uniform(jax.random.PRNGKey(2), (2, 18), minval=-jnp.pi, maxval=jnp.pi)
    x, m = intrinsic_to_cossin(x_intr, m_intr)
    m_ext = jnp.repeat(m, 2, axis=-1)
    manifold = ExtrinsicMaskedTorus(dim=18)
    beta = LinearBetaSchedule(tf=1.0, t0=0.0, beta_0=10.0, beta_f=0.1)
    mix = DiffusionMixture(manifold=manifold, beta_schedule=beta, prior_type="unif")

    dest = manifold.projection(x, mask=m)
    bridge = mix.bridge(dest=dest)
    t = jnp.array([0.3, 0.7], dtype=jnp.float32)

    # Perturb only masked coordinates per sample.
    x_perturbed = x + (1.0 - m_ext) * 999.0
    d1 = bridge.drift(x, t, mask=m)
    d2 = bridge.drift(x_perturbed, t, mask=m)

    assert d1.shape == x.shape
    # masked changes should not affect active drift coordinates
    assert jnp.allclose(d1 * m_ext, d2 * m_ext, atol=1e-5)

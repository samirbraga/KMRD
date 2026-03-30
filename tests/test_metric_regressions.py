from __future__ import annotations

import jax
import jax.numpy as jnp

from diffgeo.kinetic_metric import compute_kinetic_metric_diag
from diffgeo.manifold import KineticIntrinsicTorus


def test_kinetic_metric_is_finite_positive_and_mask_neutral_on_padding() -> None:
    bsz, lmax = 2, 4
    d = 6 * (lmax - 1)
    angles = jax.random.uniform(jax.random.PRNGKey(0), (bsz, lmax, 6), minval=-jnp.pi, maxval=jnp.pi)
    lengths = jnp.array([3, 4], dtype=jnp.int32)
    geo_mask = jnp.zeros((bsz, d), dtype=jnp.float32)
    geo_mask = geo_mask.at[0, :12].set(1.0)
    geo_mask = geo_mask.at[1, :18].set(1.0)

    g = compute_kinetic_metric_diag(
        angles_batch=angles,
        lengths=lengths,
        geo_mask=geo_mask,
        cutoff=10.0,
        eps=1e-3,
        normalize=True,
        clamp_min=1e-3,
    )
    assert g.shape == (bsz, d)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.all(g > 0.0)
    # neutral on masked/padded dims
    assert jnp.allclose(g[0, 12:], 1.0, atol=1e-5)


def test_metric_anneal_lambda_endpoints_and_monotonicity() -> None:
    m = KineticIntrinsicTorus(
        dim=18,
        metric_anneal=True,
        metric_anneal_data_lambda=0.0,
        metric_anneal_prior_lambda=1.0,
        metric_anneal_power=1.0,
    )
    sigma2 = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=jnp.float32)
    lam = m.metric_anneal_lambda_from_sigma2(sigma2=sigma2, sigma2_max=1.0)
    assert jnp.isclose(lam[0], 0.0, atol=1e-6)
    assert jnp.isclose(lam[-1], 1.0, atol=1e-6)
    assert jnp.all(jnp.diff(lam) >= -1e-6)


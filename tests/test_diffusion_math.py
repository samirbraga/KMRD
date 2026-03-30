from __future__ import annotations

import jax.numpy as jnp

from score_based.diffusion_math import (
    beta_t_linear,
    metric_anneal_lambda_from_sigma2,
    sigma2_linear,
    to_angles_lengths,
    wrap_to_pi,
)


def test_wrap_to_pi_range() -> None:
    x = jnp.array([-100.0, -7.0, -3.2, 0.0, 3.2, 7.0, 100.0], dtype=jnp.float32)
    y = wrap_to_pi(x)
    assert jnp.all(y <= jnp.pi + 1e-6)
    assert jnp.all(y >= -jnp.pi - 1e-6)


def test_to_angles_lengths_shapes_and_lengths() -> None:
    bsz = 2
    d = 18  # 3 residues * 6 angles => pad = 4
    x = jnp.zeros((bsz, d), dtype=jnp.float32)
    mask = jnp.zeros((bsz, d), dtype=jnp.float32)
    mask = mask.at[0, :12].set(1.0)  # length=3
    mask = mask.at[1, :18].set(1.0)  # length=4
    angles, lengths = to_angles_lengths(x, mask, n_feats=6)
    assert angles.shape == (2, 4, 6)
    assert lengths.shape == (2,)
    assert int(lengths[0]) == 3
    assert int(lengths[1]) == 4


def test_sigma2_and_beta_monotonicity() -> None:
    t = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)
    sigma2 = sigma2_linear(t, beta_0=0.1, beta_f=10.0)
    beta = beta_t_linear(t, beta_0=0.1, beta_f=10.0)
    assert jnp.all(jnp.diff(sigma2) >= -1e-6)
    assert jnp.all(jnp.diff(beta) >= -1e-6)


def test_metric_anneal_lambda_bounds() -> None:
    sigma2 = jnp.array([0.01, 0.1, 1.0], dtype=jnp.float32)
    sigma2_max = jnp.ones_like(sigma2)
    lam = metric_anneal_lambda_from_sigma2(
        sigma2=sigma2,
        sigma2_max=sigma2_max,
        enabled=True,
        data_lambda=0.0,
        prior_lambda=1.0,
        power=1.0,
    )
    assert jnp.all(lam >= 0.0)
    assert jnp.all(lam <= 1.0)


"""Shared JAX math helpers for score-based diffusion."""

from __future__ import annotations

import jax.numpy as jnp


def wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


def to_angles_lengths(x: jnp.ndarray, mask: jnp.ndarray, n_feats: int = 6) -> tuple[jnp.ndarray, jnp.ndarray]:
    bsz, d = x.shape
    if d % n_feats != 0:
        raise ValueError(f"Intrinsic dimension {d} is not divisible by n_feats={n_feats}.")
    pad_minus_1 = d // n_feats
    pad = pad_minus_1 + 1
    angles = jnp.zeros((bsz, pad, n_feats), dtype=x.dtype).at[:, :pad_minus_1, :].set(
        x.reshape(bsz, pad_minus_1, n_feats)
    )
    valid = jnp.sum(mask, axis=-1)
    lengths = jnp.clip((valid / n_feats).astype(jnp.int32) + 1, a_min=1, a_max=pad)
    return angles, lengths


def sigma2_linear(t: jnp.ndarray, beta_0: float, beta_f: float) -> jnp.ndarray:
    return beta_0 * t + 0.5 * (beta_f - beta_0) * (t * t)


def beta_t_linear(t: jnp.ndarray, beta_0: float, beta_f: float) -> jnp.ndarray:
    return beta_0 + (beta_f - beta_0) * t


def metric_anneal_lambda_from_sigma2(
    sigma2: jnp.ndarray,
    sigma2_max: jnp.ndarray,
    enabled: bool,
    data_lambda: float,
    prior_lambda: float,
    power: float,
) -> jnp.ndarray:
    if not enabled:
        return jnp.ones_like(sigma2)
    u = jnp.clip(sigma2 / jnp.clip(sigma2_max, a_min=1e-8), a_min=0.0, a_max=1.0)
    u = jnp.power(u, power)
    lam = data_lambda + (prior_lambda - data_lambda) * u
    return jnp.clip(lam, a_min=0.0, a_max=1.0)


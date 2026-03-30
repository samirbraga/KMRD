"""Score-matching losses for flat and kinetic intrinsic torus setups."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from RDM.sde_lib import DiffusionMixture
from diffgeo.manifold import to_angles_lengths


def _wrapped_eps_target(
    dtheta_raw: jnp.ndarray,
    sigma2: jnp.ndarray,
    n_wrap: int,
    g_diag_0: jnp.ndarray | None = None,
) -> jnp.ndarray:
    sigma2_ = sigma2[:, None]
    sqrt_2sig = jnp.sqrt(2.0 * sigma2_)
    sqrt_g = (
        jnp.ones_like(dtheta_raw)
        if g_diag_0 is None
        else jnp.sqrt(jnp.clip(g_diag_0, a_min=1e-8))
    )
    k_vals = jnp.arange(-n_wrap, n_wrap + 1, dtype=dtheta_raw.dtype)
    delta = dtheta_raw[None, :, :] - 2.0 * jnp.pi * k_vals[:, None, None]
    eps_candidates = sqrt_g[None, :, :] * delta / sqrt_2sig[None, :, :]
    log_w = -0.5 * eps_candidates * eps_candidates
    w = jax.nn.softmax(log_w, axis=0)
    return jnp.sum(w * eps_candidates, axis=0)


def _call_model(
    model_apply: Callable,
    params,
    x: jnp.ndarray,
    t: jnp.ndarray,
    mask: jnp.ndarray,
    manifold,
    *,
    g_diag: jnp.ndarray | None,
    rng: jax.Array | None = None,
    deterministic: bool = True,
) -> jnp.ndarray:
    if rng is None:
        return model_apply(
            {"params": params},
            inputs=x,
            timestep=t,
            mask=mask,
            manifold=manifold,
            g_diag=g_diag,
            deterministic=deterministic,
        )
    return model_apply(
        {"params": params},
        inputs=x,
        timestep=t,
        mask=mask,
        manifold=manifold,
        g_diag=g_diag,
        deterministic=deterministic,
        rngs={"dropout": rng},
    )


def get_flat_score_loss_fn(
    *,
    mix: DiffusionMixture,
    model_apply: Callable,
    eps: float = 1e-5,
    weight_type: str = "importance",
    n_wrap: int = 2,
):
    """DSM loss for intrinsic flat torus where model predicts epsilon."""

    def weight_fn(t: jnp.ndarray) -> jnp.ndarray:
        if weight_type == "default":
            return 1.0 / jnp.clip(mix.beta_schedule.beta_t(t), a_min=1e-8)
        if weight_type.startswith("const_"):
            return jnp.full_like(t, float(weight_type.split("_")[-1]))
        if weight_type == "importance":
            z = mix.importance_cum_weight(mix.tf - eps, eps)
            return jnp.ones_like(t) * z
        raise NotImplementedError(f"{weight_type} not implemented")

    def loss_fn(
        rng: jax.Array,
        params,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        deterministic: bool = True,
    ):
        bsz = x.shape[0]
        rng_t, rng_eps, rng_model = jax.random.split(rng, 3)
        t = jax.random.uniform(rng_t, (bsz,), minval=eps, maxval=mix.tf - eps)
        sigma2 = jnp.clip(
            mix.beta_schedule.rescale_t_delta(
                jnp.full_like(t, mix.t0),
                t,
            ),
            a_min=1e-8,
        )

        noise = jax.random.normal(rng_eps, x.shape, dtype=x.dtype) * mask
        theta_t = mix.manifold.projection(x + jnp.sqrt(2.0 * sigma2)[:, None] * noise, mask=mask)
        dtheta_raw = theta_t - x
        eps_target = _wrapped_eps_target(dtheta_raw=dtheta_raw, sigma2=sigma2, n_wrap=n_wrap) * mask

        eps_pred = _call_model(
            model_apply=model_apply,
            params=params,
            x=theta_t,
            t=t,
            mask=mask,
            manifold=mix.manifold,
            g_diag=None,
            rng=rng_model,
            deterministic=deterministic,
        )
        residual = eps_pred - eps_target

        per_dim = (residual * residual) * mask
        denom = jnp.clip(jnp.sum(mask, axis=-1), a_min=1.0)
        per_sample = jnp.sum(per_dim, axis=-1) / denom
        loss = jnp.mean(weight_fn(t) * per_sample)
        return loss, {"sigma2_mean": jnp.mean(sigma2)}

    return loss_fn


def get_kinetic_score_loss_fn(
    *,
    mix: DiffusionMixture,
    model_apply: Callable,
    eps: float = 1e-5,
    weight_type: str = "importance",
    n_wrap: int = 2,
):
    """DSM loss for kinetic-diagonal torus metric where model predicts epsilon."""

    manifold = mix.manifold
    if not hasattr(manifold, "kinetic_metric_diag"):
        raise ValueError("kinetic_score loss requires manifold.kinetic_metric_diag")

    def weight_fn(t: jnp.ndarray) -> jnp.ndarray:
        if weight_type == "default":
            return 1.0 / jnp.clip(mix.beta_schedule.beta_t(t), a_min=1e-8)
        if weight_type.startswith("const_"):
            return jnp.full_like(t, float(weight_type.split("_")[-1]))
        if weight_type == "importance":
            z = mix.importance_cum_weight(mix.tf - eps, eps)
            return jnp.ones_like(t) * z
        raise NotImplementedError(f"{weight_type} not implemented")

    def loss_fn(
        rng: jax.Array,
        params,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        deterministic: bool = True,
    ):
        bsz = x.shape[0]
        rng_t, rng_eps, rng_model = jax.random.split(rng, 3)
        t = jax.random.uniform(rng_t, (bsz,), minval=eps, maxval=mix.tf - eps)
        sigma2 = jnp.clip(
            mix.beta_schedule.rescale_t_delta(
                jnp.full_like(t, mix.t0),
                t,
            ),
            a_min=1e-8,
        )
        sigma2_max = jnp.clip(
            mix.beta_schedule.rescale_t_delta(
                jnp.full_like(t, mix.t0),
                jnp.full_like(t, mix.tf),
            ),
            a_min=1e-8,
        )
        if hasattr(manifold, "metric_anneal_lambda_from_sigma2"):
            anneal_lambda = manifold.metric_anneal_lambda_from_sigma2(sigma2, sigma2_max)
        else:
            anneal_lambda = jnp.ones_like(sigma2)

        angles_0, lengths_0 = to_angles_lengths(x, mask)
        g_diag_0 = manifold.kinetic_metric_diag(
            angles=angles_0,
            lengths=lengths_0,
            geo_mask=mask,
            anneal_lambda=anneal_lambda,
        )
        sigma_diag_0 = jnp.clip(1.0 / g_diag_0, a_min=1e-8)

        noise = jax.random.normal(rng_eps, x.shape, dtype=x.dtype) * mask
        theta_t = manifold.projection(
            x + jnp.sqrt(2.0 * sigma2)[:, None] * jnp.sqrt(sigma_diag_0) * noise,
            mask=mask,
        )

        angles_t, lengths_t = to_angles_lengths(theta_t, mask)
        g_diag_t = manifold.kinetic_metric_diag(
            angles=angles_t,
            lengths=lengths_t,
            geo_mask=mask,
            anneal_lambda=anneal_lambda,
        )
        sigma_diag_t = jnp.clip(1.0 / g_diag_t, a_min=1e-8)

        dtheta_raw = theta_t - x
        eps_target = _wrapped_eps_target(
            dtheta_raw=dtheta_raw,
            sigma2=sigma2,
            n_wrap=n_wrap,
            g_diag_0=g_diag_0,
        ) * mask

        eps_pred = _call_model(
            model_apply=model_apply,
            params=params,
            x=theta_t,
            t=t,
            mask=mask,
            manifold=manifold,
            g_diag=g_diag_t,
            rng=rng_model,
            deterministic=deterministic,
        )
        residual = eps_pred - eps_target

        per_dim = (residual * residual) * sigma_diag_t * mask
        denom = jnp.clip(jnp.sum(mask, axis=-1), a_min=1.0)
        per_sample = jnp.sum(per_dim, axis=-1) / denom
        loss = jnp.mean(weight_fn(t) * per_sample)
        return loss, {
            "sigma2_mean": jnp.mean(sigma2),
            "g0_mean": jnp.mean(g_diag_0),
            "gt_mean": jnp.mean(g_diag_t),
        }

    return loss_fn


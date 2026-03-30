"""RDM losses in JAX.

Bridge loss lives here. Score losses are now hosted in ``score_based.losses``
and re-exported here for backward compatibility.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from RDM.sde_lib import DiffusionMixture
from RDM.solver import get_twoway_sampler


def _reduce_batch(losses: jnp.ndarray, reduce_mean: bool) -> jnp.ndarray:
    return jnp.mean(losses, axis=-1) if reduce_mean else jnp.sum(losses, axis=-1)


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


def get_bridge_loss_fn(
    *,
    mix: DiffusionMixture,
    model_apply_f: Callable,
    model_apply_b: Callable,
    reduce_mean: bool = False,
    eps: float = 1e-5,
    num_steps: int = 10,
    weight_type: str = "default",
    normalize_by_dim: bool = True,
):
    """Two-way bridge matching loss (flat-style bridge objective)."""

    sampler = get_twoway_sampler(mix, num_steps=num_steps, eps=eps)

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
        params_f,
        params_b,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        deterministic: bool = True,
    ):
        bsz = x.shape[0]
        rng_t, rng_prior, rng_path, rng_f, rng_b = jax.random.split(rng, 5)
        if weight_type == "importance":
            t = mix.sample_importance_weighted_time(rng_t, (bsz,), eps)
        else:
            t = jax.random.uniform(rng_t, (bsz,), minval=eps, maxval=mix.tf - eps)

        x0 = mix.prior.sample(rng_prior, x.shape, mask=mask)
        xt = sampler(rng_path, x0=x0, xf=x, t=t, mask=mask)
        weight = weight_fn(t)

        pred_f = _call_model(
            model_apply=model_apply_f,
            params=params_f,
            x=xt,
            t=t,
            mask=mask,
            manifold=mix.manifold,
            g_diag=None,
            rng=rng_f,
            deterministic=deterministic,
        )
        target_f = mix.bridge(dest=x).drift(xt, t, mask=mask)
        res_f = pred_f - target_f
        loss_f = 0.5 * mix.manifold.metric.squared_norm(res_f, base_point=xt, mask=mask)
        loss_f = _reduce_batch((weight * loss_f).reshape(bsz, -1), reduce_mean=reduce_mean)

        pred_b = _call_model(
            model_apply=model_apply_b,
            params=params_b,
            x=xt,
            t=mix.tf - t,
            mask=mask,
            manifold=mix.manifold,
            g_diag=None,
            rng=rng_b,
            deterministic=deterministic,
        )
        target_b = mix.rev().bridge(dest=x0).drift(xt, mix.tf - t, mask=mask)
        res_b = pred_b - target_b
        loss_b = 0.5 * mix.manifold.metric.squared_norm(res_b, base_point=xt, mask=mask)
        loss_b = _reduce_batch((weight * loss_b).reshape(bsz, -1), reduce_mean=reduce_mean)

        lf = jnp.mean(loss_f)
        lb = jnp.mean(loss_b)
        if normalize_by_dim:
            dim = jnp.asarray(x.shape[-1], dtype=lf.dtype)
            lf = lf / jnp.clip(dim, a_min=1.0)
            lb = lb / jnp.clip(dim, a_min=1.0)
        return lf + lb, {"loss_f": lf, "loss_b": lb}

    return loss_fn

"""Shared intrinsic reverse sampling utilities."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax

from diffgeo.kinetic_metric import compute_kinetic_metric_diag
from utils.diffusion_math import (
    beta_t_linear,
    metric_anneal_lambda_from_sigma2,
    sigma2_linear,
    to_angles_lengths,
    wrap_to_pi,
)


def sample_intrinsic_batch(
    *,
    params,
    model,
    mask: jnp.ndarray,
    rng: jax.Array,
    n_steps: int,
    eps: float,
    beta_0: float,
    beta_f: float,
    metric_type: Literal["kinetic_diag", "flat_torus"],
    metric_cutoff: float = 10.0,
    metric_eps: float = 1e-3,
    metric_normalize: bool = True,
    metric_clamp_min: float = 1e-3,
    metric_clamp_max: float | None = None,
    metric_anneal: bool = True,
    metric_anneal_data_lambda: float = 0.0,
    metric_anneal_prior_lambda: float = 1.0,
    metric_anneal_power: float = 1.0,
    pc_corrector_steps: int = 0,
    pc_corrector_step_scale: float = 1.0,
    pc_corrector_noise_scale: float = 1.0,
) -> jnp.ndarray:
    bsz, _ = mask.shape
    rng, rng_init = jax.random.split(rng)
    x0 = (jax.random.uniform(rng_init, mask.shape, minval=-jnp.pi, maxval=jnp.pi) * mask).astype(jnp.float32)

    ts = jnp.linspace(1.0 - eps, 0.0 + eps, n_steps, dtype=jnp.float32)
    sigma2_max = jnp.clip(sigma2_linear(jnp.ones((bsz,), dtype=jnp.float32), beta_0, beta_f), a_min=1e-8)

    def _score_and_preconditioner(
        x_cur: jnp.ndarray,
        vec_t: jnp.ndarray,
        sigma2_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        g_diag = None
        sigma_diag = jnp.ones_like(mask)
        if metric_type == "kinetic_diag":
            angles, lengths = to_angles_lengths(x_cur, mask)
            g_diag = compute_kinetic_metric_diag(
                angles_batch=angles,
                lengths=lengths,
                geo_mask=mask,
                cutoff=metric_cutoff,
                eps=metric_eps,
                normalize=metric_normalize,
                clamp_min=metric_clamp_min,
                clamp_max=metric_clamp_max,
            )
            lam = metric_anneal_lambda_from_sigma2(
                sigma2=sigma2_t,
                sigma2_max=sigma2_max,
                enabled=metric_anneal,
                data_lambda=metric_anneal_data_lambda,
                prior_lambda=metric_anneal_prior_lambda,
                power=metric_anneal_power,
            )
            g_diag = (1.0 - lam[:, None]) + lam[:, None] * g_diag
            g_diag = jnp.nan_to_num(g_diag, nan=1.0, posinf=1e6, neginf=1.0)
            sigma_diag = jnp.clip(1.0 / g_diag, a_min=1e-8)

        eps_pred = model.apply(
            {"params": params},
            inputs=x_cur,
            timestep=vec_t,
            mask=mask,
            manifold=None,
            g_diag=(g_diag if metric_type == "kinetic_diag" else None),
            deterministic=True,
        )
        if metric_type == "kinetic_diag":
            score = -jnp.sqrt(jnp.clip(g_diag, a_min=1e-8)) * eps_pred / jnp.sqrt(2.0 * sigma2_t[:, None])
        else:
            score = -eps_pred / jnp.sqrt(2.0 * sigma2_t[:, None])
        return score, sigma_diag

    def _body(k: int, carry: tuple[jnp.ndarray, jax.Array]) -> tuple[jnp.ndarray, jax.Array]:
        x, rng_loop = carry
        t = ts[k]
        t_next = ts[k + 1]
        dt = jnp.abs(t - t_next)
        vec_t = jnp.full((bsz,), t, dtype=jnp.float32)
        beta_t = beta_t_linear(vec_t, beta_0, beta_f)
        sigma2_t = jnp.clip(sigma2_linear(vec_t, beta_0, beta_f), a_min=1e-8)
        score, sigma_diag = _score_and_preconditioner(x, vec_t, sigma2_t)

        rng_loop, rng_noise = jax.random.split(rng_loop)
        noise = jax.random.normal(rng_noise, x.shape, dtype=x.dtype) * mask
        drift = 2.0 * dt * beta_t[:, None] * sigma_diag * score
        diff = jnp.sqrt(2.0 * dt * beta_t)[:, None] * jnp.sqrt(sigma_diag) * noise
        x_next = wrap_to_pi(x + drift + diff) * mask

        if pc_corrector_steps > 0:
            corr_dt = (pc_corrector_step_scale * dt) / float(pc_corrector_steps)
            corr_dt = jnp.clip(corr_dt, a_min=1e-8)

            def _corr_body(_: int, corr_carry: tuple[jnp.ndarray, jax.Array]) -> tuple[jnp.ndarray, jax.Array]:
                x_corr, rng_corr = corr_carry
                score_corr, sigma_diag_corr = _score_and_preconditioner(x_corr, vec_t, sigma2_t)
                rng_corr, rng_corr_noise = jax.random.split(rng_corr)
                corr_noise = jax.random.normal(rng_corr_noise, x_corr.shape, dtype=x_corr.dtype) * mask
                corr_drift = corr_dt * beta_t[:, None] * sigma_diag_corr * score_corr
                corr_diff = jnp.sqrt(2.0 * corr_dt * pc_corrector_noise_scale) * jnp.sqrt(sigma_diag_corr) * corr_noise
                x_corr_next = wrap_to_pi(x_corr + corr_drift + corr_diff) * mask
                return x_corr_next, rng_corr

            x_next, rng_loop = lax.fori_loop(0, pc_corrector_steps, _corr_body, (x_next, rng_loop))
        return (x_next, rng_loop)

    x_fin, _ = lax.fori_loop(0, n_steps - 1, _body, (x0, rng))
    return x_fin

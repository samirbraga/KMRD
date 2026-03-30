"""RDM samplers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from RDM.sde_lib import DiffusionMixture


def _twoway_update(
    *,
    mix: DiffusionMixture,
    x0: jnp.ndarray,
    xf: jnp.ndarray,
    choose_forward: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    dt: jnp.ndarray,
    rng: jax.Array,
    mask: jnp.ndarray | None,
) -> jnp.ndarray:
    manifold = mix.manifold
    z = manifold.random_normal_tangent(rng=rng, base_point=x, mask=mask)

    fsde = mix.bridge(xf)
    bsde = mix.rev().bridge(x0)
    fdrift, fdiff = fsde.coefficients(x, t, mask=mask)
    bdrift, bdiff = bsde.coefficients(x, t, mask=mask)

    f = jnp.asarray(choose_forward, dtype=x.dtype)[:, None]
    drift = f * fdrift + (1.0 - f) * bdrift
    diffusion = f[:, 0] * fdiff + (1.0 - f[:, 0]) * bdiff

    tangent = drift * dt[:, None] + z * jnp.sqrt(jnp.abs(dt))[:, None] * diffusion[:, None]
    return manifold.exp(tangent_vec=tangent, base_point=x, mask=mask)


def get_twoway_sampler(
    mix: DiffusionMixture,
    num_steps: int = 10,
    eps: float = 1e-3,
):
    """Two-way bridge sampler used by bridge matching."""

    n = int(num_steps)
    if n <= 0:
        raise ValueError("num_steps must be >= 1")

    def sampler(
        rng: jax.Array,
        x0: jnp.ndarray,
        xf: jnp.ndarray,
        t: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        choose_forward = t < 0.5
        cf = jnp.asarray(choose_forward, dtype=x0.dtype)[:, None]
        x = cf * x0 + (1.0 - cf) * xf

        ts = jnp.where(choose_forward, t, 1.0 - t)
        t0 = jnp.full_like(ts, mix.t0 + eps)
        ts = jnp.clip(ts, a_min=mix.t0 + eps, a_max=mix.tf - eps)
        dt = (ts - t0) / float(n)
        us = (jnp.arange(n, dtype=x0.dtype) + 0.5) / float(n)

        def body(carry, u):
            rng_i, x_i = carry
            rng_i, step_rng = jax.random.split(rng_i)
            ti = t0 + u * (ts - t0)
            x_next = _twoway_update(
                mix=mix,
                x0=x0,
                xf=xf,
                choose_forward=choose_forward,
                x=x_i,
                t=ti,
                dt=dt,
                rng=step_rng,
                mask=mask,
            )
            return (rng_i, x_next), None

        (_, x_final), _ = jax.lax.scan(body, (rng, x), xs=us)
        return x_final

    return sampler

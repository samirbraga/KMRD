"""RDM samplers."""

from __future__ import annotations

import abc

import jax
import jax.numpy as jnp

from RDM.sde_lib import DiffusionMixture


class Predictor(abc.ABC):
    """Abstract predictor API for one SDE update."""

    @abc.abstractmethod
    def update_fn(
        self,
        *,
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt: jnp.ndarray,
        rng: jax.Array,
        mask: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """One predictor update. Returns (x_next, x_mean)."""
        raise NotImplementedError


class EulerMaruyamaTwoWayPredictor(Predictor):
    """Two-way bridge predictor matching the legacy Torch layout."""

    def __init__(
        self,
        *,
        mix: DiffusionMixture,
        x0: jnp.ndarray,
        xf: jnp.ndarray,
        choose_forward: jnp.ndarray,
    ):
        self.mix = mix
        self.fsde = mix.bridge(xf)
        self.bsde = mix.rev().bridge(x0)
        self.choose_forward = choose_forward

    def update_fn(
        self,
        *,
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt: jnp.ndarray,
        rng: jax.Array,
        mask: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        manifold = self.mix.manifold
        z = manifold.random_normal_tangent(rng=rng, base_point=x, mask=mask)

        fdrift, fdiff = self.fsde.coefficients(x, t, mask=mask)
        bdrift, bdiff = self.bsde.coefficients(x, t, mask=mask)

        f = jnp.asarray(self.choose_forward, dtype=x.dtype)[:, None]
        drift = f * fdrift + (1.0 - f) * bdrift
        diffusion = f[:, 0] * fdiff + (1.0 - f[:, 0]) * bdiff

        tangent = drift * dt[:, None] + z * jnp.sqrt(jnp.abs(dt))[:, None] * diffusion[:, None]
        x_next = manifold.exp(tangent_vec=tangent, base_point=x, mask=mask)
        return x_next, x_next


class EulerMaruyamaBridgePredictor(Predictor):
    """Bridge sampling predictor used by evaluation (PC/PF-ODE mode)."""

    def __init__(
        self,
        *,
        params_f,
        params_b,
        model_apply_f,
        model_apply_b,
        mix: DiffusionMixture,
        use_pode: bool,
    ):
        self.params_f = params_f
        self.params_b = params_b
        self.model_apply_f = model_apply_f
        self.model_apply_b = model_apply_b
        self.mix = mix
        self.use_pode = use_pode

    def update_fn(
        self,
        *,
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt: jnp.ndarray,
        rng: jax.Array,
        mask: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        manifold = self.mix.manifold
        drift_f = self.model_apply_f(
            {"params": self.params_f},
            inputs=x,
            timestep=t,
            mask=mask,
            manifold=manifold,
            g_diag=None,
            deterministic=True,
        )
        if self.use_pode:
            drift_b = self.model_apply_b(
                {"params": self.params_b},
                inputs=x,
                timestep=jnp.full((x.shape[0],), self.mix.tf, dtype=x.dtype) - t,
                mask=mask,
                manifold=manifold,
                g_diag=None,
                deterministic=True,
            )
            scaled_score = drift_f + drift_b
            drift = drift_f - 0.5 * scaled_score
            diffusion = jnp.zeros((x.shape[0],), dtype=x.dtype)
        else:
            drift = drift_f
            diffusion = jnp.sqrt(jnp.clip(self.mix.beta_schedule.beta_t(t), a_min=1e-8))

        z = manifold.random_normal_tangent(rng=rng, base_point=x, mask=mask)
        tangent = drift * dt + z * jnp.sqrt(jnp.abs(dt)) * diffusion[:, None]
        x_next = manifold.exp(tangent_vec=tangent, base_point=x, mask=mask)
        return x_next, x_next


def get_twoway_sampler(
    mix: DiffusionMixture,
    num_steps: int = 10,
    eps: float = 1e-3,
):
    """Two-way bridge sampler used by bridge matching."""

    del eps
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
        predictor = EulerMaruyamaTwoWayPredictor(
            mix=mix,
            x0=x0,
            xf=xf,
            choose_forward=choose_forward,
        )

        ts = jnp.where(choose_forward, t, 1.0 - t)
        t0 = jnp.full_like(ts, mix.t0)
        dt = (ts - t0) / float(n)
        us = jnp.linspace(0.0, 1.0, n, dtype=x0.dtype)

        def body(i, carry):
            rng_i, x_i = carry
            rng_i, step_rng = jax.random.split(rng_i)
            u = us[i]
            ti = t0 + u * (ts - t0)
            x_next, _ = predictor.update_fn(
                x=x_i,
                t=ti,
                dt=dt,
                rng=step_rng,
                mask=mask,
            )
            return (rng_i, x_next)

        _, x_final = jax.lax.fori_loop(0, n, body, (rng, x))
        return x_final

    return sampler


def sample_bridge_pc_batch(
    *,
    params_f,
    params_b,
    model_apply_f,
    model_apply_b,
    mix: DiffusionMixture,
    mask: jnp.ndarray,
    rng: jax.Array,
    n_steps: int = 1000,
    eps: float = 1e-3,
    use_pode: bool = True,
) -> jnp.ndarray:
    """Sample from learned bridge drifts with a simple PC (Euler-Maruyama) loop.

    This mirrors the old Torch evaluation bridge path:
    - predictor on approximate SDE driven by forward/backward bridge drifts,
    - optional probability-flow ODE mode (``use_pode=True``) by zeroing diffusion.
    """

    n = int(n_steps)
    if n <= 0:
        raise ValueError("n_steps must be >= 1")
    bsz = int(mask.shape[0])
    data_dim = int(getattr(mix.manifold, "extrinsic_dim", getattr(mix.manifold, "dim")))
    predictor = EulerMaruyamaBridgePredictor(
        params_f=params_f,
        params_b=params_b,
        model_apply_f=model_apply_f,
        model_apply_b=model_apply_b,
        mix=mix,
        use_pode=use_pode,
    )

    rng, prior_rng = jax.random.split(rng)
    x0 = mix.prior.sample(prior_rng, (bsz, data_dim), mask=mask)

    ts = jnp.linspace(
        jnp.asarray(mix.t0, dtype=x0.dtype),
        jnp.asarray(mix.tf - eps, dtype=x0.dtype),
        n,
        dtype=x0.dtype,
    )
    dt = (ts[-1] - ts[0]) / jnp.asarray(n, dtype=x0.dtype)

    def body(i, carry):
        rng_i, x_i = carry
        rng_i, step_rng = jax.random.split(rng_i)
        t_scalar = ts[i]
        t = jnp.full((x_i.shape[0],), t_scalar, dtype=x_i.dtype)
        x_next, _ = predictor.update_fn(
            x=x_i,
            t=t,
            dt=dt,
            rng=step_rng,
            mask=mask,
        )
        return (rng_i, x_next)

    _, x_final = jax.lax.fori_loop(0, n, body, (rng, x0))
    return x_final

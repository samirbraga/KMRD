"""RDM mixture and bridge definitions."""

from __future__ import annotations

import abc

import jax
import jax.numpy as jnp

from RDM.distribution import UniformDistribution


class Mixture(abc.ABC):
    def __init__(self, manifold, beta_schedule, prior_type: str = "unif", **kwargs):
        self.manifold = manifold
        self.beta_schedule = beta_schedule
        self.t0 = beta_schedule.t0
        self.tf = beta_schedule.tf
        self.prior_type = prior_type
        self.kwargs = kwargs

    def time_scale(self, t: jnp.ndarray) -> jnp.ndarray:
        scale = self.beta_schedule.rescale_t_delta(t, self.tf)
        return self.beta_schedule.beta_t(t) / jnp.clip(scale, a_min=1e-8)

    def diffusion(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        del x
        return jnp.sqrt(jnp.clip(self.beta_schedule.beta_t(t), a_min=1e-8))

    @property
    def prior(self):
        if self.prior_type == "unif":
            return UniformDistribution(self.manifold)
        if self.prior_type in {"data", "none"}:
            return None
        raise NotImplementedError(f"prior_type={self.prior_type} not implemented")

    def importance_cum_weight(self, t: jnp.ndarray, eps: float) -> jnp.ndarray:
        if self.beta_schedule._beta == 0:
            return t / self.beta_schedule.beta_0
        num = self.beta_schedule.beta_t(t)
        den = self.beta_schedule.beta_t(self.t0 + eps)
        z = jnp.log(jnp.clip(num / jnp.clip(den, a_min=1e-8), a_min=1e-8))
        return z / self.beta_schedule._beta

    def sample_importance_weighted_time(
        self,
        rng: jax.Array,
        shape: tuple[int, ...],
        eps: float,
        steps: int = 100,
    ) -> jnp.ndarray:
        z = self.importance_cum_weight(self.tf - eps, eps=eps)
        quantile = jax.random.uniform(rng, shape=shape, minval=0.0, maxval=z)
        lb = jnp.full(shape, self.t0 + eps, dtype=jnp.float32)
        ub = jnp.full(shape, self.tf - eps, dtype=jnp.float32)

        def bisect(carry, _):
            lb_i, ub_i = carry
            mid = 0.5 * (lb_i + ub_i)
            val = self.importance_cum_weight(mid, eps=eps)
            choose = val <= quantile
            new_lb = jnp.where(choose, mid, lb_i)
            new_ub = jnp.where(choose, ub_i, mid)
            return (new_lb, new_ub), None

        (lb, ub), _ = jax.lax.scan(bisect, (lb, ub), xs=None, length=steps)
        return 0.5 * (lb + ub)


class DiffusionMixture(Mixture):
    def __init__(
        self,
        manifold,
        beta_schedule,
        prior_type: str = "unif",
        drift_scale: float = 1.0,
        mix_type: str = "log",
        **kwargs,
    ):
        super().__init__(
            manifold=manifold, beta_schedule=beta_schedule, prior_type=prior_type, **kwargs
        )
        self.drift_scale = float(drift_scale)
        self.mix_type = str(mix_type)

    def bridge(self, dest: jnp.ndarray):
        if self.mix_type != "log":
            raise NotImplementedError(f"mix_type={self.mix_type} not implemented")
        return BrownianBridge(
            manifold=self.manifold,
            beta_schedule=self.beta_schedule,
            dest=dest,
            drift_scale=self.drift_scale,
        )

    def rev(self):
        return DiffusionMixture(
            manifold=self.manifold,
            beta_schedule=self.beta_schedule.reverse(),
            prior_type="data",
            drift_scale=self.drift_scale,
            mix_type=self.mix_type,
            **self.kwargs,
        )


class Bridge(abc.ABC):
    def __init__(self, manifold, beta_schedule, dest: jnp.ndarray, drift_scale: float):
        self.manifold = manifold
        self.beta_schedule = beta_schedule
        self.t0 = beta_schedule.t0
        self.tf = beta_schedule.tf
        self.dest = dest
        self.drift_scale = float(drift_scale)

    def time_scale(self, t: jnp.ndarray) -> jnp.ndarray:
        scale = self.beta_schedule.rescale_t_delta(t, self.tf)
        return self.beta_schedule.beta_t(t) / jnp.clip(scale, a_min=1e-8)

    def drift(self, x: jnp.ndarray, t: jnp.ndarray, mask: jnp.ndarray | None = None):
        drift_raw = self.drift_before_scale(x, t, mask=mask)
        coeff = self.time_scale(t) * self.drift_scale
        return drift_raw * coeff[:, None]

    def diffusion(self, x: jnp.ndarray, t: jnp.ndarray):
        del x
        return jnp.sqrt(jnp.clip(self.beta_schedule.beta_t(t), a_min=1e-8))

    def coefficients(self, x: jnp.ndarray, t: jnp.ndarray, mask: jnp.ndarray | None = None):
        return self.drift(x, t, mask=mask), self.diffusion(x, t)

    @abc.abstractmethod
    def drift_before_scale(self, x: jnp.ndarray, t: jnp.ndarray, mask: jnp.ndarray | None = None):
        raise NotImplementedError


class BrownianBridge(Bridge):
    def drift_before_scale(self, x: jnp.ndarray, t: jnp.ndarray, mask: jnp.ndarray | None = None):
        del t
        return self.manifold.log(point=self.dest, base_point=x, mask=mask)

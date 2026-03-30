"""Beta schedules for RDM."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class LinearBetaSchedule:
    tf: float = 1.0
    t0: float = 0.0
    beta_0: float = 0.2
    beta_f: float = 0.001

    @property
    def _beta(self) -> float:
        return float(self.beta_f - self.beta_0)

    @property
    def _t(self) -> float:
        return float(self.tf - self.t0)

    @property
    def normed(self) -> bool:
        return bool(self.t0 == 0.0 and self.tf == 1.0)

    def normed_t(self, t):
        return (t - self.t0) / self._t

    def rescale_t_delta(self, s, t):
        dt = t - s
        if self.normed:
            return dt * (self.beta_0 + 0.5 * (t + s) * self._beta)
        return dt * self.beta_0 + (0.5 * (t + s) - self.t0) * self._beta * dt / self._t

    def beta_t(self, t):
        return self.beta_0 + self.normed_t(t) * self._beta

    def reverse(self) -> "LinearBetaSchedule":
        return LinearBetaSchedule(
            tf=self.tf,
            t0=self.t0,
            beta_0=self.beta_f,
            beta_f=self.beta_0,
        )


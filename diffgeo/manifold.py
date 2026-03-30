"""Masked torus manifolds used by diffusion losses/samplers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from diffgeo.kinetic_metric import compute_kinetic_metric_diag

try:
    import geomstats.backend as gs
except Exception:  # pragma: no cover - optional at runtime
    gs = None


def _to_mask(mask: Optional[jnp.ndarray], like: jnp.ndarray) -> jnp.ndarray:
    if mask is None:
        return jnp.ones_like(like)
    return jnp.asarray(mask, dtype=like.dtype)


def to_angles_lengths(x: jnp.ndarray, mask: jnp.ndarray, n_feats: int = 6):
    bsz, d = x.shape
    if d % n_feats != 0:
        raise ValueError(f"Input dim {d} is not divisible by n_feats={n_feats}.")
    pad_minus_1 = d // n_feats
    pad = pad_minus_1 + 1
    angles = jnp.zeros((bsz, pad, n_feats), dtype=x.dtype)
    angles = angles.at[:, :pad_minus_1, :].set(x.reshape(bsz, pad_minus_1, n_feats))
    lengths = jnp.clip((jnp.sum(mask, axis=-1) / n_feats).astype(jnp.int32) + 1, 1, pad)
    return angles, lengths


@dataclass(frozen=True)
class IntrinsicTorusMetric:
    def squared_norm(
        self,
        vector: jnp.ndarray,
        base_point: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        del base_point
        sq = vector * vector
        if mask is not None:
            sq = sq * jnp.asarray(mask, dtype=sq.dtype)
        return jnp.sum(sq, axis=-1)


class IntrinsicMaskedTorus:
    """Flat torus in intrinsic angle coordinates with optional masking."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.metric = IntrinsicTorusMetric()

    @staticmethod
    def wrap(x: jnp.ndarray) -> jnp.ndarray:
        if gs is not None:
            return gs.arctan2(gs.sin(x), gs.cos(x))
        return jnp.arctan2(jnp.sin(x), jnp.cos(x))

    def projection(self, point: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        out = self.wrap(point)
        return out * _to_mask(mask, out)

    def to_tangent(
        self,
        vector: jnp.ndarray,
        base_point: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        del base_point
        return vector * _to_mask(mask, vector)

    def log(
        self,
        point: jnp.ndarray,
        base_point: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        out = self.wrap(point - base_point)
        return out * _to_mask(mask, out)

    def exp(
        self,
        tangent_vec: jnp.ndarray,
        base_point: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        t = tangent_vec * _to_mask(mask, tangent_vec)
        out = self.wrap(base_point + t)
        return out * _to_mask(mask, out)

    def random_uniform(
        self,
        rng: jax.Array,
        shape: tuple[int, int],
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        pi = float(gs.pi) if gs is not None else float(jnp.pi)
        x = jax.random.uniform(rng, shape, minval=-pi, maxval=pi, dtype=jnp.float32)
        return x * _to_mask(mask, x)

    def random_normal_tangent(
        self,
        rng: jax.Array,
        base_point: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        z = jax.random.normal(rng, shape=base_point.shape, dtype=base_point.dtype)
        return z * _to_mask(mask, z)

    def log_volume(self, mask: Optional[jnp.ndarray] = None):
        if mask is None:
            return self.dim * math.log(2.0 * math.pi)
        return jnp.sum(mask, axis=-1) * math.log(2.0 * math.pi)


class KineticIntrinsicTorus(IntrinsicMaskedTorus):
    """Intrinsic torus with kinetic diagonal metric preconditioning."""

    def __init__(
        self,
        dim: int,
        metric_cutoff: float = 10.0,
        metric_eps: float = 1e-4,
        metric_normalize: bool = True,
        metric_clamp_min: float = 1e-3,
        metric_clamp_max: float | None = None,
        metric_anneal: bool = False,
        metric_anneal_data_lambda: float = 0.0,
        metric_anneal_prior_lambda: float = 1.0,
        metric_anneal_power: float = 1.0,
    ):
        super().__init__(dim=dim)
        self.metric_cutoff = float(metric_cutoff)
        self.metric_eps = float(metric_eps)
        self.metric_normalize = bool(metric_normalize)
        self.metric_clamp_min = float(metric_clamp_min)
        self.metric_clamp_max = metric_clamp_max
        self.metric_anneal = bool(metric_anneal)
        self.metric_anneal_data_lambda = float(metric_anneal_data_lambda)
        self.metric_anneal_prior_lambda = float(metric_anneal_prior_lambda)
        self.metric_anneal_power = float(metric_anneal_power)

    def metric_anneal_lambda_from_sigma2(
        self,
        sigma2: jnp.ndarray,
        sigma2_max: jnp.ndarray | float,
    ) -> jnp.ndarray:
        if not self.metric_anneal:
            return jnp.ones_like(sigma2)
        sigma2_max_t = jnp.asarray(sigma2_max, dtype=sigma2.dtype)
        u = jnp.clip(sigma2 / jnp.clip(sigma2_max_t, a_min=1e-8), a_min=0.0, a_max=1.0)
        u = jnp.power(u, self.metric_anneal_power)
        lam = (
            self.metric_anneal_data_lambda
            + (self.metric_anneal_prior_lambda - self.metric_anneal_data_lambda) * u
        )
        return jnp.clip(lam, a_min=0.0, a_max=1.0)

    def kinetic_metric_diag(
        self,
        angles: jnp.ndarray,
        lengths: jnp.ndarray,
        geo_mask: Optional[jnp.ndarray] = None,
        anneal_lambda: Optional[jnp.ndarray | float] = None,
    ) -> jnp.ndarray:
        g_diag = compute_kinetic_metric_diag(
            angles_batch=angles,
            lengths=lengths,
            geo_mask=geo_mask,
            cutoff=self.metric_cutoff,
            eps=self.metric_eps,
            normalize=self.metric_normalize,
            clamp_min=self.metric_clamp_min,
            clamp_max=self.metric_clamp_max,
        )
        if anneal_lambda is None:
            lam = jnp.ones((g_diag.shape[0],), dtype=g_diag.dtype)
        elif isinstance(anneal_lambda, (float, int)):
            lam = jnp.full((g_diag.shape[0],), float(anneal_lambda), dtype=g_diag.dtype)
        else:
            lam = jnp.asarray(anneal_lambda, dtype=g_diag.dtype).reshape(-1)
        lam = jnp.clip(lam, a_min=0.0, a_max=1.0)[:, None]
        return (1.0 - lam) + lam * g_diag

    def weighted_tangent_sqnorm(
        self,
        residual: jnp.ndarray,
        g_diag: jnp.ndarray,
        geo_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        sq = residual * residual
        if geo_mask is not None:
            sq = sq * jnp.asarray(geo_mask, dtype=sq.dtype)
        out = jnp.sum(sq * g_diag, axis=-1)
        if geo_mask is not None:
            denom = jnp.clip(jnp.sum(geo_mask, axis=-1), a_min=1.0)
            out = out / denom
        return out


class ExtrinsicMaskedTorus:
    """Product of circles S1 in extrinsic cos/sin coordinates with masking."""

    def __init__(self, dim: int):
        self.dim = int(dim)  # number of torsion angles
        self.extrinsic_dim = 2 * self.dim
        self.metric = IntrinsicTorusMetric()

    @staticmethod
    def _reshape_pairs(x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(*x.shape[:-1], -1, 2)

    @staticmethod
    def _flatten_pairs(x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(*x.shape[:-2], -1)

    def projection(self, point: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        p = self._reshape_pairs(point)
        norm = jnp.linalg.norm(p, axis=-1, keepdims=True)
        p = p / jnp.clip(norm, a_min=1e-8)
        out = self._flatten_pairs(p)
        return out * _to_mask(mask, out)

    def to_tangent(
        self,
        vector: jnp.ndarray,
        base_point: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        v = self._reshape_pairs(vector)
        b = self._reshape_pairs(base_point)
        inner = jnp.sum(v * b, axis=-1, keepdims=True)
        t = v - inner * b
        out = self._flatten_pairs(t)
        return out * _to_mask(mask, out)

    def log(
        self,
        point: jnp.ndarray,
        base_point: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        p = self._reshape_pairs(point)
        b = self._reshape_pairs(base_point)
        theta_p = jnp.arctan2(p[..., 1], p[..., 0])
        theta_b = jnp.arctan2(b[..., 1], b[..., 0])
        dtheta = jnp.arctan2(jnp.sin(theta_p - theta_b), jnp.cos(theta_p - theta_b))
        e_theta = jnp.stack([-b[..., 1], b[..., 0]], axis=-1)
        out = self._flatten_pairs(e_theta * dtheta[..., None])
        return out * _to_mask(mask, out)

    def exp(
        self,
        tangent_vec: jnp.ndarray,
        base_point: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        t = self.to_tangent(tangent_vec, base_point, mask=mask)
        tv = self._reshape_pairs(t)
        b = self._reshape_pairs(base_point)
        theta_b = jnp.arctan2(b[..., 1], b[..., 0])
        e_theta = jnp.stack([-b[..., 1], b[..., 0]], axis=-1)
        dtheta = jnp.sum(tv * e_theta, axis=-1)
        theta_new = theta_b + dtheta
        out_pairs = jnp.stack([jnp.cos(theta_new), jnp.sin(theta_new)], axis=-1)
        out = self._flatten_pairs(out_pairs)
        return self.projection(out, mask=mask)

    def random_uniform(
        self,
        rng: jax.Array,
        shape: tuple[int, int],
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        batch, d = shape
        if d != self.extrinsic_dim:
            raise ValueError(f"Expected shape[-1]={self.extrinsic_dim}, got {d}")
        theta = jax.random.uniform(
            rng, (batch, self.dim), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32
        )
        out = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1).reshape(
            batch, self.extrinsic_dim
        )
        return out * _to_mask(mask, out)

    def random_normal_tangent(
        self,
        rng: jax.Array,
        base_point: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        b = self._reshape_pairs(base_point)
        z = jax.random.normal(rng, shape=b.shape[:-1], dtype=base_point.dtype)
        e_theta = jnp.stack([-b[..., 1], b[..., 0]], axis=-1)
        out = self._flatten_pairs(e_theta * z[..., None])
        return out * _to_mask(mask, out)

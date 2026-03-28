"""JAX kinetic/contact-proxy metric utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from diffgeo.angles_and_coords import angles_tensor_to_coords

# Max kinematics batch (B * torsion_chunk). Same role as the torch version.
_KIN_BATCH = 4096


def _compute_contact_proxy_metric_batch_impl(
    angles_batch: jnp.ndarray,
    lengths: jnp.ndarray,
    cutoff: float = 10.0,
    eps: float = 1e-4,
    kin_batch: int = _KIN_BATCH,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized batched contact-proxy metric in torsion space.

    Args:
        angles_batch: (B, Lmax, 6) canonical angles.
        lengths: (B,) true lengths.

    Returns:
        metric_batch: (B, Dmax), Dmax = 6 * (Lmax - 1), padded with small floor.
        n_contacts: (B,) number of CA contacts used for each sample.
    """
    angles_batch = jnp.asarray(angles_batch, dtype=jnp.float32)
    lengths = jnp.asarray(lengths, dtype=jnp.int32)

    bsz, lmax, nfeats = angles_batch.shape
    if nfeats != 6:
        raise ValueError(f"Expected 6 angle features, got {nfeats}")
    if lmax < 2:
        return (
            jnp.zeros((bsz, 0), dtype=jnp.float32),
            jnp.zeros((bsz,), dtype=jnp.int32),
        )

    dmax = 6 * (lmax - 1)
    ca0 = angles_tensor_to_coords(
        angles_batch,
        center_coords=False,
        return_ca_only=True,
    )  # (B, L, 3)
    ca0 = jnp.nan_to_num(ca0, nan=0.0, posinf=0.0, neginf=0.0)
    if ca0.shape != (bsz, lmax, 3):
        raise ValueError(
            f"angles_tensor_to_coords must return shape {(bsz, lmax, 3)}, got {ca0.shape}"
        )

    # Contact graph on CA coordinates.
    rij = ca0[:, :, None, :] - ca0[:, None, :, :]
    dist = jnp.linalg.norm(rij, axis=-1)

    i_idx = jnp.arange(lmax, dtype=jnp.int32).reshape(1, lmax, 1)
    j_idx = jnp.arange(lmax, dtype=jnp.int32).reshape(1, 1, lmax)
    sep_ok = (j_idx - i_idx) > 2
    tri_ok = j_idx > i_idx
    li = lengths.reshape(-1, 1, 1)
    len_ok = (i_idx < li) & (j_idx < li)
    contact_mask = (dist < cutoff) & sep_ok & tri_ok & len_ok

    # Symmetric adjacency for quadratic-form accumulation.
    adj = contact_mask.astype(jnp.float32) + jnp.swapaxes(contact_mask.astype(jnp.float32), 1, 2)
    degree = jnp.sum(adj, axis=-1)

    csz = max(1, kin_batch // max(1, bsz))
    k_all = jnp.arange(dmax, dtype=jnp.int32)
    res_idx_all = k_all // 6
    ang_idx_all = k_all % 6
    valid_k = k_all[None, :] < (6 * (lengths - 1))[:, None]

    metric_full = jnp.zeros((bsz, dmax), dtype=jnp.float32)

    for start in range(0, dmax, csz):
        end = min(start + csz, dmax)
        c = end - start

        ri = res_idx_all[start:end]
        ai = ang_idx_all[start:end]
        valid_chunk = valid_k[:, start:end].astype(jnp.float32).reshape(bsz, c, 1, 1)

        base = jnp.broadcast_to(angles_batch[:, None, :, :], (bsz, c, lmax, 6))
        res_oh = jax.nn.one_hot(ri, lmax, dtype=jnp.float32)  # (c, L)
        ang_oh = jax.nn.one_hot(ai, 6, dtype=jnp.float32)  # (c, 6)
        delta_chunk = eps * (res_oh[:, :, None] * ang_oh[:, None, :])  # (c, L, 6)
        delta_chunk = delta_chunk[None, :, :, :] * valid_chunk  # (B, c, L, 6)

        ab_p = base + delta_chunk
        ab_m = base - delta_chunk

        ca_p = angles_tensor_to_coords(
            ab_p.reshape(bsz * c, lmax, 6),
            center_coords=False,
            return_ca_only=True,
        ).reshape(bsz, c, lmax, 3)
        ca_m = angles_tensor_to_coords(
            ab_m.reshape(bsz * c, lmax, 6),
            center_coords=False,
            return_ca_only=True,
        ).reshape(bsz, c, lmax, 3)
        ca_p = jnp.nan_to_num(ca_p, nan=0.0, posinf=0.0, neginf=0.0)
        ca_m = jnp.nan_to_num(ca_m, nan=0.0, posinf=0.0, neginf=0.0)

        dca = (ca_p - ca_m) / (2.0 * eps)
        dca = jnp.nan_to_num(dca, nan=0.0, posinf=0.0, neginf=0.0)
        sqnorm = jnp.sum(dca * dca, axis=-1)
        term1 = jnp.sum(sqnorm * degree[:, None, :], axis=-1)
        ax = jnp.einsum("bij,bcjd->bcid", adj, dca)
        term2 = jnp.sum(dca * ax, axis=(-1, -2))
        chunk_metric = jnp.clip(term1 - term2, a_min=0.0)
        chunk_metric = jnp.nan_to_num(chunk_metric, nan=0.0, posinf=0.0, neginf=0.0)
        metric_full = metric_full.at[:, start:end].set(chunk_metric)

    metric_full = metric_full * valid_k.astype(jnp.float32) + 1e-6
    metric_full = jnp.nan_to_num(metric_full, nan=1e-6, posinf=1e6, neginf=1e-6)
    n_contacts = jnp.sum(contact_mask, axis=(-1, -2), dtype=jnp.int32)
    return metric_full, n_contacts


def compute_contact_proxy_metric_batch(
    angles_batch: jnp.ndarray,
    lengths: jnp.ndarray,
    cutoff: float = 10.0,
    eps: float = 1e-4,
    kin_batch: int = _KIN_BATCH,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX/TPU-safe contact-proxy metric.
    """
    return _compute_contact_proxy_metric_batch_impl(
        angles_batch=angles_batch,
        lengths=lengths,
        cutoff=cutoff,
        eps=eps,
        kin_batch=kin_batch,
    )


def compute_kinetic_metric_diag(
    angles_batch: jnp.ndarray,
    lengths: jnp.ndarray,
    geo_mask: Optional[jnp.ndarray] = None,
    cutoff: float = 10.0,
    eps: float = 1e-4,
    normalize: bool = True,
    clamp_min: float = 1e-3,
    clamp_max: Optional[float] = None,
    kin_batch: int = _KIN_BATCH,
) -> jnp.ndarray:
    """
    Compute diagonal kinetic metric weights g(theta) over torsion dimensions.

    Note:
        Behavior intentionally matches the active torch implementation, where
        normalization/clamping and geo_mask reweighting are currently disabled.
    """
    geo_mask_f = None if geo_mask is None else jnp.asarray(geo_mask, dtype=jnp.float32)
    g_diag, _ = compute_contact_proxy_metric_batch(
        angles_batch=angles_batch,
        lengths=lengths,
        cutoff=cutoff,
        eps=eps,
        kin_batch=kin_batch,
    )
    g_diag = jnp.nan_to_num(g_diag, nan=1.0, posinf=1e6, neginf=1.0)
    if normalize:
        if geo_mask_f is not None:
            denom = jnp.clip(jnp.sum(geo_mask_f, axis=-1, keepdims=True), a_min=1.0)
            mean_g = jnp.sum(g_diag * geo_mask_f, axis=-1, keepdims=True) / denom
        else:
            mean_g = jnp.mean(g_diag, axis=-1, keepdims=True)
        g_diag = g_diag / jnp.clip(mean_g, a_min=1e-8)
    g_diag = jnp.clip(g_diag, a_min=clamp_min, a_max=(jnp.inf if clamp_max is None else clamp_max))
    if geo_mask_f is not None:
        # Keep padded dimensions neutral under diagonal weighting.
        g_diag = g_diag * geo_mask_f + (1.0 - geo_mask_f)
    return g_diag

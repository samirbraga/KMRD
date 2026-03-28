"""JAX coordinate reconstruction utilities (angles -> backbone coords)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import jax.numpy as jnp
from jax import lax

EXHAUSTIVE_ANGLES = ("phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA")

N_CA_LENGTH = 1.46
CA_C_LENGTH = 1.54
C_N_LENGTH = 1.34

N_INIT = jnp.array([17.047, 14.099, 3.625], dtype=jnp.float32)
CA_INIT = jnp.array([16.967, 12.784, 4.338], dtype=jnp.float32)
C_INIT = jnp.array([15.685, 12.755, 5.133], dtype=jnp.float32)


def _safe_normalize(v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    n = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.clip(n, a_min=eps)


def _place_dihedral(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    bond_angle: jnp.ndarray,
    bond_length: jnp.ndarray,
    torsion_angle: jnp.ndarray,
) -> jnp.ndarray:
    """Place the next point using NERF constraints."""
    ab = b - a
    bc = c - b
    bc = _safe_normalize(bc)
    n = jnp.cross(ab, bc, axis=-1)
    n = _safe_normalize(n)
    nbc = jnp.cross(n, bc, axis=-1)

    m = jnp.stack([bc, nbc, n], axis=-1)  # (..., 3, 3)
    d_local = jnp.stack(
        [
            -bond_length * jnp.cos(bond_angle),
            bond_length * jnp.cos(torsion_angle) * jnp.sin(bond_angle),
            bond_length * jnp.sin(torsion_angle) * jnp.sin(bond_angle),
        ],
        axis=a.ndim - 1,
    )
    # Match torch path: matmul with a column-like vector then squeeze.
    d = jnp.matmul(m, d_local)
    d = jnp.squeeze(d, axis=-1)
    return jnp.nan_to_num(d + c, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_bond_array(v: Union[float, jnp.ndarray], ref: jnp.ndarray) -> jnp.ndarray:
    if isinstance(v, float):
        return jnp.full_like(ref, v)
    return jnp.asarray(v, dtype=ref.dtype)


def nerf_build_batch(
    phi: jnp.ndarray,
    psi: jnp.ndarray,
    omega: jnp.ndarray,
    bond_angle_n_ca_c: jnp.ndarray,  # theta1 (tau)
    bond_angle_ca_c_n: jnp.ndarray,  # theta2 (CA:C:1N)
    bond_angle_c_n_ca: jnp.ndarray,  # theta3 (C:1N:1CA)
    bond_len_n_ca: Union[float, jnp.ndarray] = N_CA_LENGTH,
    bond_len_ca_c: Union[float, jnp.ndarray] = CA_C_LENGTH,
    bond_len_c_n: Union[float, jnp.ndarray] = C_N_LENGTH,
    return_ca_only: bool = False,
) -> jnp.ndarray:
    """Build backbone coordinates for batched torsion inputs."""
    phi = jnp.asarray(phi, dtype=jnp.float32)
    psi = jnp.asarray(psi, dtype=jnp.float32)
    omega = jnp.asarray(omega, dtype=jnp.float32)
    if phi.ndim != 2 or psi.ndim != 2 or omega.ndim != 2:
        raise ValueError("phi/psi/omega must have shape (B, L)")
    if phi.shape != psi.shape or phi.shape != omega.shape:
        raise ValueError("phi/psi/omega shapes must match")

    batch, seq_len = phi.shape
    phi_t = phi[:, 1:]
    psi_t = psi[:, :-1]
    omega_t = omega[:, :-1]
    steps = phi_t.shape[1]

    bond_len_n_ca = _ensure_bond_array(bond_len_n_ca, phi)
    bond_len_ca_c = _ensure_bond_array(bond_len_ca_c, phi)
    bond_len_c_n = _ensure_bond_array(bond_len_c_n, phi)

    n0 = jnp.broadcast_to(N_INIT[None, :], (batch, 3))
    ca0 = jnp.broadcast_to(CA_INIT[None, :], (batch, 3))
    c0 = jnp.broadcast_to(C_INIT[None, :], (batch, 3))

    scan_inputs = (
        jnp.swapaxes(bond_angle_ca_c_n[:, :steps], 0, 1)[..., None],
        jnp.swapaxes(bond_len_c_n[:, :steps], 0, 1)[..., None],
        jnp.swapaxes(psi_t, 0, 1)[..., None],
        jnp.swapaxes(bond_angle_c_n_ca[:, :steps], 0, 1)[..., None],
        jnp.swapaxes(bond_len_n_ca[:, :steps], 0, 1)[..., None],
        jnp.swapaxes(omega_t, 0, 1)[..., None],
        jnp.swapaxes(bond_angle_n_ca_c[:, :steps], 0, 1)[..., None],
        jnp.swapaxes(bond_len_ca_c[:, :steps], 0, 1)[..., None],
        jnp.swapaxes(phi_t, 0, 1)[..., None],
    )

    def _step(carry, xs):
        n_prev, ca_prev, c_prev = carry
        (
            ba_cacn,
            bl_cn,
            psi_i,
            ba_cnca,
            bl_nca,
            omega_i,
            ba_ncac,
            bl_cac,
            phi_i,
        ) = xs

        n_new = _place_dihedral(n_prev, ca_prev, c_prev, ba_cacn, bl_cn, psi_i)
        ca_new = _place_dihedral(ca_prev, c_prev, n_new, ba_cnca, bl_nca, omega_i)
        c_new = _place_dihedral(c_prev, n_new, ca_new, ba_ncac, bl_cac, phi_i)
        return (n_new, ca_new, c_new), (n_new, ca_new, c_new)

    _, scan_out = lax.scan(_step, (n0, ca0, c0), scan_inputs)
    n_steps, ca_steps, c_steps = scan_out  # (steps, B, 3)
    n_steps = jnp.swapaxes(n_steps, 0, 1)  # (B, steps, 3)
    ca_steps = jnp.swapaxes(ca_steps, 0, 1)
    c_steps = jnp.swapaxes(c_steps, 0, 1)

    if return_ca_only:
        return jnp.concatenate([ca0[:, None, :], ca_steps], axis=1)

    init_triplet = jnp.stack([n0, ca0, c0], axis=1)  # (B, 3, 3)
    step_triplets = jnp.stack([n_steps, ca_steps, c_steps], axis=2).reshape(batch, steps * 3, 3)
    return jnp.concatenate([init_triplet, step_triplets], axis=1)


def angles_tensor_to_coords(
    angles: jnp.ndarray,
    center_coords: bool = True,
    angle_names: Optional[Sequence[str]] = None,
    return_ca_only: bool = False,
) -> jnp.ndarray:
    """
    Convert canonical angle tensors to backbone coordinates.

    Args:
        angles: (L, F) or (B, L, F)
        center_coords: center each structure by subtracting mean coordinate.
        angle_names: ordering of the feature axis; defaults to EXHAUSTIVE_ANGLES.
        return_ca_only: if True, return only CA coords of shape (B, L, 3)/(L, 3).
    """
    if angle_names is None:
        angle_names = EXHAUSTIVE_ANGLES

    required = ("phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA")
    name_to_idx = {n: i for i, n in enumerate(angle_names)}
    missing = [n for n in required if n not in name_to_idx]
    if missing:
        raise ValueError(f"Missing required angle names: {missing}")

    angles = jnp.asarray(angles, dtype=jnp.float32)
    if angles.ndim == 2:
        angles_b = angles[None, ...]
        squeeze_out = True
    elif angles.ndim == 3:
        angles_b = angles
        squeeze_out = False
    else:
        raise ValueError(f"Expected shape (L,F) or (B,L,F), got {angles.shape}")

    phi = angles_b[..., name_to_idx["phi"]]
    psi = angles_b[..., name_to_idx["psi"]]
    omega = angles_b[..., name_to_idx["omega"]]
    tau = angles_b[..., name_to_idx["tau"]]
    ca_c_1n = angles_b[..., name_to_idx["CA:C:1N"]]
    c_1n_1ca = angles_b[..., name_to_idx["C:1N:1CA"]]

    coords = nerf_build_batch(
        phi=phi,
        psi=psi,
        omega=omega,
        bond_angle_n_ca_c=tau,
        bond_angle_ca_c_n=ca_c_1n,
        bond_angle_c_n_ca=c_1n_1ca,
        return_ca_only=return_ca_only,
    )

    if center_coords:
        coords = coords - jnp.mean(coords, axis=1, keepdims=True)
    coords = jnp.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

    if squeeze_out:
        return coords[0]
    return coords

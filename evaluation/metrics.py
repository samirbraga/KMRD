"""Validation reference and KL helpers."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def decode_sample_to_angles(x: np.ndarray, length: int, n_feats: int = 6) -> np.ndarray:
    n = int((length - 1) * n_feats)
    vals = x[:n]
    vals = np.pad(vals, (1, n_feats - 1), mode="constant", constant_values=0.0)
    return vals.reshape(-1, n_feats).astype(np.float32, copy=False)


def kl_from_empirical(sampled: np.ndarray, reference: np.ndarray, nbins: int) -> float:
    hist_s, edges = np.histogram(sampled, bins=nbins, range=(-math.pi, math.pi), density=False)
    hist_r, _ = np.histogram(reference, bins=edges, density=False)
    p = hist_s.astype(np.float64) + 1.0
    q = hist_r.astype(np.float64) + 1.0
    p /= np.sum(p)
    q /= np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def collect_reference_angles(
    dataset,
    limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    stacked: list[np.ndarray] = []
    lengths: list[int] = []
    n = len(dataset) if limit <= 0 else min(len(dataset), limit)
    for i in range(n):
        item = dataset[i]
        lengths.append(int(item["lengths"]))
        angles = item["angles"][:-1, :]
        mask = item["geo_mask"].reshape(-1, 6).astype(bool)
        valid_rows = mask.all(axis=1)
        if np.any(valid_rows):
            stacked.append(angles[valid_rows].astype(np.float32, copy=False))
    if not stacked:
        raise RuntimeError("Validation reference is empty.")
    return np.concatenate(stacked, axis=0), np.asarray(lengths, dtype=np.int32)


def compute_val_kl(
    *,
    params,
    cfg,
    reference_angles: np.ndarray,
    val_lengths: np.ndarray,
    epoch: int,
    sample_fn,
) -> float:
    if cfg.val_kl_samples <= 0:
        return float("inf")
    d = 6 * (cfg.max_seq_len - 1)
    rng_np = np.random.default_rng(cfg.seed + 1009 * epoch)
    sampled_lengths = rng_np.choice(val_lengths, size=cfg.val_kl_samples, replace=True)

    rng_jax = jax.random.PRNGKey(cfg.seed + 811 * epoch)
    generated: list[np.ndarray] = []
    bs = max(1, int(cfg.val_kl_batch_size))

    for start in range(0, cfg.val_kl_samples, bs):
        batch_lengths = sampled_lengths[start : start + bs]
        b = len(batch_lengths)
        mask = np.zeros((bs, d), dtype=np.float32)
        for i, length_i in enumerate(batch_lengths):
            mask[i, : (int(length_i) - 1) * 6] = 1.0
        rng_jax, step_rng = jax.random.split(rng_jax)
        x = sample_fn(params, jnp.asarray(mask), step_rng)
        x_np = np.asarray(jax.device_get(x), dtype=np.float32)[:b]
        for i in range(b):
            angles = decode_sample_to_angles(x_np[i], int(batch_lengths[i]), n_feats=6)
            generated.append(angles)

    sampled_angles = np.concatenate(generated, axis=0)
    kl_vals = [
        kl_from_empirical(sampled_angles[:, i], reference_angles[:, i], nbins=int(cfg.val_kl_bins))
        for i in range(6)
    ]
    return float(np.mean(kl_vals))


def params_for_eval(state: Any, use_ema: bool):
    if use_ema and getattr(state, "ema_params", None) is not None:
        return state.ema_params
    return state.params


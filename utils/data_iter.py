"""Dataset batch iterator utilities."""

from __future__ import annotations

from typing import Iterable

import jax.numpy as jnp
import numpy as np


def batch_iter(
    dataset,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterable[dict[str, jnp.ndarray]]:
    idx = np.arange(len(dataset))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sl = idx[start : start + batch_size]
        if sl.size == 0:
            continue
        items = [dataset[int(i)] for i in sl]
        angles = np.stack([it["angles"] for it in items], axis=0).astype(np.float32, copy=False)
        geo_mask = np.stack([it["geo_mask"] for it in items], axis=0).astype(np.float32, copy=False)
        yield {
            "angles": jnp.asarray(angles),
            "geo_mask": jnp.asarray(geo_mask),
        }


def batch_iter_sharded(
    dataset,
    global_batch_size: int,
    rng: np.random.Generator,
    n_devices: int,
    shuffle: bool = True,
) -> Iterable[dict[str, jnp.ndarray]]:
    if global_batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size ({global_batch_size}) must be divisible by device_count ({n_devices})"
        )
    per_device_batch = global_batch_size // n_devices
    for batch in batch_iter(dataset, global_batch_size, rng, shuffle=shuffle):
        b = batch["angles"].shape[0]
        if b != global_batch_size:
            continue
        yield {
            "angles": batch["angles"].reshape(
                n_devices, per_device_batch, *batch["angles"].shape[1:]
            ),
            "geo_mask": batch["geo_mask"].reshape(
                n_devices, per_device_batch, *batch["geo_mask"].shape[1:]
            ),
        }


def batch_iter_for_mode(
    dataset,
    batch_size: int,
    rng: np.random.Generator,
    distributed: bool,
    n_devices: int,
    shuffle: bool = True,
) -> Iterable[dict[str, jnp.ndarray]]:
    if distributed:
        return batch_iter_sharded(
            dataset=dataset,
            global_batch_size=batch_size,
            rng=rng,
            n_devices=n_devices,
            shuffle=shuffle,
        )
    return batch_iter(
        dataset=dataset,
        batch_size=batch_size,
        rng=rng,
        shuffle=shuffle,
    )

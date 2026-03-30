"""Simple prior distributions used in RDM."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class UniformDistribution:
    """Uniform prior on a compact manifold."""

    def __init__(self, manifold):
        self.manifold = manifold

    def sample(
        self,
        rng: jax.Array,
        shape: tuple[int, int],
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return self.manifold.random_uniform(rng, shape=shape, mask=mask)

    def log_prob(self, z: jnp.ndarray):
        return -np.ones((z.shape[0],), dtype=np.float32) * self.manifold.log_volume()

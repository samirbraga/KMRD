"""Compatibility shim; canonical manifold implementations live in diffgeo.manifold."""

from diffgeo.manifold import (
    ExtrinsicMaskedTorus,
    IntrinsicMaskedTorus,
    KineticIntrinsicTorus,
    to_angles_lengths,
)

__all__ = [
    "ExtrinsicMaskedTorus",
    "IntrinsicMaskedTorus",
    "KineticIntrinsicTorus",
    "to_angles_lengths",
]

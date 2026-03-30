"""DiffGeo package."""

from .angles_and_coords import angles_tensor_to_coords
from .kinetic_metric import (
    compute_contact_proxy_metric_batch,
    compute_kinetic_metric_diag,
)
from .manifold import (
    ExtrinsicMaskedTorus,
    IntrinsicMaskedTorus,
    KineticIntrinsicTorus,
    to_angles_lengths,
)

__all__ = [
    "compute_contact_proxy_metric_batch",
    "compute_kinetic_metric_diag",
    "angles_tensor_to_coords",
    "ExtrinsicMaskedTorus",
    "IntrinsicMaskedTorus",
    "KineticIntrinsicTorus",
    "to_angles_lengths",
]

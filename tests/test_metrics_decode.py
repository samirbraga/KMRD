from __future__ import annotations

import numpy as np

from evaluation.metrics import decode_sample_to_angles


def test_decode_sample_to_angles_extrinsic_uses_atan2_pairwise() -> None:
    length = 4
    n_feats = 6
    n = (length - 1) * n_feats
    theta = np.linspace(-np.pi + 0.1, np.pi - 0.1, n, dtype=np.float32)
    x_ext = np.stack([np.cos(theta), np.sin(theta)], axis=-1).reshape(-1)

    angles = decode_sample_to_angles(
        x_ext,
        length=length,
        n_feats=n_feats,
        coordinate_system="extrinsic",
    )
    expected = np.pad(theta, (1, n_feats - 1), mode="constant", constant_values=0.0).reshape(
        length, n_feats
    )
    assert np.allclose(angles, expected, atol=1e-6)

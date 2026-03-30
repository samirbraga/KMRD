from __future__ import annotations

import numpy as np
import pytest

from foldingdiff.dataset import CathCanonicalAnglesOnlyDataset


def test_dataset_item_shapes_and_masks() -> None:
    pad = 64
    try:
        dset = CathCanonicalAnglesOnlyDataset(
            pdbs="cath",
            split="train",
            pad=pad,
            min_length=2,
            trim_strategy="leftalign",
            toy=8,
            zero_center=False,
            use_cache=False,
            num_workers=1,
        )
    except FileNotFoundError as exc:
        pytest.skip(f"Local dataset not available: {exc}")

    if len(dset) == 0:
        pytest.skip("Dataset returned zero samples in this environment.")

    item = dset[0]
    assert set(item.keys()) == {"angles", "lengths", "cossin", "attn_mask", "geo_mask"}
    assert item["angles"].shape == (pad, 6)
    assert item["attn_mask"].shape == (pad,)
    assert item["geo_mask"].shape == ((pad - 1) * 6,)
    assert item["cossin"].shape == (2 * (pad - 1) * 6,)

    length = int(item["lengths"])
    assert 1 <= length <= pad
    geo_mask_sum = int(np.sum(item["geo_mask"]))
    assert geo_mask_sum == max(length - 1, 0) * 6


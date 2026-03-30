"""Checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import flax.serialization


def save_checkpoint(
    path: Path,
    state: Any,
    epoch: int,
    metrics: dict[str, float],
    cfg,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(state))

    meta = {
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg.model_dump(),
    }
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

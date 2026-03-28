"""W&B helpers for config resume, checkpoint artifacts, and run state."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import wandb

from utils.config import TrainConfig


def parse_wandb_run_path(run_ref: str, entity: str, project: str) -> tuple[str, str, str]:
    parts = run_ref.strip().split("/")
    if len(parts) == 1:
        return entity, project, parts[0]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(
        f"Invalid resume_run={run_ref!r}. Use run_id or 'entity/project/run_id'."
    )


def load_config_from_resumed_run(cfg: TrainConfig) -> TrainConfig:
    if not cfg.resume_run:
        return cfg
    if cfg.wandb_mode == "disabled":
        raise ValueError("resume_run requires wandb_mode != disabled")

    entity, project, run_id = parse_wandb_run_path(
        cfg.resume_run, cfg.wandb_entity, cfg.wandb_project
    )
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    known_fields = set(TrainConfig.model_fields.keys())
    resumed_values = {k: v for k, v in run.config.items() if k in known_fields}
    merged = cfg.model_dump()
    merged.update(resumed_values)
    merged["resume_run"] = run_id
    merged["wandb_entity"] = entity
    merged["wandb_project"] = project
    return TrainConfig(**merged)


def download_wandb_checkpoint(
    *,
    entity: str,
    project: str,
    run_id: str,
    artifact_name: str,
    out_dir: Path,
) -> Path:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    artifact = None
    artifact_path = None

    try:
        logged = list(run.logged_artifacts())
    except Exception:
        logged = []
    own_model_artifacts = [
        a for a in logged if f"{entity}/{project}/{artifact_name}:" in getattr(a, "name", "")
    ]

    def _alias_values(artifact_obj) -> list[str]:
        vals: list[str] = []
        for x in getattr(artifact_obj, "aliases", []):
            vals.append(x if isinstance(x, str) else getattr(x, "alias", ""))
        return vals

    best_candidates = [a for a in own_model_artifacts if "best" in _alias_values(a)]
    latest_candidates = [a for a in own_model_artifacts if "latest" in _alias_values(a)]
    if best_candidates:
        artifact = best_candidates[-1]
        artifact_path = artifact.name
    elif latest_candidates:
        artifact = latest_candidates[-1]
        artifact_path = artifact.name
    elif own_model_artifacts:
        artifact = own_model_artifacts[-1]
        artifact_path = artifact.name

    if artifact is None:
        for alias in ("best", "latest", f"run-{run_id}"):
            candidate = f"{entity}/{project}/{artifact_name}:{alias}"
            try:
                artifact = api.artifact(candidate)
                artifact_path = candidate
                break
            except Exception:
                continue
    if artifact is None:
        raise RuntimeError(
            f"Could not find checkpoint artifact for run {entity}/{project}/{run_id} and name {artifact_name}."
        )

    # Download to an isolated folder to avoid mixing with local training checkpoints.
    resume_root = out_dir / ".wandb_resume" / run_id
    if resume_root.exists():
        shutil.rmtree(resume_root)
    resume_root.mkdir(parents=True, exist_ok=True)
    download_dir = Path(artifact.download(root=str(resume_root)))
    msgpacks = sorted(download_dir.glob("*.msgpack"))
    if not msgpacks:
        raise RuntimeError(
            f"Downloaded {artifact_path}, but no .msgpack checkpoint was found in {download_dir}."
        )
    # Artifact should include exactly one checkpoint; if not, pick highest epoch.
    if len(msgpacks) == 1:
        return msgpacks[0]

    def _epoch_from_name(path: Path) -> int:
        stem = path.stem
        parts = stem.split("_")
        if parts and parts[-1].isdigit():
            return int(parts[-1])
        return -1

    return max(msgpacks, key=_epoch_from_name)


def get_resume_epoch_from_wandb(entity: str, project: str, run_id: str) -> int:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    step = run.summary.get("_step", None)
    if step is None:
        step = getattr(run, "lastHistoryStep", None)
    return int(step) if step is not None else 0


def get_best_val_loss_from_wandb(entity: str, project: str, run_id: str) -> float:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    val = run.summary.get("best_val_loss", None)
    if val is None:
        return float("inf")
    try:
        return float(val)
    except Exception:
        return float("inf")


def get_best_scalar_from_wandb(entity: str, project: str, run_id: str, key: str) -> float:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    val = run.summary.get(key, None)
    if val is None:
        return float("inf")
    try:
        return float(val)
    except Exception:
        return float("inf")


def log_checkpoint_artifact(
    run: Any,
    *,
    ckpt_path: Path,
    epoch: int,
    is_best: bool,
    artifact_name: str,
) -> None:
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata={"run_id": run.id, "epoch": epoch},
    )
    artifact.add_file(str(ckpt_path), name=ckpt_path.name)
    meta_path = ckpt_path.with_suffix(".json")
    if meta_path.exists():
        artifact.add_file(str(meta_path), name=meta_path.name)
    aliases = ["latest", f"epoch-{epoch:04d}", f"run-{run.id}"]
    if is_best:
        aliases.append("best")
    run.log_artifact(artifact, aliases=aliases)

"""Train score-based diffusion in intrinsic torsion coordinates (JAX/Flax)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import flax.jax_utils as flax_jax_utils
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from foldingdiff.bert_for_diffusion import BertDiffusionConfig, BertForDiffusion
from foldingdiff.dataset import CathCanonicalAnglesOnlyDataset
from utils.config import TrainConfig
from utils.train import (
    ScoreTrainConfig,
    TrainState,
    create_train_state,
    make_train_step,
    make_train_step_pmap,
    train_one_epoch,
    train_one_epoch_pmap,
)


def _parse_wandb_run_path(run_ref: str, entity: str, project: str) -> tuple[str, str, str]:
    parts = run_ref.strip().split("/")
    if len(parts) == 1:
        return entity, project, parts[0]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(
        f"Invalid resume_run={run_ref!r}. Use run_id or 'entity/project/run_id'."
    )


def _load_config_from_resumed_run(cfg: TrainConfig) -> TrainConfig:
    if not cfg.resume_run:
        return cfg
    if cfg.wandb_mode == "disabled":
        raise ValueError("resume_run requires wandb_mode != disabled")

    entity, project, run_id = _parse_wandb_run_path(
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


def _download_wandb_checkpoint(
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
        # Fallback path if API cannot enumerate run artifacts.
        for alias in (f"run-{run_id}", "best", "latest"):
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

    download_dir = Path(artifact.download(root=str(out_dir)))
    msgpacks = sorted(download_dir.glob("*.msgpack"))
    if not msgpacks:
        raise RuntimeError(
            f"Downloaded {artifact_path}, but no .msgpack checkpoint was found in {download_dir}."
        )
    return msgpacks[0]


def _get_resume_epoch_from_wandb(entity: str, project: str, run_id: str) -> int:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    step = run.summary.get("_step", None)
    if step is None:
        step = getattr(run, "lastHistoryStep", None)
    return int(step) if step is not None else 0


def _batch_iter(
    dataset: CathCanonicalAnglesOnlyDataset,
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


def _batch_iter_sharded(
    dataset: CathCanonicalAnglesOnlyDataset,
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
    for batch in _batch_iter(dataset, global_batch_size, rng, shuffle=shuffle):
        b = batch["angles"].shape[0]
        if b != global_batch_size:
            continue
        yield {
            "angles": batch["angles"].reshape(n_devices, per_device_batch, *batch["angles"].shape[1:]),
            "geo_mask": batch["geo_mask"].reshape(n_devices, per_device_batch, *batch["geo_mask"].shape[1:]),
        }


def _save_checkpoint(path: Path, state: TrainState, epoch: int, metrics: dict[str, float], cfg: TrainConfig) -> None:
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


def _log_checkpoint_artifact(
    run: wandb.sdk.wandb_run.Run,
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

def main() -> None:
    cfg = _load_config_from_resumed_run(TrainConfig())
    np_rng = np.random.default_rng(cfg.seed)
    jax_rng = jax.random.PRNGKey(cfg.seed)
    wandb_run = None

    if cfg.wandb_mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.model_dump(),
            mode=cfg.wandb_mode,
            id=cfg.resume_run,
            resume="must" if cfg.resume_run else None,
        )

    train_ds = CathCanonicalAnglesOnlyDataset(
        pdbs=cfg.pdbs,
        split="train",
        pad=cfg.max_seq_len,
        min_length=cfg.min_seq_len,
        trim_strategy="leftalign",
        toy=cfg.toy,
        zero_center=cfg.zero_center,
        use_cache=True,
        num_workers=cfg.dataset_workers,
    )
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Check data path/split constraints.")

    model_cfg = BertDiffusionConfig(
        num_attention_heads=cfg.net_size * 4,
        hidden_size=cfg.net_size * 128,
        intermediate_size=cfg.net_size * 256,
        num_hidden_layers=cfg.net_size * 4,
        hidden_dropout_prob=cfg.dropout,
        attention_probs_dropout_prob=cfg.dropout,
        input_feat_dim=6,  # intrinsic torsion coordinates
        torsion_feat_dim=6,
        condition_on_g_diag=(
            cfg.metric_condition_model and cfg.metric_type == "kinetic_diag"
        ),
    )
    model = BertForDiffusion(config=model_cfg)
    n_devices = jax.local_device_count()
    use_distributed = bool(cfg.distributed and n_devices > 1)

    init_batch = next(_batch_iter(train_ds, batch_size=min(cfg.batch_size, len(train_ds)), rng=np_rng, shuffle=False))
    sample_x = init_batch["angles"][:, :-1, :].reshape(init_batch["angles"].shape[0], -1)
    state_single = create_train_state(
        model=model,
        rng=jax_rng,
        sample_x=sample_x,
        sample_mask=init_batch["geo_mask"],
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    start_epoch = 1
    if cfg.resume_run:
        resume_ckpt = _download_wandb_checkpoint(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            run_id=cfg.resume_run,
            artifact_name=cfg.checkpoint_artifact_name,
            out_dir=Path(cfg.weights_path),
        )
        state_single = flax.serialization.from_bytes(state_single, resume_ckpt.read_bytes())
        start_epoch = _get_resume_epoch_from_wandb(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            run_id=cfg.resume_run,
        ) + 1
        print(f"resumed_checkpoint={resume_ckpt}")
        print(f"resumed_start_epoch={start_epoch}")
    state: TrainState = flax_jax_utils.replicate(state_single) if use_distributed else state_single

    loss_cfg = ScoreTrainConfig(
        coordinate_system="intrinsic",
        metric_type=cfg.metric_type,
        metric_cutoff=cfg.metric_cutoff,
        metric_eps=cfg.metric_eps,
        metric_normalize=cfg.metric_normalize,
        metric_clamp_min=cfg.metric_clamp_min,
        metric_clamp_max=cfg.metric_clamp_max,
        metric_anneal=cfg.metric_anneal,
        metric_anneal_data_lambda=cfg.metric_anneal_data_lambda,
        metric_anneal_prior_lambda=cfg.metric_anneal_prior_lambda,
        metric_anneal_power=cfg.metric_anneal_power,
        beta_0=cfg.beta_0,
        beta_f=cfg.beta_f,
        t_eps=cfg.t_eps,
        n_wrap=cfg.n_wrap,
        max_grad_norm=cfg.grad_norm,
    )

    ckpt_root = Path(cfg.weights_path)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Training start: n_train={len(train_ds)} batch={cfg.batch_size} "
        f"metric={cfg.metric_type} cond_g={model_cfg.condition_on_g_diag} "
        f"devices={n_devices} distributed={use_distributed}"
    )
    train_step_fn = make_train_step_pmap(loss_cfg) if use_distributed else make_train_step(loss_cfg)

    best_loss = float("inf")
    for epoch in range(start_epoch, cfg.epochs + 1):
        if use_distributed:
            batches = _batch_iter_sharded(
                train_ds,
                global_batch_size=cfg.batch_size,
                rng=np_rng,
                n_devices=n_devices,
                shuffle=True,
            )
            state, metrics = train_one_epoch_pmap(
                state,
                batches,
                loss_cfg,
                epoch=epoch,
                log_every=cfg.train_log_every,
                train_step_fn=train_step_fn,
            )
        else:
            batches = _batch_iter(train_ds, batch_size=cfg.batch_size, rng=np_rng, shuffle=True)
            state, metrics = train_one_epoch(
                state,
                batches,
                loss_cfg,
                epoch=epoch,
                log_every=cfg.train_log_every,
                train_step_fn=train_step_fn,
            )
        print(
            f"epoch={epoch:04d} loss={metrics['loss']:.6f} "
            f"g0={metrics['g0_mean']:.4f} gt={metrics['gt_mean']:.4f} sigma2={metrics['sigma2_mean']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "loss": metrics["loss"],
                    "g0_mean": metrics["g0_mean"],
                    "gt_mean": metrics["gt_mean"],
                    "sigma2_mean": metrics["sigma2_mean"],
                },
                step=epoch,
            )
        is_best = float(metrics["loss"]) <= best_loss
        if is_best:
            best_loss = float(metrics["loss"])

        if cfg.save_every_epochs > 0 and (epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs):
            ckpt_path = ckpt_root / f"model_epoch_{epoch:04d}.msgpack"
            save_state = flax_jax_utils.unreplicate(state) if use_distributed else state
            _save_checkpoint(ckpt_path, save_state, epoch, metrics, cfg)
            print(f"saved={ckpt_path}")
            if wandb_run is not None:
                wandb_run.log({"checkpoint_path": str(ckpt_path)}, step=epoch)
                _log_checkpoint_artifact(
                    wandb_run,
                    ckpt_path=ckpt_path,
                    epoch=epoch,
                    is_best=is_best,
                    artifact_name=cfg.checkpoint_artifact_name,
                )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

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
    train_one_epoch,
    train_one_epoch_pmap,
)


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

def main() -> None:
    cfg = TrainConfig()
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
            resume="allow" if cfg.resume_run else None,
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
    state: TrainState = flax_jax_utils.replicate(state_single) if use_distributed else state_single

    loss_cfg = ScoreTrainConfig(
        coordinate_system="intrinsic",
        metric_type=cfg.metric_type,
        metric_cutoff=cfg.metric_cutoff,
        metric_eps=cfg.metric_eps,
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
    for epoch in range(1, cfg.epochs + 1):
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
            )
        else:
            batches = _batch_iter(train_ds, batch_size=cfg.batch_size, rng=np_rng, shuffle=True)
            state, metrics = train_one_epoch(
                state,
                batches,
                loss_cfg,
                epoch=epoch,
                log_every=cfg.train_log_every,
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

        if cfg.save_every_epochs > 0 and (epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs):
            ckpt_path = ckpt_root / f"model_epoch_{epoch:04d}.msgpack"
            save_state = flax_jax_utils.unreplicate(state) if use_distributed else state
            _save_checkpoint(ckpt_path, save_state, epoch, metrics, cfg)
            print(f"saved={ckpt_path}")
            if wandb_run is not None:
                wandb_run.log({"checkpoint_path": str(ckpt_path)}, step=epoch)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

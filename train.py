"""Train score-based diffusion in intrinsic torsion coordinates (JAX/Flax)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import flax.jax_utils as flax_jax_utils
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from RDM.beta_schedule import LinearBetaSchedule
from RDM.losses import get_bridge_loss_fn
from RDM.sde_lib import DiffusionMixture
from RDM.training import intrinsic_to_cossin, make_bridge_train_step, train_one_epoch_bridge
from diffgeo.manifold import ExtrinsicMaskedTorus, IntrinsicMaskedTorus
from foldingdiff.bert_for_diffusion import BertDiffusionConfig, BertForDiffusion
from foldingdiff.dataset import CathCanonicalAnglesOnlyDataset
from score_based.sampling import sample_intrinsic_batch
from score_based.training import (
    ScoreTrainConfig,
    TrainState,
    create_train_state,
    eval_one_epoch,
    eval_one_epoch_pmap,
    make_eval_step,
    make_eval_step_pmap,
    make_train_step,
    make_train_step_pmap,
    train_one_epoch,
    train_one_epoch_pmap,
)
from utils.config import TrainConfig
from utils.wandb import (
    download_wandb_checkpoint,
    get_best_scalar_from_wandb,
    get_best_val_loss_from_wandb,
    get_resume_epoch_from_wandb,
    load_config_from_resumed_run,
    log_checkpoint_artifact,
)


def _build_learning_rate_schedule(
    cfg: TrainConfig,
    *,
    total_steps: int,
) -> float | optax.Schedule:
    if not cfg.lr_sched:
        return float(cfg.learning_rate)

    total_steps = max(1, int(total_steps))
    warmup_steps = int(max(0, cfg.lr_warmup_frac) * total_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    min_lr = float(cfg.learning_rate) * float(cfg.min_lr_ratio)

    if cfg.lr_schedule_type == "linear":
        decay_sched = optax.linear_schedule(
            init_value=float(cfg.learning_rate),
            end_value=min_lr,
            transition_steps=decay_steps,
        )
    else:
        decay_sched = optax.cosine_decay_schedule(
            init_value=float(cfg.learning_rate),
            decay_steps=decay_steps,
            alpha=float(cfg.min_lr_ratio),
        )

    if warmup_steps <= 0:
        return decay_sched

    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=float(cfg.learning_rate),
        transition_steps=max(1, warmup_steps),
    )
    return optax.join_schedules(
        schedules=[warmup_sched, decay_sched],
        boundaries=[warmup_steps],
    )


def _decode_sample_to_angles(x: np.ndarray, length: int, n_feats: int = 6) -> np.ndarray:
    n = int((length - 1) * n_feats)
    vals = x[:n]
    vals = np.pad(vals, (1, n_feats - 1), mode="constant", constant_values=0.0)
    return vals.reshape(-1, n_feats).astype(np.float32, copy=False)


def _collect_reference_angles(
    dataset: CathCanonicalAnglesOnlyDataset,
    limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    stacked: list[np.ndarray] = []
    lengths: list[int] = []
    n = len(dataset) if limit <= 0 else min(len(dataset), limit)
    for i in range(n):
        item = dataset[i]
        lengths.append(int(item["lengths"]))
        angles = item["angles"][:-1, :]
        mask = item["geo_mask"].reshape(-1, 6).astype(bool)
        valid_rows = mask.all(axis=1)
        if np.any(valid_rows):
            stacked.append(angles[valid_rows].astype(np.float32, copy=False))
    if not stacked:
        raise RuntimeError("Validation reference is empty.")
    return np.concatenate(stacked, axis=0), np.asarray(lengths, dtype=np.int32)


def _kl_from_empirical(sampled: np.ndarray, reference: np.ndarray, nbins: int) -> float:
    hist_s, edges = np.histogram(sampled, bins=nbins, range=(-math.pi, math.pi), density=False)
    hist_r, _ = np.histogram(reference, bins=edges, density=False)
    p = hist_s.astype(np.float64) + 1.0
    q = hist_r.astype(np.float64) + 1.0
    p /= np.sum(p)
    q /= np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _sample_batch_eval(
    params,
    model: BertForDiffusion,
    mask: jnp.ndarray,
    cfg: TrainConfig,
    rng: jax.Array,
) -> jnp.ndarray:
    return sample_intrinsic_batch(
        params=params,
        model=model,
        mask=mask,
        rng=rng,
        n_steps=cfg.val_kl_n_steps,
        eps=cfg.val_kl_eps,
        beta_0=cfg.beta_0,
        beta_f=cfg.beta_f,
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
        pc_corrector_steps=0,
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


def _save_checkpoint(
    path: Path,
    state: Any,
    epoch: int,
    metrics: dict[str, float],
    cfg: TrainConfig,
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


def _compute_val_kl(
    *,
    params,
    cfg: TrainConfig,
    reference_angles: np.ndarray,
    val_lengths: np.ndarray,
    epoch: int,
    sample_fn,
) -> float:
    if cfg.val_kl_samples <= 0:
        return float("inf")
    d = 6 * (cfg.max_seq_len - 1)
    rng_np = np.random.default_rng(cfg.seed + 1009 * epoch)
    sampled_lengths = rng_np.choice(val_lengths, size=cfg.val_kl_samples, replace=True)

    rng_jax = jax.random.PRNGKey(cfg.seed + 811 * epoch)
    generated: list[np.ndarray] = []
    bs = max(1, int(cfg.val_kl_batch_size))

    for start in range(0, cfg.val_kl_samples, bs):
        batch_lengths = sampled_lengths[start : start + bs]
        b = len(batch_lengths)
        mask = np.zeros((bs, d), dtype=np.float32)
        for i, length_i in enumerate(batch_lengths):
            mask[i, : (int(length_i) - 1) * 6] = 1.0
        rng_jax, step_rng = jax.random.split(rng_jax)
        x = sample_fn(params, jnp.asarray(mask), step_rng)
        x_np = np.asarray(jax.device_get(x), dtype=np.float32)[:b]
        for i in range(b):
            angles = _decode_sample_to_angles(x_np[i], int(batch_lengths[i]), n_feats=6)
            generated.append(angles)

    sampled_angles = np.concatenate(generated, axis=0)
    kl_vals = [
        _kl_from_empirical(sampled_angles[:, i], reference_angles[:, i], nbins=int(cfg.val_kl_bins))
        for i in range(6)
    ]
    return float(np.mean(kl_vals))


def _params_for_eval(state: TrainState, use_ema: bool):
    if use_ema and state.ema_params is not None:
        return state.ema_params
    return state.params


def main() -> None:
    cfg = load_config_from_resumed_run(TrainConfig())
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
    val_ds = CathCanonicalAnglesOnlyDataset(
        pdbs=cfg.pdbs,
        split="validation",
        pad=cfg.max_seq_len,
        min_length=cfg.min_seq_len,
        trim_strategy="leftalign",
        toy=cfg.toy,
        zero_center=cfg.zero_center,
        use_cache=True,
        num_workers=cfg.dataset_workers,
    )
    val_ref_angles = None
    val_ref_lengths = None
    if cfg.val_kl_enable and len(val_ds) > 0:
        val_ref_angles, val_ref_lengths = _collect_reference_angles(
            val_ds, limit=cfg.val_kl_ref_limit
        )

    bridge_feat_dim = 12 if (
        cfg.training_objective == "bridge_matching" and cfg.bridge_coordinates == "extrinsic"
    ) else 6
    model_cfg = BertDiffusionConfig(
        num_attention_heads=cfg.net_size * 4,
        hidden_size=cfg.net_size * 128,
        intermediate_size=cfg.net_size * 256,
        num_hidden_layers=cfg.net_size * 4,
        hidden_dropout_prob=cfg.dropout,
        attention_probs_dropout_prob=cfg.dropout,
        input_feat_dim=bridge_feat_dim,
        torsion_feat_dim=bridge_feat_dim,
        condition_on_g_diag=(
            cfg.training_objective == "score"
            and cfg.metric_condition_model
            and cfg.metric_type == "kinetic_diag"
        ),
    )
    model = BertForDiffusion(config=model_cfg)
    n_devices = jax.local_device_count()
    use_distributed = bool(cfg.distributed and n_devices > 1)

    init_batch = next(_batch_iter(train_ds, batch_size=min(cfg.batch_size, len(train_ds)), rng=np_rng, shuffle=False))
    sample_x = init_batch["angles"][:, :-1, :].reshape(init_batch["angles"].shape[0], -1)
    sample_mask = init_batch["geo_mask"]
    if cfg.training_objective == "bridge_matching" and cfg.bridge_coordinates == "extrinsic":
        sample_x, sample_mask = intrinsic_to_cossin(sample_x, sample_mask)
    if use_distributed:
        steps_per_epoch = len(train_ds) // cfg.batch_size
    else:
        steps_per_epoch = (len(train_ds) + cfg.batch_size - 1) // cfg.batch_size
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    learning_rate = _build_learning_rate_schedule(cfg, total_steps=total_steps)
    state_single = create_train_state(
        model=model,
        rng=jax_rng,
        sample_x=sample_x,
        sample_mask=sample_mask,
        learning_rate=learning_rate,
        weight_decay=cfg.weight_decay,
        use_ema=cfg.use_ema,
        ema_decay=cfg.ema_decay,
    )
    state_b_single = None
    model_b = None
    if cfg.training_objective == "bridge_matching":
        model_b = BertForDiffusion(config=model_cfg)
        state_b_single = create_train_state(
            model=model_b,
            rng=jax.random.PRNGKey(cfg.seed + 1337),
            sample_x=sample_x,
            sample_mask=sample_mask,
            learning_rate=learning_rate,
            weight_decay=cfg.weight_decay,
            use_ema=cfg.use_ema,
            ema_decay=cfg.ema_decay,
        )
    start_epoch = 1
    if cfg.resume_run:
        if cfg.resume_checkpoint_path:
            resume_ckpt = Path(cfg.resume_checkpoint_path)
            if not resume_ckpt.exists():
                raise FileNotFoundError(f"resume_checkpoint_path not found: {resume_ckpt}")
        else:
            resume_ckpt = download_wandb_checkpoint(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                run_id=cfg.resume_run,
                artifact_name=cfg.checkpoint_artifact_name,
                out_dir=Path(cfg.weights_path),
            )
        ckpt_bytes = resume_ckpt.read_bytes()
        if cfg.training_objective == "bridge_matching":
            raw_state = flax.serialization.msgpack_restore(ckpt_bytes)
            if isinstance(raw_state, dict) and "state_f" in raw_state and "state_b" in raw_state:
                state_single = flax.serialization.from_state_dict(state_single, raw_state["state_f"])
                assert state_b_single is not None
                state_b_single = flax.serialization.from_state_dict(state_b_single, raw_state["state_b"])
            else:
                state_single = flax.serialization.from_bytes(state_single, ckpt_bytes)
        else:
            state_single = flax.serialization.from_bytes(state_single, ckpt_bytes)
        start_epoch = get_resume_epoch_from_wandb(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            run_id=cfg.resume_run,
        ) + 1
        print(f"resumed_checkpoint={resume_ckpt}")
        print(f"resumed_start_epoch={start_epoch}")
    state: TrainState = flax_jax_utils.replicate(state_single) if use_distributed else state_single

    if cfg.training_objective == "bridge_matching":
        if use_distributed:
            raise ValueError("training_objective=bridge_matching currently supports only distributed=false")
        assert model_b is not None and state_b_single is not None

        if cfg.bridge_coordinates == "extrinsic":
            manifold = ExtrinsicMaskedTorus(dim=sample_x.shape[-1] // 2)
            preprocess_fn = intrinsic_to_cossin
        else:
            manifold = IntrinsicMaskedTorus(dim=sample_x.shape[-1])
            preprocess_fn = lambda x, m: (x, m)
        beta_schedule = LinearBetaSchedule(
            tf=1.0,
            t0=0.0,
            beta_0=cfg.beta_0,
            beta_f=cfg.beta_f,
        )
        mix = DiffusionMixture(
            manifold=manifold,
            beta_schedule=beta_schedule,
            prior_type="unif",
            drift_scale=1.0,
            mix_type="log",
        )
        bridge_loss_fn = get_bridge_loss_fn(
            mix=mix,
            model_apply_f=model.apply,
            model_apply_b=model_b.apply,
            reduce_mean=False,
            eps=cfg.t_eps,
            num_steps=cfg.bridge_num_steps,
            weight_type=cfg.bridge_weight_type,
            normalize_by_dim=True,
        )
        bridge_train_step = make_bridge_train_step(
            loss_fn=bridge_loss_fn,
            grad_norm=cfg.grad_norm,
            preprocess_fn=preprocess_fn,
        )
        state_f = state_single
        state_b = state_b_single
        ckpt_root = Path(cfg.weights_path)
        ckpt_root.mkdir(parents=True, exist_ok=True)

        best_bridge_loss = float("inf")
        print(
            f"Training start: objective=bridge_matching n_train={len(train_ds)} batch={cfg.batch_size} "
            f"steps={cfg.bridge_num_steps} weight={cfg.bridge_weight_type} "
            f"coords={cfg.bridge_coordinates}"
        )
        for epoch in range(start_epoch, cfg.epochs + 1):
            batches = _batch_iter(train_ds, batch_size=cfg.batch_size, rng=np_rng, shuffle=True)
            state_f, state_b, metrics = train_one_epoch_bridge(
                state_f=state_f,
                state_b=state_b,
                train_batches=batches,
                train_step_fn=bridge_train_step,
                epoch=epoch,
                log_every=cfg.train_log_every,
            )
            print(
                f"epoch={epoch:04d} loss={metrics['loss']:.6f} "
                f"loss_f={metrics['loss_f']:.6f} loss_b={metrics['loss_b']:.6f}"
            )
            is_best = metrics["loss"] < best_bridge_loss
            if is_best:
                best_bridge_loss = metrics["loss"]

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "loss": metrics["loss"],
                        "bridge_loss_f": metrics["loss_f"],
                        "bridge_loss_b": metrics["loss_b"],
                        "best_bridge_loss": best_bridge_loss,
                    },
                    step=epoch,
                )

            if cfg.save_every_epochs > 0 and (epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs or is_best):
                ckpt_path = ckpt_root / f"model_epoch_{epoch:04d}.msgpack"
                save_state = {"state_f": state_f, "state_b": state_b}
                _save_checkpoint(ckpt_path, save_state, epoch, metrics, cfg)
                print(f"saved={ckpt_path}")
                if wandb_run is not None:
                    wandb_run.log({"checkpoint_path": str(ckpt_path)}, step=epoch)
                    log_checkpoint_artifact(
                        wandb_run,
                        ckpt_path=ckpt_path,
                        epoch=epoch,
                        is_best=is_best,
                        artifact_name=cfg.checkpoint_artifact_name,
                    )
        if wandb_run is not None:
            wandb_run.finish()
        return

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
        eval_use_ema=cfg.eval_use_ema,
    )

    ckpt_root = Path(cfg.weights_path)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Training start: objective=score n_train={len(train_ds)} n_val={len(val_ds)} batch={cfg.batch_size} "
        f"val_batch={cfg.val_batch_size if cfg.val_batch_size > 0 else cfg.batch_size} "
        f"metric={cfg.metric_type} cond_g={model_cfg.condition_on_g_diag} "
        f"devices={n_devices} distributed={use_distributed} "
        f"lr_sched={cfg.lr_sched} lr_type={cfg.lr_schedule_type if cfg.lr_sched else 'constant'} "
        f"use_ema={cfg.use_ema} eval_use_ema={cfg.eval_use_ema}"
    )
    train_step_fn = make_train_step_pmap(loss_cfg) if use_distributed else make_train_step(loss_cfg)
    eval_step_fn = make_eval_step_pmap(loss_cfg) if use_distributed else make_eval_step(loss_cfg)
    val_sample_fn = jax.jit(
        lambda p, m, r: _sample_batch_eval(
            params=p,
            model=model,
            mask=m,
            cfg=cfg,
            rng=r,
        )
    )

    best_val_loss = (
        get_best_val_loss_from_wandb(cfg.wandb_entity, cfg.wandb_project, cfg.resume_run)
        if cfg.resume_run
        else float("inf")
    )
    best_val_kl = (
        get_best_scalar_from_wandb(cfg.wandb_entity, cfg.wandb_project, cfg.resume_run, "best_val_kl")
        if cfg.resume_run
        else float("inf")
    )
    if cfg.best_metric == "val_kl" and not cfg.val_kl_enable:
        raise ValueError("best_metric=val_kl requires val_kl_enable=true")
    val_batch_size = cfg.val_batch_size if cfg.val_batch_size > 0 else cfg.batch_size
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
        val_metrics = None
        val_kl = None
        should_eval = bool(
            cfg.train_val
            and len(val_ds) > 0
            and epoch >= cfg.start_eval_epoch
            and ((epoch - cfg.start_eval_epoch) % max(1, cfg.val_freq) == 0)
        )
        if should_eval:
            if use_distributed:
                val_batches = _batch_iter_sharded(
                    val_ds,
                    global_batch_size=val_batch_size,
                    rng=np_rng,
                    n_devices=n_devices,
                    shuffle=False,
                )
                state, val_metrics = eval_one_epoch_pmap(
                    state,
                    val_batches,
                    loss_cfg,
                    epoch=epoch,
                    log_every=cfg.val_log_every,
                    max_batches=cfg.val_max_batches,
                    eval_step_fn=eval_step_fn,
                )
            else:
                val_batches = _batch_iter(
                    val_ds,
                    batch_size=val_batch_size,
                    rng=np_rng,
                    shuffle=False,
                )
                state, val_metrics = eval_one_epoch(
                    state,
                    val_batches,
                    loss_cfg,
                    epoch=epoch,
                    log_every=cfg.val_log_every,
                    max_batches=cfg.val_max_batches,
                    eval_step_fn=eval_step_fn,
                )
            print(
                f"epoch={epoch:04d} val_loss={val_metrics['loss']:.6f} "
                f"val_g0={val_metrics['g0_mean']:.4f} "
                f"val_gt={val_metrics['gt_mean']:.4f} "
                f"val_sigma2={val_metrics['sigma2_mean']:.4f}"
            )
            if cfg.val_kl_enable and val_ref_angles is not None and val_ref_lengths is not None:
                eval_state = flax_jax_utils.unreplicate(state) if use_distributed else state
                val_kl = _compute_val_kl(
                    params=_params_for_eval(eval_state, cfg.eval_use_ema),
                    cfg=cfg,
                    reference_angles=val_ref_angles,
                    val_lengths=val_ref_lengths,
                    epoch=epoch,
                    sample_fn=val_sample_fn,
                )
                print(f"epoch={epoch:04d} val_kl={val_kl:.6f}")

        improved_val_loss = bool(val_metrics is not None and float(val_metrics["loss"]) < best_val_loss)
        if improved_val_loss and val_metrics is not None:
            best_val_loss = float(val_metrics["loss"])
        improved_val_kl = bool(val_kl is not None and val_kl < best_val_kl)
        if improved_val_kl and val_kl is not None:
            best_val_kl = float(val_kl)

        if cfg.best_metric == "val_kl":
            is_best = improved_val_kl
        else:
            is_best = improved_val_loss

        if wandb_run is not None:
            payload = {
                "loss": metrics["loss"],
                "g0_mean": metrics["g0_mean"],
                "gt_mean": metrics["gt_mean"],
                "sigma2_mean": metrics["sigma2_mean"],
            }
            if val_metrics is not None:
                payload.update(
                    {
                        "val_loss": val_metrics["loss"],
                        "val_g0_mean": val_metrics["g0_mean"],
                        "val_gt_mean": val_metrics["gt_mean"],
                        "val_sigma2_mean": val_metrics["sigma2_mean"],
                    }
                )
                payload["best_val_loss"] = best_val_loss
            if val_kl is not None:
                payload["val_kl"] = val_kl
                payload["best_val_kl"] = best_val_kl
            wandb_run.log(payload, step=epoch)

        if cfg.save_every_epochs > 0 and (epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs):
            ckpt_path = ckpt_root / f"model_epoch_{epoch:04d}.msgpack"
            save_state = flax_jax_utils.unreplicate(state) if use_distributed else state
            _save_checkpoint(ckpt_path, save_state, epoch, metrics, cfg)
            print(f"saved={ckpt_path}")
            if wandb_run is not None:
                wandb_run.log({"checkpoint_path": str(ckpt_path)}, step=epoch)
                log_checkpoint_artifact(
                    wandb_run,
                    ckpt_path=ckpt_path,
                    epoch=epoch,
                    is_best=is_best,
                    artifact_name=cfg.checkpoint_artifact_name,
                )
        elif is_best:
            ckpt_path = ckpt_root / f"model_epoch_{epoch:04d}.msgpack"
            save_state = flax_jax_utils.unreplicate(state) if use_distributed else state
            _save_checkpoint(ckpt_path, save_state, epoch, metrics, cfg)
            print(f"saved_best={ckpt_path}")
            if wandb_run is not None:
                wandb_run.log({"checkpoint_path": str(ckpt_path)}, step=epoch)
                log_checkpoint_artifact(
                    wandb_run,
                    ckpt_path=ckpt_path,
                    epoch=epoch,
                    is_best=True,
                    artifact_name=cfg.checkpoint_artifact_name,
                )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

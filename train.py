"""Train score-based diffusion in intrinsic torsion coordinates (JAX/Flax)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flax.jax_utils as flax_jax_utils
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

import wandb
from diffgeo.manifold import ExtrinsicMaskedTorus, IntrinsicMaskedTorus
from evaluation.metrics import collect_reference_angles, compute_val_kl, params_for_eval
from foldingdiff.bert_for_diffusion import BertDiffusionConfig, BertForDiffusion
from foldingdiff.dataset import CathCanonicalAnglesOnlyDataset
from RDM.beta_schedule import LinearBetaSchedule
from RDM.losses import get_bridge_loss_fn
from RDM.sde_lib import DiffusionMixture
from RDM.training import intrinsic_to_cossin, make_bridge_train_step, train_one_epoch_bridge
from score_based.sampling import sample_intrinsic_batch
from score_based.training import (
    ScoreTrainConfig,
    TrainState,
    create_train_state,
    eval_one_epoch_for_mode,
    make_eval_step_for_mode,
    make_train_step_for_mode,
    train_one_epoch_for_mode,
)
from utils.checkpoint import save_checkpoint
from utils.config import TrainConfig
from utils.data_iter import batch_iter, batch_iter_for_mode
from utils.lr_schedule import build_learning_rate_schedule
from utils.wandb import (
    download_wandb_checkpoint,
    get_best_scalar_from_wandb,
    get_best_val_loss_from_wandb,
    get_resume_epoch_from_wandb,
    load_config_from_resumed_run,
    log_checkpoint_artifact,
)


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


def train_bridge_objective(
    *,
    cfg: TrainConfig,
    train_ds: CathCanonicalAnglesOnlyDataset,
    model: BertForDiffusion,
    model_b: BertForDiffusion,
    state_f: TrainState,
    state_b: TrainState,
    sample_x: jnp.ndarray,
    start_epoch: int,
    np_rng: np.random.Generator,
    wandb_run: Any,
) -> None:
    if cfg.distributed and jax.local_device_count() > 1:
        raise ValueError(
            "training_objective=bridge_matching currently supports only distributed=false"
        )

    def _identity_preprocess(x, m):
        return x, m

    if cfg.bridge_coordinates == "extrinsic":
        manifold = ExtrinsicMaskedTorus(dim=sample_x.shape[-1] // 2)
        preprocess_fn = intrinsic_to_cossin
    else:
        manifold = IntrinsicMaskedTorus(dim=sample_x.shape[-1])
        preprocess_fn = _identity_preprocess
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
    ckpt_root = Path(cfg.weights_path)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    best_bridge_loss = float("inf")
    print(
        f"Training start: objective=bridge_matching n_train={len(train_ds)} batch={cfg.batch_size} "
        f"steps={cfg.bridge_num_steps} weight={cfg.bridge_weight_type} "
        f"coords={cfg.bridge_coordinates}"
    )
    for epoch in range(start_epoch, cfg.epochs + 1):
        batches = batch_iter(train_ds, batch_size=cfg.batch_size, rng=np_rng, shuffle=True)
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

        if cfg.save_every_epochs > 0 and (
            epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs or is_best
        ):
            ckpt_path = ckpt_root / f"model_epoch_{epoch:04d}.msgpack"
            save_state = {"state_f": state_f, "state_b": state_b}
            save_checkpoint(ckpt_path, save_state, epoch, metrics, cfg)
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


def train_score_objective(
    *,
    cfg: TrainConfig,
    model: BertForDiffusion,
    model_cfg: BertDiffusionConfig,
    state: TrainState,
    train_ds: CathCanonicalAnglesOnlyDataset,
    val_ds: CathCanonicalAnglesOnlyDataset,
    val_ref_angles: np.ndarray | None,
    val_ref_lengths: np.ndarray | None,
    start_epoch: int,
    np_rng: np.random.Generator,
    use_distributed: bool,
    n_devices: int,
    wandb_run: Any,
) -> None:
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
    train_step_fn = make_train_step_for_mode(loss_cfg, distributed=use_distributed)
    eval_step_fn = make_eval_step_for_mode(loss_cfg, distributed=use_distributed)
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
        get_best_scalar_from_wandb(
            cfg.wandb_entity, cfg.wandb_project, cfg.resume_run, "best_val_kl"
        )
        if cfg.resume_run
        else float("inf")
    )
    if cfg.best_metric == "val_kl" and not cfg.val_kl_enable:
        raise ValueError("best_metric=val_kl requires val_kl_enable=true")
    val_batch_size = cfg.val_batch_size if cfg.val_batch_size > 0 else cfg.batch_size
    for epoch in range(start_epoch, cfg.epochs + 1):
        batches = batch_iter_for_mode(
            dataset=train_ds,
            batch_size=cfg.batch_size,
            rng=np_rng,
            distributed=use_distributed,
            n_devices=n_devices,
            shuffle=True,
        )
        state, metrics = train_one_epoch_for_mode(
            state=state,
            train_batches=batches,
            cfg=loss_cfg,
            distributed=use_distributed,
            epoch=epoch,
            log_every=cfg.train_log_every,
            train_step_fn=train_step_fn,
        )
        print(f"epoch={epoch:04d} loss={metrics['loss']:.6f} sigma2={metrics['sigma2_mean']:.4f}")
        val_metrics = None
        val_kl = None
        should_eval = bool(
            cfg.train_val
            and len(val_ds) > 0
            and epoch >= cfg.start_eval_epoch
            and ((epoch - cfg.start_eval_epoch) % max(1, cfg.val_freq) == 0)
        )
        if should_eval:
            val_batches = batch_iter_for_mode(
                dataset=val_ds,
                batch_size=val_batch_size,
                rng=np_rng,
                distributed=use_distributed,
                n_devices=n_devices,
                shuffle=False,
            )
            state, val_metrics = eval_one_epoch_for_mode(
                state=state,
                eval_batches=val_batches,
                cfg=loss_cfg,
                distributed=use_distributed,
                epoch=epoch,
                log_every=cfg.val_log_every,
                max_batches=cfg.val_max_batches,
                eval_step_fn=eval_step_fn,
            )
            print(
                f"epoch={epoch:04d} val_loss={val_metrics['loss']:.6f} "
                f"val_sigma2={val_metrics['sigma2_mean']:.4f}"
            )
            if cfg.val_kl_enable and val_ref_angles is not None and val_ref_lengths is not None:
                eval_state = flax_jax_utils.unreplicate(state) if use_distributed else state
                val_kl = compute_val_kl(
                    params=params_for_eval(eval_state, cfg.eval_use_ema),
                    cfg=cfg,
                    reference_angles=val_ref_angles,
                    val_lengths=val_ref_lengths,
                    epoch=epoch,
                    sample_fn=val_sample_fn,
                )
                print(f"epoch={epoch:04d} val_kl={val_kl:.6f}")

        improved_val_loss = bool(
            val_metrics is not None and float(val_metrics["loss"]) < best_val_loss
        )
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

        if cfg.save_every_epochs > 0 and (
            epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs
        ):
            ckpt_path = ckpt_root / f"model_epoch_{epoch:04d}.msgpack"
            save_state = flax_jax_utils.unreplicate(state) if use_distributed else state
            save_checkpoint(ckpt_path, save_state, epoch, metrics, cfg)
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
            save_checkpoint(ckpt_path, save_state, epoch, metrics, cfg)
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
        val_ref_angles, val_ref_lengths = collect_reference_angles(
            val_ds, limit=cfg.val_kl_ref_limit
        )

    bridge_feat_dim = (
        12
        if (cfg.training_objective == "bridge_matching" and cfg.bridge_coordinates == "extrinsic")
        else 6
    )
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

    init_batch = next(
        batch_iter(
            train_ds, batch_size=min(cfg.batch_size, len(train_ds)), rng=np_rng, shuffle=False
        )
    )
    sample_x = init_batch["angles"][:, :-1, :].reshape(init_batch["angles"].shape[0], -1)
    sample_mask = init_batch["geo_mask"]
    if cfg.training_objective == "bridge_matching" and cfg.bridge_coordinates == "extrinsic":
        sample_x, sample_mask = intrinsic_to_cossin(sample_x, sample_mask)
    if use_distributed:
        steps_per_epoch = len(train_ds) // cfg.batch_size
    else:
        steps_per_epoch = (len(train_ds) + cfg.batch_size - 1) // cfg.batch_size
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    learning_rate = build_learning_rate_schedule(cfg, total_steps=total_steps)
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
                state_single = flax.serialization.from_state_dict(
                    state_single, raw_state["state_f"]
                )
                assert state_b_single is not None
                state_b_single = flax.serialization.from_state_dict(
                    state_b_single, raw_state["state_b"]
                )
            else:
                state_single = flax.serialization.from_bytes(state_single, ckpt_bytes)
        else:
            state_single = flax.serialization.from_bytes(state_single, ckpt_bytes)
        start_epoch = (
            get_resume_epoch_from_wandb(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                run_id=cfg.resume_run,
            )
            + 1
        )
        print(f"resumed_checkpoint={resume_ckpt}")
        print(f"resumed_start_epoch={start_epoch}")
    try:
        if cfg.training_objective == "bridge_matching":
            assert model_b is not None and state_b_single is not None
            train_bridge_objective(
                cfg=cfg,
                train_ds=train_ds,
                model=model,
                model_b=model_b,
                state_f=state_single,
                state_b=state_b_single,
                sample_x=sample_x,
                start_epoch=start_epoch,
                np_rng=np_rng,
                wandb_run=wandb_run,
            )
        elif cfg.training_objective == "score":
            state: TrainState = (
                flax_jax_utils.replicate(state_single) if use_distributed else state_single
            )
            train_score_objective(
                cfg=cfg,
                model=model,
                model_cfg=model_cfg,
                state=state,
                train_ds=train_ds,
                val_ds=val_ds,
                val_ref_angles=val_ref_angles,
                val_ref_lengths=val_ref_lengths,
                start_epoch=start_epoch,
                np_rng=np_rng,
                use_distributed=use_distributed,
                n_devices=n_devices,
                wandb_run=wandb_run,
            )
        else:
            raise ValueError(f"Unsupported training_objective={cfg.training_objective}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

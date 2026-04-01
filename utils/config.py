"""CLI/runtime configuration for training."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainConfig(BaseSettings, cli_parse_args=True):
    model_config = SettingsConfigDict(cli_kebab_case=True)

    training_objective: Literal["score", "bridge_matching"] = "score"

    epochs: int = 200
    batch_size: int = 32
    val_batch_size: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    lr_sched: bool = False
    lr_schedule_type: Literal["cosine", "linear"] = "cosine"
    lr_warmup_frac: float = 0.1
    min_lr_ratio: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.999
    eval_use_ema: bool = True
    seed: int = 42

    max_seq_len: int = 128
    min_seq_len: int = 40
    pdbs: str = "cath"
    toy: int = 0
    zero_center: bool = False
    dataset_workers: int = 1

    net_size: int = 3
    dropout: float = 0.1
    relative_position: bool = True
    metric_condition_model: bool = False

    metric_type: Literal["kinetic_diag", "flat_torus"] = "kinetic_diag"
    metric_cutoff: float = 10.0
    metric_eps: float = 1e-3
    metric_normalize: bool = True
    metric_clamp_min: float = 1e-3
    metric_clamp_max: float | None = None
    metric_anneal: bool = True
    metric_anneal_data_lambda: float = 0.0
    metric_anneal_prior_lambda: float = 1.0
    metric_anneal_power: float = 1.0
    beta_0: float = 10.0
    beta_f: float = 0.1
    n_wrap: int = 2
    t_eps: float = 1e-5
    grad_norm: float = 1.0

    weights_path: str = "./weights"
    save_every_epochs: int = 20
    train_log_every: int = 0
    train_val: bool = True
    val_freq: int = 1
    start_eval_epoch: int = 1
    val_log_every: int = 0
    val_max_batches: int = 0
    val_kl_enable: bool = True
    val_kl_samples: int = 64
    val_kl_batch_size: int = 32
    val_kl_n_steps: int = 200
    val_kl_eps: float = 1e-3
    val_kl_bins: int = 120
    val_kl_ref_limit: int = 0
    best_metric: Literal["val_loss", "val_kl"] = "val_kl"
    distributed: bool = True

    bridge_num_steps: int = 15
    bridge_weight_type: Literal["default", "importance"] = "importance"
    bridge_coordinates: Literal["intrinsic", "extrinsic"] = "extrinsic"
    bridge_beta_0: float = 0.2
    bridge_beta_f: float = 0.001
    bridge_eps: float = 1e-3

    wandb_entity: str = "rdem"
    wandb_project: str = "Standard Metric - RiemannDiff"
    wandb_run_name: str | None = None
    wandb_mode: Literal["online", "offline", "disabled", "shared"] | None = "disabled"
    resume_run: str | None = None
    resume_checkpoint_path: str | None = None
    checkpoint_artifact_name: str = "model_state"

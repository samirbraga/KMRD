"""CLI/runtime configuration for training."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainConfig(BaseSettings, cli_parse_args=True):
    model_config = SettingsConfigDict(cli_kebab_case=True)

    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    seed: int = 42

    max_seq_len: int = 128
    min_seq_len: int = 40
    pdbs: str = "cath"
    toy: int = 0
    zero_center: bool = False
    dataset_workers: int = 1

    net_size: int = 3
    dropout: float = 0.1
    metric_condition_model: bool = False

    metric_type: Literal["kinetic_diag", "flat_torus"] = "kinetic_diag"
    metric_cutoff: float = 10.0
    metric_eps: float = 1e-3
    beta_0: float = 10.0
    beta_f: float = 0.1
    n_wrap: int = 2
    t_eps: float = 1e-5
    grad_norm: float = 1.0

    weights_path: str = "./weights"
    save_every_epochs: int = 20
    train_log_every: int = 0
    distributed: bool = True

    wandb_entity: str = "rdem"
    wandb_project: str = "Standard Metric - RiemannDiff"
    wandb_run_name: str | None = None
    wandb_mode: Literal["online", "offline", "disabled", "shared"] | None = "disabled"
    resume_run: str | None = None

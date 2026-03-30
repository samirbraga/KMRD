"""Learning-rate schedule helpers."""

from __future__ import annotations

import optax


def build_learning_rate_schedule(
    cfg,
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


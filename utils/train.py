"""JAX training utilities for intrinsic score-based diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import jax
import jax.numpy as jnp
import optax
from jax import lax
from flax.training import train_state

from diffgeo.kinetic_metric import compute_kinetic_metric_diag
from utils.diffusion_math import (
    metric_anneal_lambda_from_sigma2,
    sigma2_linear,
    to_angles_lengths,
    wrap_to_pi,
)


def _wrapped_eps_target(
    dtheta_raw: jnp.ndarray,
    g_diag_0: jnp.ndarray,
    sigma2: jnp.ndarray,
    n_wrap: int,
) -> jnp.ndarray:
    sigma2_ = sigma2[:, None]
    sqrt_2sig = jnp.sqrt(2.0 * sigma2_)
    sqrt_g = jnp.sqrt(jnp.clip(g_diag_0, a_min=1e-8))

    k_vals = jnp.arange(-n_wrap, n_wrap + 1, dtype=dtheta_raw.dtype)
    delta = dtheta_raw[None, :, :] - (2.0 * jnp.pi * k_vals[:, None, None])
    eps_candidates = sqrt_g[None, :, :] * delta / sqrt_2sig[None, :, :]
    log_w = -0.5 * eps_candidates * eps_candidates
    w = jax.nn.softmax(log_w, axis=0)
    return jnp.sum(w * eps_candidates, axis=0)


@dataclass(frozen=True)
class ScoreTrainConfig:
    coordinate_system: Literal["intrinsic"] = "intrinsic"
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
    t_eps: float = 1e-5
    n_wrap: int = 2
    max_grad_norm: float = 1.0


class TrainState(train_state.TrainState):
    rng: jax.Array
    ema_params: Any | None = None
    ema_decay: float = 0.999


def create_train_state(
    model: Any,
    rng: jax.Array,
    sample_x: jnp.ndarray,
    sample_mask: jnp.ndarray,
    learning_rate: float | optax.Schedule = 1e-4,
    weight_decay: float = 1e-2,
    use_ema: bool = True,
    ema_decay: float = 0.999,
) -> TrainState:
    init_g = jnp.zeros_like(sample_x)
    variables = model.init(
        {"params": rng, "dropout": rng},
        inputs=sample_x,
        timestep=jnp.zeros((sample_x.shape[0],), dtype=sample_x.dtype),
        mask=sample_mask,
        manifold=None,
        g_diag=init_g,
        deterministic=True,
    )
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        rng=rng,
        ema_params=variables["params"] if use_ema else None,
        ema_decay=float(ema_decay),
    )


def _compute_metric_diag(
    theta: jnp.ndarray,
    mask: jnp.ndarray,
    cfg: ScoreTrainConfig,
) -> jnp.ndarray:
    if cfg.metric_type == "flat_torus":
        return jnp.ones_like(theta)
    angles, lengths = to_angles_lengths(theta, mask)
    return compute_kinetic_metric_diag(
        angles_batch=angles,
        lengths=lengths,
        geo_mask=mask,
        cutoff=cfg.metric_cutoff,
        eps=cfg.metric_eps,
        normalize=cfg.metric_normalize,
        clamp_min=cfg.metric_clamp_min,
        clamp_max=cfg.metric_clamp_max,
    )


def _score_loss(
    params: Any,
    apply_fn: Any,
    batch: dict[str, jnp.ndarray],
    rng: jax.Array,
    cfg: ScoreTrainConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    if cfg.coordinate_system != "intrinsic":
        raise ValueError("Score-based Riemannian diffusion training only supports intrinsic coordinates.")

    angles = jnp.asarray(batch["angles"], dtype=jnp.float32)
    mask = jnp.asarray(batch["geo_mask"], dtype=jnp.float32)
    x0 = angles[:, :-1, :].reshape(angles.shape[0], -1)

    bsz = x0.shape[0]
    rng_t, rng_eps, rng_dropout = jax.random.split(rng, 3)
    t = jax.random.uniform(rng_t, (bsz,), minval=cfg.t_eps, maxval=1.0 - cfg.t_eps)
    sigma2 = jnp.clip(sigma2_linear(t, cfg.beta_0, cfg.beta_f), a_min=1e-8)
    sigma2_max = jnp.clip(
        sigma2_linear(jnp.ones_like(t), cfg.beta_0, cfg.beta_f),
        a_min=1e-8,
    )
    anneal_lambda = metric_anneal_lambda_from_sigma2(
        sigma2=sigma2,
        sigma2_max=sigma2_max,
        enabled=cfg.metric_anneal,
        data_lambda=cfg.metric_anneal_data_lambda,
        prior_lambda=cfg.metric_anneal_prior_lambda,
        power=cfg.metric_anneal_power,
    )

    g_diag_0 = _compute_metric_diag(x0, mask, cfg)
    g_diag_0 = jnp.nan_to_num(g_diag_0, nan=1.0, posinf=1e6, neginf=1.0)
    g_diag_0 = (1.0 - anneal_lambda[:, None]) + anneal_lambda[:, None] * g_diag_0
    sigma_diag_0 = jnp.clip(1.0 / g_diag_0, a_min=1e-8)

    eps_noise = jax.random.normal(rng_eps, x0.shape, dtype=x0.dtype) * mask
    scale = jnp.sqrt(2.0 * sigma2)[:, None]
    theta_t = wrap_to_pi(x0 + scale * jnp.sqrt(sigma_diag_0) * eps_noise) * mask

    g_diag_t = _compute_metric_diag(theta_t, mask, cfg)
    g_diag_t = jnp.nan_to_num(g_diag_t, nan=1.0, posinf=1e6, neginf=1.0)
    g_diag_t = (1.0 - anneal_lambda[:, None]) + anneal_lambda[:, None] * g_diag_t
    sigma_diag_t = jnp.clip(1.0 / g_diag_t, a_min=1e-8)

    dtheta_raw = theta_t - x0
    eps_target = _wrapped_eps_target(dtheta_raw, g_diag_0, sigma2, cfg.n_wrap) * mask
    eps_target = jnp.nan_to_num(eps_target, nan=0.0, posinf=0.0, neginf=0.0)
    g_input = g_diag_t if cfg.metric_type == "kinetic_diag" else None

    eps_pred = apply_fn(
        {"params": params},
        inputs=theta_t,
        timestep=t,
        mask=mask,
        manifold=None,
        g_diag=g_input,
        deterministic=False,
        rngs={"dropout": rng_dropout},
    )
    residual = eps_pred - eps_target
    residual = jnp.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    losses = (residual * residual) * sigma_diag_t * mask
    denom = jnp.clip(jnp.sum(mask, axis=-1), a_min=1.0)
    loss_per_sample = jnp.sum(losses, axis=-1) / denom
    loss = jnp.mean(loss_per_sample)
    return loss, {
        "loss": loss,
        "sigma2_mean": jnp.mean(sigma2),
        "g0_mean": jnp.mean(g_diag_0),
        "gt_mean": jnp.mean(g_diag_t),
    }


def make_train_step(cfg: ScoreTrainConfig):
    def train_step(state: TrainState, batch: dict[str, jnp.ndarray]) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        rng, step_rng = jax.random.split(state.rng)

        def loss_fn(params):
            return _score_loss(params, state.apply_fn, batch, step_rng, cfg)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        if cfg.max_grad_norm > 0:
            grad_norm = optax.global_norm(grads)
            scale = jnp.minimum(1.0, cfg.max_grad_norm / (grad_norm + 1e-6))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        new_state = state.apply_gradients(grads=grads)
        if new_state.ema_params is not None:
            step_size = jnp.asarray(1.0, dtype=jnp.float32) - jnp.asarray(
                new_state.ema_decay, dtype=jnp.float32
            )
            new_ema = optax.incremental_update(
                new_state.params,
                new_state.ema_params,
                step_size=step_size,
            )
            new_state = new_state.replace(ema_params=new_ema)
        new_state = new_state.replace(rng=rng)
        metrics = dict(aux)
        metrics["loss"] = loss
        return new_state, metrics

    return jax.jit(train_step)


def make_train_step_pmap(cfg: ScoreTrainConfig):
    def train_step(state: TrainState, batch: dict[str, jnp.ndarray]) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        rng, step_rng = jax.random.split(state.rng)

        def loss_fn(params):
            return _score_loss(params, state.apply_fn, batch, step_rng, cfg)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = lax.pmean(grads, axis_name="data")
        loss = lax.pmean(loss, axis_name="data")
        aux = lax.pmean(aux, axis_name="data")

        if cfg.max_grad_norm > 0:
            grad_norm = optax.global_norm(grads)
            scale = jnp.minimum(1.0, cfg.max_grad_norm / (grad_norm + 1e-6))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

        new_state = state.apply_gradients(grads=grads)
        if new_state.ema_params is not None:
            step_size = jnp.asarray(1.0, dtype=jnp.float32) - jnp.asarray(
                new_state.ema_decay, dtype=jnp.float32
            )
            new_ema = optax.incremental_update(
                new_state.params,
                new_state.ema_params,
                step_size=step_size,
            )
            new_state = new_state.replace(ema_params=new_ema)
        new_state = new_state.replace(rng=rng)
        metrics = dict(aux)
        metrics["loss"] = loss
        return new_state, metrics

    return jax.pmap(train_step, axis_name="data")


def make_eval_step(cfg: ScoreTrainConfig):
    def eval_step(state: TrainState, batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        rng, step_rng = jax.random.split(state.rng)
        eval_params = state.ema_params if state.ema_params is not None else state.params
        loss, aux = _score_loss(eval_params, state.apply_fn, batch, step_rng, cfg)
        metrics = dict(aux)
        metrics["loss"] = loss
        metrics["rng_next"] = rng
        return metrics

    return jax.jit(eval_step)


def make_eval_step_pmap(cfg: ScoreTrainConfig):
    def eval_step(state: TrainState, batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        rng, step_rng = jax.random.split(state.rng)
        eval_params = state.ema_params if state.ema_params is not None else state.params
        loss, aux = _score_loss(eval_params, state.apply_fn, batch, step_rng, cfg)
        loss = lax.pmean(loss, axis_name="data")
        aux = lax.pmean(aux, axis_name="data")
        metrics = dict(aux)
        metrics["loss"] = loss
        metrics["rng_next"] = rng
        return metrics

    return jax.pmap(eval_step, axis_name="data")


def train_one_epoch(
    state: TrainState,
    train_batches: Iterable[dict[str, jnp.ndarray]],
    cfg: ScoreTrainConfig,
    epoch: int | None = None,
    log_every: int = 0,
    train_step_fn: Any | None = None,
) -> tuple[TrainState, dict[str, float]]:
    train_step = train_step_fn if train_step_fn is not None else make_train_step(cfg)

    loss_sum = 0.0
    g0_sum = 0.0
    gt_sum = 0.0
    sigma2_sum = 0.0
    n_steps = 0

    for batch in train_batches:
        state, metrics = train_step(state, batch)
        m_loss = float(metrics["loss"])
        m_g0 = float(metrics["g0_mean"])
        m_gt = float(metrics["gt_mean"])
        m_s2 = float(metrics["sigma2_mean"])
        loss_sum += m_loss
        g0_sum += m_g0
        gt_sum += m_gt
        sigma2_sum += m_s2
        n_steps += 1
        if log_every > 0 and (n_steps % log_every == 0):
            prefix = f"epoch={epoch:04d} " if epoch is not None else ""
            print(
                f"{prefix}step={n_steps:04d} "
                f"loss={m_loss:.6f} "
                f"g0={m_g0:.4f} "
                f"gt={m_gt:.4f} "
                f"sigma2={m_s2:.4f}"
            )

    denom = max(n_steps, 1)
    return state, {
        "loss": loss_sum / denom,
        "g0_mean": g0_sum / denom,
        "gt_mean": gt_sum / denom,
        "sigma2_mean": sigma2_sum / denom,
    }


def train_one_epoch_pmap(
    state: TrainState,
    train_batches: Iterable[dict[str, jnp.ndarray]],
    cfg: ScoreTrainConfig,
    epoch: int | None = None,
    log_every: int = 0,
    train_step_fn: Any | None = None,
) -> tuple[TrainState, dict[str, float]]:
    train_step = train_step_fn if train_step_fn is not None else make_train_step_pmap(cfg)

    loss_sum = 0.0
    g0_sum = 0.0
    gt_sum = 0.0
    sigma2_sum = 0.0
    n_steps = 0

    for batch in train_batches:
        state, metrics = train_step(state, batch)
        # pmean makes all replicas equal; read from replica 0.
        m_loss = float(jax.device_get(metrics["loss"])[0])
        m_g0 = float(jax.device_get(metrics["g0_mean"])[0])
        m_gt = float(jax.device_get(metrics["gt_mean"])[0])
        m_s2 = float(jax.device_get(metrics["sigma2_mean"])[0])

        loss_sum += m_loss
        g0_sum += m_g0
        gt_sum += m_gt
        sigma2_sum += m_s2
        n_steps += 1
        if log_every > 0 and (n_steps % log_every == 0):
            prefix = f"epoch={epoch:04d} " if epoch is not None else ""
            print(
                f"{prefix}step={n_steps:04d} "
                f"loss={m_loss:.6f} "
                f"g0={m_g0:.4f} "
                f"gt={m_gt:.4f} "
                f"sigma2={m_s2:.4f}"
            )

    denom = max(n_steps, 1)
    return state, {
        "loss": loss_sum / denom,
        "g0_mean": g0_sum / denom,
        "gt_mean": gt_sum / denom,
        "sigma2_mean": sigma2_sum / denom,
    }


def eval_one_epoch(
    state: TrainState,
    eval_batches: Iterable[dict[str, jnp.ndarray]],
    cfg: ScoreTrainConfig,
    epoch: int | None = None,
    log_every: int = 0,
    max_batches: int = 0,
    eval_step_fn: Any | None = None,
) -> tuple[TrainState, dict[str, float]]:
    eval_step = eval_step_fn if eval_step_fn is not None else make_eval_step(cfg)

    loss_sum = 0.0
    g0_sum = 0.0
    gt_sum = 0.0
    sigma2_sum = 0.0
    n_steps = 0
    cur_state = state

    for batch in eval_batches:
        metrics = eval_step(cur_state, batch)
        cur_state = cur_state.replace(rng=metrics["rng_next"])
        m_loss = float(metrics["loss"])
        m_g0 = float(metrics["g0_mean"])
        m_gt = float(metrics["gt_mean"])
        m_s2 = float(metrics["sigma2_mean"])
        loss_sum += m_loss
        g0_sum += m_g0
        gt_sum += m_gt
        sigma2_sum += m_s2
        n_steps += 1
        if log_every > 0 and (n_steps % log_every == 0):
            prefix = f"epoch={epoch:04d} " if epoch is not None else ""
            print(
                f"{prefix}val_step={n_steps:04d} "
                f"val_loss={m_loss:.6f} "
                f"g0={m_g0:.4f} "
                f"gt={m_gt:.4f} "
                f"sigma2={m_s2:.4f}"
            )
        if max_batches > 0 and n_steps >= max_batches:
            break

    denom = max(n_steps, 1)
    return cur_state, {
        "loss": loss_sum / denom,
        "g0_mean": g0_sum / denom,
        "gt_mean": gt_sum / denom,
        "sigma2_mean": sigma2_sum / denom,
    }


def eval_one_epoch_pmap(
    state: TrainState,
    eval_batches: Iterable[dict[str, jnp.ndarray]],
    cfg: ScoreTrainConfig,
    epoch: int | None = None,
    log_every: int = 0,
    max_batches: int = 0,
    eval_step_fn: Any | None = None,
) -> tuple[TrainState, dict[str, float]]:
    eval_step = eval_step_fn if eval_step_fn is not None else make_eval_step_pmap(cfg)

    loss_sum = 0.0
    g0_sum = 0.0
    gt_sum = 0.0
    sigma2_sum = 0.0
    n_steps = 0
    cur_state = state

    for batch in eval_batches:
        metrics = eval_step(cur_state, batch)
        cur_state = cur_state.replace(rng=metrics["rng_next"])
        m_loss = float(jax.device_get(metrics["loss"])[0])
        m_g0 = float(jax.device_get(metrics["g0_mean"])[0])
        m_gt = float(jax.device_get(metrics["gt_mean"])[0])
        m_s2 = float(jax.device_get(metrics["sigma2_mean"])[0])
        loss_sum += m_loss
        g0_sum += m_g0
        gt_sum += m_gt
        sigma2_sum += m_s2
        n_steps += 1
        if log_every > 0 and (n_steps % log_every == 0):
            prefix = f"epoch={epoch:04d} " if epoch is not None else ""
            print(
                f"{prefix}val_step={n_steps:04d} "
                f"val_loss={m_loss:.6f} "
                f"g0={m_g0:.4f} "
                f"gt={m_gt:.4f} "
                f"sigma2={m_s2:.4f}"
            )
        if max_batches > 0 and n_steps >= max_batches:
            break

    denom = max(n_steps, 1)
    return cur_state, {
        "loss": loss_sum / denom,
        "g0_mean": g0_sum / denom,
        "gt_mean": gt_sum / denom,
        "sigma2_mean": sigma2_sum / denom,
    }

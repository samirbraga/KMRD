"""Training helpers for RDM bridge matching."""

from __future__ import annotations

from typing import Iterable

import jax
import jax.numpy as jnp
import optax

from score_based.training import TrainState


def batch_to_x_mask(batch: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    angles = jnp.asarray(batch["angles"], dtype=jnp.float32)
    mask = jnp.asarray(batch["geo_mask"], dtype=jnp.float32)
    x = angles[:, :-1, :].reshape(angles.shape[0], -1)
    return x, mask


def intrinsic_to_cossin(x: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_pair = jnp.stack([jnp.cos(x), jnp.sin(x)], axis=-1)
    x_ext = x_pair.reshape(x.shape[0], -1)
    # Keep mask in intrinsic layout (one entry per torsion angle), matching old Torch behavior.
    return x_ext, mask


def make_bridge_train_step(
    *,
    loss_fn,
    grad_norm: float,
    preprocess_fn,
):
    def train_step(
        state_f: TrainState,
        state_b: TrainState,
        batch: dict[str, jnp.ndarray],
    ) -> tuple[TrainState, TrainState, dict[str, jnp.ndarray]]:
        rng, step_rng = jax.random.split(state_f.rng)
        x, mask = batch_to_x_mask(batch)
        x, mask = preprocess_fn(x, mask)

        def wrapped_loss(params_f, params_b):
            loss, aux = loss_fn(
                step_rng,
                params_f=params_f,
                params_b=params_b,
                x=x,
                mask=mask,
                deterministic=False,
            )
            return loss, aux

        (loss, aux), grads = jax.value_and_grad(wrapped_loss, argnums=(0, 1), has_aux=True)(
            state_f.params,
            state_b.params,
        )
        grads_f, grads_b = grads

        if grad_norm > 0:
            gn_f = optax.global_norm(grads_f)
            gn_b = optax.global_norm(grads_b)
            sf = jnp.minimum(1.0, grad_norm / (gn_f + 1e-6))
            sb = jnp.minimum(1.0, grad_norm / (gn_b + 1e-6))
            grads_f = jax.tree_util.tree_map(lambda g: g * sf, grads_f)
            grads_b = jax.tree_util.tree_map(lambda g: g * sb, grads_b)

        new_f = state_f.apply_gradients(grads=grads_f)
        new_b = state_b.apply_gradients(grads=grads_b)

        if new_f.ema_params is not None:
            step_size_f = jnp.asarray(1.0, dtype=jnp.float32) - jnp.asarray(
                new_f.ema_decay, dtype=jnp.float32
            )
            ema_f = optax.incremental_update(new_f.params, new_f.ema_params, step_size=step_size_f)
            new_f = new_f.replace(ema_params=ema_f)
        if new_b.ema_params is not None:
            step_size_b = jnp.asarray(1.0, dtype=jnp.float32) - jnp.asarray(
                new_b.ema_decay, dtype=jnp.float32
            )
            ema_b = optax.incremental_update(new_b.params, new_b.ema_params, step_size=step_size_b)
            new_b = new_b.replace(ema_params=ema_b)

        new_f = new_f.replace(rng=rng)
        new_b = new_b.replace(rng=rng)
        metrics = {
            "loss": loss,
            "loss_f": aux["loss_f"],
            "loss_b": aux["loss_b"],
        }
        return new_f, new_b, metrics

    return jax.jit(train_step)


def train_one_epoch_bridge(
    *,
    state_f: TrainState,
    state_b: TrainState,
    train_batches: Iterable[dict[str, jnp.ndarray]],
    train_step_fn,
    epoch: int,
    log_every: int,
) -> tuple[TrainState, TrainState, dict[str, float]]:
    loss_sum = 0.0
    loss_f_sum = 0.0
    loss_b_sum = 0.0
    n_steps = 0
    for batch in train_batches:
        state_f, state_b, metrics = train_step_fn(state_f, state_b, batch)
        m_loss = float(metrics["loss"])
        m_f = float(metrics["loss_f"])
        m_b = float(metrics["loss_b"])
        loss_sum += m_loss
        loss_f_sum += m_f
        loss_b_sum += m_b
        n_steps += 1
        if log_every > 0 and (n_steps % log_every == 0):
            print(
                f"epoch={epoch:04d} step={n_steps:04d} loss={m_loss:.6f} "
                f"loss_f={m_f:.6f} loss_b={m_b:.6f}"
            )
    denom = max(1, n_steps)
    return (
        state_f,
        state_b,
        {
            "loss": loss_sum / denom,
            "loss_f": loss_f_sum / denom,
            "loss_b": loss_b_sum / denom,
        },
    )

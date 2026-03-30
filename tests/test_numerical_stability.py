from __future__ import annotations

import jax
import jax.numpy as jnp

from RDM.beta_schedule import LinearBetaSchedule
from RDM.sde_lib import DiffusionMixture
from diffgeo.manifold import KineticIntrinsicTorus, IntrinsicMaskedTorus
from score_based.losses import get_flat_score_loss_fn, get_kinetic_score_loss_fn
from score_based.sampling import sample_intrinsic_batch


def _dummy_model_apply(
    variables,
    *,
    inputs,
    timestep,
    mask,
    manifold,
    g_diag,
    deterministic,
    rngs=None,
):
    del variables, timestep, mask, manifold, g_diag, deterministic, rngs
    return jnp.zeros_like(inputs)


class _DummyModel:
    def apply(self, variables, **kwargs):
        return _dummy_model_apply(variables, **kwargs)


def test_extreme_flat_loss_remains_finite() -> None:
    manifold = IntrinsicMaskedTorus(dim=18)
    beta = LinearBetaSchedule(tf=1.0, t0=0.0, beta_0=1e-4, beta_f=80.0)
    mix = DiffusionMixture(manifold=manifold, beta_schedule=beta, prior_type="unif")
    loss_fn = get_flat_score_loss_fn(mix=mix, model_apply=_dummy_model_apply, eps=1e-5, n_wrap=6)

    x = jax.random.uniform(jax.random.PRNGKey(10), (2, 18), minval=-jnp.pi, maxval=jnp.pi)
    mask = jnp.concatenate([jnp.ones((2, 12)), jnp.zeros((2, 6))], axis=-1).astype(jnp.float32)
    loss, aux = loss_fn(jax.random.PRNGKey(11), params=jnp.array(0.0), x=x, mask=mask, deterministic=True)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux["sigma2_mean"])


def test_extreme_kinetic_loss_remains_finite() -> None:
    manifold = KineticIntrinsicTorus(dim=18, metric_clamp_min=1e-6, metric_clamp_max=1e6)
    beta = LinearBetaSchedule(tf=1.0, t0=0.0, beta_0=5e-4, beta_f=100.0)
    mix = DiffusionMixture(manifold=manifold, beta_schedule=beta, prior_type="unif")
    loss_fn = get_kinetic_score_loss_fn(mix=mix, model_apply=_dummy_model_apply, eps=1e-5, n_wrap=6)

    x = jax.random.uniform(jax.random.PRNGKey(12), (2, 18), minval=-jnp.pi, maxval=jnp.pi)
    mask = jnp.concatenate([jnp.ones((2, 12)), jnp.zeros((2, 6))], axis=-1).astype(jnp.float32)
    loss, aux = loss_fn(jax.random.PRNGKey(13), params=jnp.array(0.0), x=x, mask=mask, deterministic=True)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux["sigma2_mean"])
    assert jnp.isfinite(aux["g0_mean"])
    assert jnp.isfinite(aux["gt_mean"])


def test_sampling_extreme_schedule_stays_finite_and_masked() -> None:
    mask = jnp.concatenate([jnp.ones((2, 12)), jnp.zeros((2, 6))], axis=-1).astype(jnp.float32)
    out = sample_intrinsic_batch(
        params=jnp.array(0.0),
        model=_DummyModel(),
        mask=mask,
        rng=jax.random.PRNGKey(14),
        n_steps=16,
        eps=1e-4,
        beta_0=1e-4,
        beta_f=80.0,
        metric_type="flat_torus",
        pc_corrector_steps=2,
        pc_corrector_step_scale=0.5,
        pc_corrector_noise_scale=1.5,
    )
    assert jnp.all(jnp.isfinite(out))
    assert jnp.allclose(out[:, 12:], 0.0, atol=1e-5)


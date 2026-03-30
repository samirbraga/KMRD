from __future__ import annotations

import jax
import jax.numpy as jnp

from RDM.beta_schedule import LinearBetaSchedule
from RDM.losses import get_bridge_loss_fn
from RDM.sde_lib import DiffusionMixture
from diffgeo.manifold import ExtrinsicMaskedTorus, KineticIntrinsicTorus, IntrinsicMaskedTorus
from score_based.losses import get_flat_score_loss_fn, get_kinetic_score_loss_fn


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


def test_flat_score_loss_finite_and_deterministic() -> None:
    manifold = IntrinsicMaskedTorus(dim=18)
    beta = LinearBetaSchedule(tf=1.0, t0=0.0, beta_0=10.0, beta_f=0.1)
    mix = DiffusionMixture(manifold=manifold, beta_schedule=beta, prior_type="unif")
    loss_fn = get_flat_score_loss_fn(mix=mix, model_apply=_dummy_model_apply, eps=1e-4, n_wrap=1)

    x = jax.random.uniform(jax.random.PRNGKey(10), (4, 18), minval=-jnp.pi, maxval=jnp.pi)
    mask = jnp.ones_like(x)
    rng = jax.random.PRNGKey(11)
    loss1, aux1 = loss_fn(rng, params=jnp.array(0.0), x=x, mask=mask, deterministic=True)
    loss2, aux2 = loss_fn(rng, params=jnp.array(0.0), x=x, mask=mask, deterministic=True)

    assert jnp.isfinite(loss1)
    assert jnp.isfinite(aux1["sigma2_mean"])
    assert jnp.allclose(loss1, loss2)
    assert jnp.allclose(aux1["sigma2_mean"], aux2["sigma2_mean"])


def test_kinetic_score_loss_finite() -> None:
    manifold = KineticIntrinsicTorus(dim=18)
    beta = LinearBetaSchedule(tf=1.0, t0=0.0, beta_0=10.0, beta_f=0.1)
    mix = DiffusionMixture(manifold=manifold, beta_schedule=beta, prior_type="unif")
    loss_fn = get_kinetic_score_loss_fn(mix=mix, model_apply=_dummy_model_apply, eps=1e-4, n_wrap=1)

    x = jax.random.uniform(jax.random.PRNGKey(12), (2, 18), minval=-jnp.pi, maxval=jnp.pi)
    mask = jnp.ones_like(x)
    loss, aux = loss_fn(jax.random.PRNGKey(13), params=jnp.array(0.0), x=x, mask=mask, deterministic=True)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux["sigma2_mean"])
    assert jnp.isfinite(aux["g0_mean"])
    assert jnp.isfinite(aux["gt_mean"])


def test_bridge_loss_finite() -> None:
    manifold = ExtrinsicMaskedTorus(dim=18)
    beta = LinearBetaSchedule(tf=1.0, t0=0.0, beta_0=10.0, beta_f=0.1)
    mix = DiffusionMixture(manifold=manifold, beta_schedule=beta, prior_type="unif")
    loss_fn = get_bridge_loss_fn(
        mix=mix,
        model_apply_f=_dummy_model_apply,
        model_apply_b=_dummy_model_apply,
        eps=1e-4,
        num_steps=3,
    )

    x = manifold.random_uniform(jax.random.PRNGKey(20), (2, 36))
    mask = jnp.ones_like(x)
    loss, aux = loss_fn(
        jax.random.PRNGKey(21),
        params_f=jnp.array(0.0),
        params_b=jnp.array(0.0),
        x=x,
        mask=mask,
        deterministic=True,
    )
    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux["loss_f"])
    assert jnp.isfinite(aux["loss_b"])


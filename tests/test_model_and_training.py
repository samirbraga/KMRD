from __future__ import annotations

import jax
import jax.numpy as jnp

from RDM.beta_schedule import LinearBetaSchedule
from RDM.losses import get_bridge_loss_fn
from RDM.sde_lib import DiffusionMixture
from RDM.training import intrinsic_to_cossin, make_bridge_train_step
from diffgeo.manifold import ExtrinsicMaskedTorus
from foldingdiff.bert_for_diffusion import BertDiffusionConfig, BertForDiffusion
from score_based.training import ScoreTrainConfig, create_train_state, make_train_step


def _make_score_model() -> BertForDiffusion:
    cfg = BertDiffusionConfig(
        num_attention_heads=4,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        input_feat_dim=6,
        torsion_feat_dim=6,
        condition_on_g_diag=True,
    )
    return BertForDiffusion(config=cfg)


def _make_bridge_model() -> BertForDiffusion:
    cfg = BertDiffusionConfig(
        num_attention_heads=4,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        input_feat_dim=12,
        torsion_feat_dim=12,
        condition_on_g_diag=False,
    )
    return BertForDiffusion(config=cfg)


def test_model_instantiation_and_forward_intrinsic() -> None:
    model = _make_score_model()
    b, d = 2, 18
    x = jax.random.normal(jax.random.PRNGKey(0), (b, d))
    m = jnp.ones((b, d), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)},
        inputs=x,
        timestep=jnp.ones((b,), dtype=jnp.float32),
        mask=m,
        manifold=None,
        g_diag=jnp.ones_like(x),
        deterministic=True,
    )
    y = model.apply(
        variables,
        inputs=x,
        timestep=jnp.ones((b,), dtype=jnp.float32),
        mask=m,
        manifold=None,
        g_diag=jnp.ones_like(x),
        deterministic=True,
    )
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_model_instantiation_and_forward_extrinsic_bridge() -> None:
    model = _make_bridge_model()
    b, d = 2, 36
    x = jax.random.normal(jax.random.PRNGKey(3), (b, d))
    m = jnp.ones((b, d), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.PRNGKey(4), "dropout": jax.random.PRNGKey(5)},
        inputs=x,
        timestep=jnp.ones((b,), dtype=jnp.float32),
        mask=m,
        manifold=None,
        g_diag=None,
        deterministic=True,
    )
    y = model.apply(
        variables,
        inputs=x,
        timestep=jnp.ones((b,), dtype=jnp.float32),
        mask=m,
        manifold=None,
        g_diag=None,
        deterministic=True,
    )
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_score_training_step_smoke() -> None:
    model = _make_score_model()
    b, pad = 2, 4
    d = (pad - 1) * 6
    sample_x = jax.random.normal(jax.random.PRNGKey(6), (b, d))
    sample_mask = jnp.ones((b, d), dtype=jnp.float32)
    state = create_train_state(
        model=model,
        rng=jax.random.PRNGKey(7),
        sample_x=sample_x,
        sample_mask=sample_mask,
        learning_rate=1e-4,
        use_ema=True,
    )
    batch = {
        "angles": jax.random.normal(jax.random.PRNGKey(8), (b, pad, 6)),
        "geo_mask": sample_mask,
    }
    cfg = ScoreTrainConfig(metric_type="flat_torus", max_grad_norm=1.0, eval_use_ema=True)
    train_step = make_train_step(cfg)
    new_state, metrics = train_step(state, batch)
    assert int(new_state.step) == int(state.step) + 1
    assert jnp.isfinite(metrics["loss"])


def test_bridge_training_step_smoke_extrinsic() -> None:
    model_f = _make_bridge_model()
    model_b = _make_bridge_model()

    b, pad = 2, 4
    d_intr = (pad - 1) * 6
    x_intr = jax.random.uniform(jax.random.PRNGKey(9), (b, d_intr), minval=-jnp.pi, maxval=jnp.pi)
    m_intr = jnp.ones_like(x_intr)
    sample_x, sample_mask = intrinsic_to_cossin(x_intr, m_intr)

    state_f = create_train_state(
        model=model_f,
        rng=jax.random.PRNGKey(10),
        sample_x=sample_x,
        sample_mask=sample_mask,
        learning_rate=1e-4,
        use_ema=True,
    )
    state_b = create_train_state(
        model=model_b,
        rng=jax.random.PRNGKey(11),
        sample_x=sample_x,
        sample_mask=sample_mask,
        learning_rate=1e-4,
        use_ema=True,
    )

    manifold = ExtrinsicMaskedTorus(dim=d_intr)
    beta = LinearBetaSchedule(tf=1.0, t0=0.0, beta_0=10.0, beta_f=0.1)
    mix = DiffusionMixture(manifold=manifold, beta_schedule=beta, prior_type="unif")
    loss_fn = get_bridge_loss_fn(
        mix=mix,
        model_apply_f=model_f.apply,
        model_apply_b=model_b.apply,
        eps=1e-4,
        num_steps=3,
    )
    step_fn = make_bridge_train_step(
        loss_fn=loss_fn,
        grad_norm=1.0,
        preprocess_fn=intrinsic_to_cossin,
    )

    batch = {
        "angles": jax.random.normal(jax.random.PRNGKey(12), (b, pad, 6)),
        "geo_mask": m_intr,
    }
    new_f, new_b, metrics = step_fn(state_f, state_b, batch)
    assert int(new_f.step) == int(state_f.step) + 1
    assert int(new_b.step) == int(state_b.step) + 1
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(metrics["loss_f"])
    assert jnp.isfinite(metrics["loss_b"])


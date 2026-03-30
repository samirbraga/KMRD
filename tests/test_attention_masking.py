from __future__ import annotations

import jax
import jax.numpy as jnp

from foldingdiff.bert_for_diffusion import BertDiffusionConfig, BertForDiffusion


def _build_model(condition_on_g: bool) -> BertForDiffusion:
    cfg = BertDiffusionConfig(
        num_attention_heads=4,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        input_feat_dim=6,
        torsion_feat_dim=6,
        condition_on_g_diag=condition_on_g,
    )
    return BertForDiffusion(config=cfg)


def test_padded_input_values_do_not_change_valid_outputs() -> None:
    model = _build_model(condition_on_g=False)
    bsz, d = 2, 18
    valid_d = 12
    mask = jnp.concatenate([jnp.ones((bsz, valid_d)), jnp.zeros((bsz, d - valid_d))], axis=-1)

    x = jax.random.normal(jax.random.PRNGKey(0), (bsz, d))
    # Only perturb the final coordinate, which maps to the last (masked) token
    # in this flatten/unflatten layout.
    x_alt = x.at[:, -1].set(123.0)

    variables = model.init(
        {"params": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)},
        inputs=x,
        timestep=jnp.full((bsz,), 0.4, dtype=jnp.float32),
        mask=mask,
        manifold=None,
        g_diag=None,
        deterministic=True,
    )
    y = model.apply(
        variables,
        inputs=x,
        timestep=jnp.full((bsz,), 0.4, dtype=jnp.float32),
        mask=mask,
        manifold=None,
        g_diag=None,
        deterministic=True,
    )
    y_alt = model.apply(
        variables,
        inputs=x_alt,
        timestep=jnp.full((bsz,), 0.4, dtype=jnp.float32),
        mask=mask,
        manifold=None,
        g_diag=None,
        deterministic=True,
    )
    assert jnp.allclose(y[:, : (d - 1)], y_alt[:, : (d - 1)], atol=1e-5)


def test_padded_gdiag_values_do_not_change_valid_outputs_when_conditioned() -> None:
    model = _build_model(condition_on_g=True)
    bsz, d = 2, 18
    valid_d = 12
    mask = jnp.concatenate([jnp.ones((bsz, valid_d)), jnp.zeros((bsz, d - valid_d))], axis=-1)
    x = jax.random.normal(jax.random.PRNGKey(3), (bsz, d))
    g = jnp.ones((bsz, d), dtype=jnp.float32)
    g_alt = g.at[:, valid_d:].set(1e6)

    variables = model.init(
        {"params": jax.random.PRNGKey(4), "dropout": jax.random.PRNGKey(5)},
        inputs=x,
        timestep=jnp.full((bsz,), 0.2, dtype=jnp.float32),
        mask=mask,
        manifold=None,
        g_diag=g,
        deterministic=True,
    )
    y = model.apply(
        variables,
        inputs=x,
        timestep=jnp.full((bsz,), 0.2, dtype=jnp.float32),
        mask=mask,
        manifold=None,
        g_diag=g,
        deterministic=True,
    )
    y_alt = model.apply(
        variables,
        inputs=x,
        timestep=jnp.full((bsz,), 0.2, dtype=jnp.float32),
        mask=mask,
        manifold=None,
        g_diag=g_alt,
        deterministic=True,
    )
    assert jnp.allclose(y[:, :valid_d], y_alt[:, :valid_d], atol=1e-5)

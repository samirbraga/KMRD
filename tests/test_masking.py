from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from diffgeo.manifold import ExtrinsicMaskedTorus, IntrinsicMaskedTorus
from foldingdiff.dataset import CathCanonicalAnglesOnlyDataset
from RDM.beta_schedule import LinearBetaSchedule
from RDM.sde_lib import DiffusionMixture
from RDM.training import batch_to_x_mask, intrinsic_to_cossin
from score_based.losses import get_flat_score_loss_fn
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


def test_batch_to_x_mask_and_cossin_preserve_masked_structure() -> None:
    # protein lengths: 3 and 4 residues => active geo dims: 12 and 18
    b, pad = 2, 4
    d = (pad - 1) * 6
    angles = jnp.zeros((b, pad, 6), dtype=jnp.float32)
    geo_mask = jnp.zeros((b, d), dtype=jnp.float32)
    geo_mask = geo_mask.at[0, :12].set(1.0)
    geo_mask = geo_mask.at[1, :18].set(1.0)
    batch = {"angles": angles, "geo_mask": geo_mask}

    x, m = batch_to_x_mask(batch)
    x_ext, m_ext = intrinsic_to_cossin(x, m)

    assert x.shape == (2, 18)
    assert m.shape == (2, 18)
    assert x_ext.shape == (2, 36)
    assert m_ext.shape == (2, 18)
    assert int(jnp.sum(m[0])) == 12
    assert int(jnp.sum(m[1])) == 18
    assert int(jnp.sum(m_ext[0])) == 12
    assert int(jnp.sum(m_ext[1])) == 18


def test_intrinsic_and_extrinsic_manifold_projection_respects_mask() -> None:
    m_intr = IntrinsicMaskedTorus(dim=18)
    x_intr = jnp.ones((1, 18), dtype=jnp.float32) * 7.0
    mask_intr = jnp.concatenate([jnp.ones((1, 12)), jnp.zeros((1, 6))], axis=-1)
    y_intr = m_intr.projection(x_intr, mask=mask_intr)
    assert jnp.allclose(y_intr[:, 12:], 0.0)

    m_ext = ExtrinsicMaskedTorus(dim=18)
    x_ext = jnp.ones((1, 36), dtype=jnp.float32)
    mask_ext = jnp.concatenate([jnp.ones((1, 24)), jnp.zeros((1, 12))], axis=-1)
    y_ext = m_ext.projection(x_ext, mask=mask_ext)
    assert jnp.allclose(y_ext[:, 24:], 0.0)


def test_flat_score_loss_invariant_to_masked_dimensions() -> None:
    manifold = IntrinsicMaskedTorus(dim=18)
    beta = LinearBetaSchedule(tf=1.0, t0=0.0, beta_0=10.0, beta_f=0.1)
    mix = DiffusionMixture(manifold=manifold, beta_schedule=beta, prior_type="unif")
    loss_fn = get_flat_score_loss_fn(mix=mix, model_apply=_dummy_model_apply, eps=1e-4, n_wrap=1)

    mask = jnp.concatenate([jnp.ones((2, 12)), jnp.zeros((2, 6))], axis=-1).astype(jnp.float32)
    x_base = jax.random.uniform(jax.random.PRNGKey(0), (2, 18), minval=-jnp.pi, maxval=jnp.pi)
    x_alt = x_base.at[:, 12:].set(123.456)  # only masked region changed
    rng = jax.random.PRNGKey(1)

    loss_base, _ = loss_fn(rng, params=jnp.array(0.0), x=x_base, mask=mask, deterministic=True)
    loss_alt, _ = loss_fn(rng, params=jnp.array(0.0), x=x_alt, mask=mask, deterministic=True)
    assert jnp.allclose(loss_base, loss_alt)


def test_sampling_keeps_masked_dimensions_zero() -> None:
    mask = jnp.concatenate([jnp.ones((2, 12)), jnp.zeros((2, 6))], axis=-1).astype(jnp.float32)
    x = sample_intrinsic_batch(
        params=jnp.array(0.0),
        model=_DummyModel(),
        mask=mask,
        rng=jax.random.PRNGKey(2),
        n_steps=8,
        eps=1e-3,
        beta_0=10.0,
        beta_f=0.1,
        metric_type="flat_torus",
    )
    assert x.shape == mask.shape
    assert jnp.allclose(x[:, 12:], 0.0, atol=1e-6)


def test_dataset_emits_variable_mask_lengths() -> None:
    try:
        dset = CathCanonicalAnglesOnlyDataset(
            pdbs="cath",
            split="train",
            pad=64,
            min_length=2,
            toy=32,
            zero_center=False,
            use_cache=False,
            num_workers=1,
        )
    except FileNotFoundError as exc:
        pytest.skip(f"Local dataset not available: {exc}")

    if len(dset) < 2:
        pytest.skip("Insufficient local samples to check variable lengths.")

    mask_sums = []
    for i in range(min(16, len(dset))):
        item = dset[i]
        mask_sums.append(int(jnp.sum(jnp.asarray(item["geo_mask"]))))

    # At least two distinct mask sizes for different protein lengths.
    assert len(set(mask_sums)) >= 2

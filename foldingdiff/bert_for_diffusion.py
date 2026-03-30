"""JAX/Flax reimplementation of FoldingDiff BERT diffusion model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import flax.linen as nn
import jax.numpy as jnp


@dataclass(frozen=True)
class BertDiffusionConfig:
    """Minimal configuration for the Flax diffusion BERT."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    input_feat_dim: int = 12
    torsion_feat_dim: int = 6
    condition_on_g_diag: bool = False
    is_decoder: bool = False


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for continuous timestep encoding."""

    embed_dim: int
    scale: float = float(2.0 * jnp.pi)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1:
            x = jnp.squeeze(x, axis=-1)
        elif x.ndim < 1:
            x = jnp.expand_dims(x, axis=0)

        # Keep the same semantics as the PyTorch implementation:
        # W ~ N(0, scale^2), then multiply by 2*pi again during projection.
        w = self.param(
            "W",
            nn.initializers.normal(stddev=self.scale),
            (self.embed_dim // 2,),
        )
        x_proj = x[:, None] * w[None, :] * (2.0 * jnp.pi)
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class BertLayer(nn.Module):
    """Single BERT encoder block in Flax."""

    config: BertDiffusionConfig

    @nn.compact
    def __call__(
        self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, deterministic: bool
    ) -> jnp.ndarray:
        cfg = self.config

        # Attention block
        residual = hidden_states
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_attention_heads,
            qkv_features=cfg.hidden_size,
            out_features=cfg.hidden_size,
            dropout_rate=cfg.attention_probs_dropout_prob,
            use_bias=True,
            name="self_attention",
        )(
            hidden_states,
            hidden_states,
            mask=attention_mask,
            deterministic=deterministic,
        )
        attn_out = nn.Dropout(rate=cfg.hidden_dropout_prob, name="attn_dropout")(
            attn_out, deterministic=deterministic
        )
        hidden_states = nn.LayerNorm(epsilon=cfg.layer_norm_eps, name="attn_ln")(
            residual + attn_out
        )

        # Feed-forward block
        residual = hidden_states
        ff = nn.Dense(cfg.intermediate_size, name="ffn_in")(hidden_states)
        ff = nn.gelu(ff, approximate=False)
        ff = nn.Dense(cfg.hidden_size, name="ffn_out")(ff)
        ff = nn.Dropout(rate=cfg.hidden_dropout_prob, name="ffn_dropout")(
            ff, deterministic=deterministic
        )
        hidden_states = nn.LayerNorm(epsilon=cfg.layer_norm_eps, name="ffn_ln")(residual + ff)
        return hidden_states


class BertForDiffusionBase(nn.Module):
    """Base diffusion BERT operating on per-token continuous features."""

    config: BertDiffusionConfig

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        timestep: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
        g_diag: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        del position_ids  # Kept only for API compatibility.
        cfg = self.config
        if cfg.is_decoder:
            raise NotImplementedError("Decoder mode is not supported.")

        x = nn.Dense(cfg.hidden_size // 2, name="inputs_proj_0")(inputs)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(cfg.hidden_size, name="inputs_proj_1")(x)

        x = nn.LayerNorm(epsilon=cfg.layer_norm_eps, name="emb_ln")(x)
        x = nn.Dropout(rate=cfg.hidden_dropout_prob, name="emb_dropout")(
            x, deterministic=deterministic
        )

        if g_diag is not None and cfg.condition_on_g_diag:
            x = x + nn.Dense(cfg.hidden_size, name="g_diag_to_hidden")(g_diag)

        t = jnp.asarray(timestep, dtype=inputs.dtype)
        t_encoded = GaussianFourierProjection(cfg.hidden_size, name="time_embed")(t)
        x = x + t_encoded[:, None, :]

        # Flax attention mask is boolean and broadcastable to [B, H, Q, K].
        mask_2d = attention_mask > 0
        attn_mask = nn.make_attention_mask(mask_2d, mask_2d, dtype=jnp.bool_)

        for i in range(cfg.num_hidden_layers):
            x = BertLayer(cfg, name=f"encoder_layer_{i}")(
                x, attention_mask=attn_mask, deterministic=deterministic
            )

        x = nn.Dense(cfg.hidden_size, name="decoder_0")(x)
        x = nn.gelu(x, approximate=False)
        x = nn.LayerNorm(epsilon=1e-12, name="decoder_ln")(x)
        x = nn.Dense(cfg.input_feat_dim, name="decoder_1")(x)
        return x


class BertForDiffusion(nn.Module):
    """Top-level FoldingDiff model that handles flatten/unflatten geometry format."""

    config: BertDiffusionConfig

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        timestep: jnp.ndarray,
        mask: jnp.ndarray,
        manifold: Optional[Any] = None,
        g_diag: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        n_feats = self.config.torsion_feat_dim
        token_feat_size = self.config.input_feat_dim
        left_pad = token_feat_size // n_feats
        right_pad = token_feat_size - left_pad

        inputs_reshaped = jnp.pad(inputs, ((0, 0), (left_pad, right_pad))).reshape(
            (inputs.shape[0], -1, token_feat_size)
        )
        attn_mask = jnp.concatenate(
            [jnp.ones((mask.shape[0], 1), dtype=mask.dtype), mask[:, ::n_feats]], axis=-1
        )

        g_diag_reshaped = None
        if g_diag is not None:
            log_g = jnp.log(jnp.clip(g_diag, a_min=1e-6)) * mask.astype(g_diag.dtype)
            g_diag_reshaped = jnp.pad(log_g, ((0, 0), (left_pad, right_pad))).reshape(
                (inputs.shape[0], -1, token_feat_size)
            )

        outputs = BertForDiffusionBase(self.config, name="base")(
            inputs=inputs_reshaped,
            timestep=timestep,
            attention_mask=attn_mask,
            g_diag=g_diag_reshaped,
            deterministic=deterministic,
        )
        outputs = outputs[:, : inputs_reshaped.shape[1], :]
        outputs = outputs.reshape((inputs.shape[0], -1))[:, left_pad:-right_pad]

        if manifold is not None:
            return manifold.to_tangent(outputs, inputs, mask=mask)
        return outputs

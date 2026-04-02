"""Basic JAX evaluation/sampling script (intrinsic flat/kinetic)."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict

import wandb
from diffgeo.angles_and_coords import angles_tensor_to_coords
from diffgeo.manifold import ExtrinsicMaskedTorus, IntrinsicMaskedTorus
from foldingdiff.bert_for_diffusion import BertDiffusionConfig, BertForDiffusion
from foldingdiff.dataset import CathCanonicalAnglesOnlyDataset
from RDM.beta_schedule import LinearBetaSchedule
from RDM.sde_lib import DiffusionMixture
from RDM.solver import sample_bridge_pc_batch
from score_based.sampling import sample_intrinsic_batch
from score_based.training import create_train_state

FT_NAMES = CathCanonicalAnglesOnlyDataset.feature_names["angles"]
FT_NAME_MAP = {
    "phi": "phi",
    "psi": "psi",
    "omega": "omega",
    "tau": "tau",
    "CA:C:1N": "CA-C-N",
    "C:1N:1CA": "C-N-CA",
}


def _decode_sample_to_angles(
    x: np.ndarray,
    length: int,
    coordinate_system: Literal["intrinsic", "extrinsic"] = "intrinsic",
    n_feats: int = 6,
) -> np.ndarray:
    # Training encodes angles[:, :-1, :] — the first (length-1) residues.
    # Decode directly to (length-1, n_feats); no padding shift.
    n = int((length - 1) * n_feats)
    if coordinate_system == "extrinsic":
        vals = x[: (2 * n)]
        vals = np.arctan2(vals[1::2], vals[0::2]).astype(np.float32, copy=False)
    else:
        vals = x[:n].astype(np.float32, copy=False)
    return vals.reshape(length - 1, n_feats)


def _kl_from_empirical(sampled: np.ndarray, reference: np.ndarray, nbins: int = 200) -> float:
    hist_s, edges = np.histogram(sampled, bins=nbins, range=(-math.pi, math.pi), density=False)
    hist_r, _ = np.histogram(reference, bins=edges, density=False)
    p = hist_s.astype(np.float64) + 1.0
    q = hist_r.astype(np.float64) + 1.0
    p /= np.sum(p)
    q /= np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _plot_distribution_overlap(
    sampled: np.ndarray, reference: np.ndarray, title: str, ax: Any, cumulative: bool = False
) -> None:
    bins = 60
    ax.hist(
        reference,
        bins=bins,
        range=(-math.pi, math.pi),
        density=True,
        alpha=0.45 if not cumulative else 1.0,
        label="Test",
        histtype="step" if cumulative else "bar",
        cumulative=cumulative,
    )
    ax.hist(
        sampled,
        bins=bins,
        range=(-math.pi, math.pi),
        density=True,
        alpha=0.45 if not cumulative else 1.0,
        label="Sampled",
        histtype="step" if cumulative else "bar",
        cumulative=cumulative,
    )
    ax.set_title(title)


def _plot_ramachandran(phi: np.ndarray, psi: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(dpi=300, figsize=(6, 5))
    ax.hist2d(phi, psi, bins=120, range=[[-math.pi, math.pi], [-math.pi, math.pi]], cmin=1)
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel("phi")
    ax.set_ylabel("psi")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_backbone_pdb(coords: np.ndarray, out_path: Path) -> None:
    n_atoms = coords.shape[0]
    if n_atoms % 3 != 0:
        raise ValueError(f"Expected 3*L atoms (N/CA/C), got {n_atoms}")
    n_res = n_atoms // 3
    atom_names = ("N", "CA", "C")
    with out_path.open("w", encoding="utf-8") as f:
        atom_id = 1
        for res_i in range(n_res):
            for atom_j, atom_name in enumerate(atom_names):
                x, y, z = coords[res_i * 3 + atom_j]
                line = (
                    f"ATOM  {atom_id:5d} {atom_name:>4s} {'GLY':>3s} A{res_i + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {atom_name[0]:>2s}\n"
                )
                f.write(line)
                atom_id += 1
        f.write("END\n")


def _angles_to_backbone_pdb(angles: np.ndarray, out_path: Path) -> None:
    coords = np.asarray(
        angles_tensor_to_coords(
            jnp.asarray(angles, dtype=jnp.float32),
            center_coords=False,
            return_ca_only=False,
        )
    )
    coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    _write_backbone_pdb(coords, out_path)


def _build_test_reference(cfg: EvalConfig) -> np.ndarray:
    test_ds = CathCanonicalAnglesOnlyDataset(
        pdbs=cfg.pdbs,
        split="test",
        pad=cfg.max_seq_len,
        min_length=cfg.min_seq_len,
        trim_strategy="leftalign",
        toy=cfg.toy,
        zero_center=False,
        use_cache=True,
        num_workers=cfg.dataset_workers,
    )
    if len(test_ds) == 0:
        raise RuntimeError("Test dataset is empty; cannot compute evaluation metrics.")

    stacked = []
    n_limit = (
        cfg.eval_test_limit if cfg.eval_test_limit and cfg.eval_test_limit > 0 else len(test_ds)
    )
    for i in range(min(len(test_ds), n_limit)):
        item = test_ds[i]
        angles = item["angles"][:-1, :]
        mask = item["geo_mask"].reshape(-1, 6).astype(bool)
        valid_rows = mask.all(axis=1)
        if np.any(valid_rows):
            stacked.append(angles[valid_rows].astype(np.float32, copy=False))
    if not stacked:
        raise RuntimeError("No valid test angles collected for evaluation.")
    return np.concatenate(stacked, axis=0)


def _compute_and_save_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
    plots_dir: Path,
) -> dict[str, float]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}

    multi_fig, multi_axes = plt.subplots(dpi=300, nrows=2, ncols=3, figsize=(14, 6), sharex=True)
    cdf_fig, cdf_axes = plt.subplots(dpi=300, nrows=2, ncols=3, figsize=(14, 6), sharex=True)

    for i, ft_name in enumerate(FT_NAMES):
        samp_values = generated[:, i]
        ref_values = reference[:, i]
        kl = _kl_from_empirical(samp_values, ref_values, nbins=200)
        metrics[f"kl_{ft_name}"] = kl

        _plot_distribution_overlap(
            samp_values,
            ref_values,
            title=f"{FT_NAME_MAP[ft_name]} distribution, KL={kl:.4f}",
            ax=multi_axes.flatten()[i],
            cumulative=False,
        )
        _plot_distribution_overlap(
            samp_values,
            ref_values,
            title=f"{FT_NAME_MAP[ft_name]} CDF",
            ax=cdf_axes.flatten()[i],
            cumulative=True,
        )
        if i == 0:
            multi_axes.flatten()[i].legend(loc="best")
            cdf_axes.flatten()[i].legend(loc="best")

    multi_fig.tight_layout()
    cdf_fig.tight_layout()
    angles_distribution_png = plots_dir / "angles_distribution.png"
    angles_cdf_png = plots_dir / "angles_cdf.png"
    multi_fig.savefig(angles_distribution_png, bbox_inches="tight")
    cdf_fig.savefig(angles_cdf_png, bbox_inches="tight")
    plt.close(multi_fig)
    plt.close(cdf_fig)

    phi_idx = FT_NAMES.index("phi")
    psi_idx = FT_NAMES.index("psi")
    rng = np.random.default_rng(seed=6489)
    ref_idx = rng.choice(len(reference), size=min(len(reference), len(generated)), replace=False)
    _plot_ramachandran(
        reference[ref_idx, phi_idx],
        reference[ref_idx, psi_idx],
        out_path=plots_dir / "ramachandran_test.png",
        title="Ramachandran Diagram - Test",
    )
    _plot_ramachandran(
        generated[:, phi_idx],
        generated[:, psi_idx],
        out_path=plots_dir / "ramachandran_generated.png",
        title="Ramachandran Diagram - Generated",
    )

    metrics["kl_mean"] = float(np.mean([metrics[f"kl_{k}"] for k in FT_NAMES]))
    return metrics


class EvalConfig(BaseSettings, cli_parse_args=True):
    model_config = SettingsConfigDict(cli_kebab_case=True)

    checkpoint_path: str = "weights/model_epoch_0001.msgpack"
    out_dir: str = "sampled"
    training_objective: Literal["score", "bridge_matching"] = "score"
    bridge_coordinates: Literal["intrinsic", "extrinsic"] = "extrinsic"
    bridge_use_pode: bool = True
    metric_type: Literal["kinetic_diag", "flat_torus"] = "kinetic_diag"

    n_steps: int = 1000
    eps: float = 1e-3
    pc_corrector_steps: int = 0
    pc_corrector_step_scale: float = 1.0
    pc_corrector_noise_scale: float = 1.0
    batch_size: int = 64
    samples_per_length: int = 10
    from_length: int = 50
    to_length: int = 60
    seed: int = 42

    # Dataset/evaluation
    pdbs: str = "cath"
    min_seq_len: int = 40
    toy: int = 0
    dataset_workers: int = 0
    eval_test_limit: int = 0
    sample_upload_count: int = 10
    save_pdb_samples: bool = True
    save_raw_angles: bool = False

    # W&B
    wandb_entity: str = "rdem"
    wandb_project: str = "Standard Metric - RiemannDiff"
    wandb_run_name: str | None = None
    wandb_mode: Literal["online", "offline", "disabled", "shared"] | None = "disabled"
    upload_samples_dir_artifact: bool = False

    # Fallback model/schedule values (overridden by checkpoint json if available)
    max_seq_len: int = 128
    net_size: int = 1
    dropout: float = 0.1
    relative_position: bool = True
    metric_condition_model: bool = True
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
    bridge_beta_0: float = 0.2
    bridge_beta_f: float = 0.001


def _fix_pdb_file(src: Path, dst: Path) -> bool:
    """Run pdbfixer on a PDB file and write fixed output to dst."""
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except Exception:
        return False
    try:
        fixer = PDBFixer(filename=str(src))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        with dst.open("w", encoding="utf-8") as out:
            PDBFile.writeFile(fixer.topology, fixer.positions, out)
        return True
    except Exception:
        return False


def _load_config_from_checkpoint_sidecar(cfg: EvalConfig) -> EvalConfig:
    json_path = Path(cfg.checkpoint_path).with_suffix(".json")
    if not json_path.exists():
        return cfg
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        c = payload.get("config", {})
    except Exception:
        return cfg
    merged = cfg.model_dump()
    for k in merged.keys():
        if k in c:
            merged[k] = c[k]
    merged["checkpoint_path"] = cfg.checkpoint_path
    merged["out_dir"] = cfg.out_dir
    merged["metric_type"] = cfg.metric_type
    merged["n_steps"] = cfg.n_steps
    merged["eps"] = cfg.eps
    merged["batch_size"] = cfg.batch_size
    merged["samples_per_length"] = cfg.samples_per_length
    merged["from_length"] = cfg.from_length
    merged["to_length"] = cfg.to_length
    return EvalConfig(**merged)


def _build_score_model(cfg: EvalConfig) -> BertForDiffusion:
    model_cfg = BertDiffusionConfig(
        num_attention_heads=cfg.net_size * 4,
        hidden_size=cfg.net_size * 128,
        intermediate_size=cfg.net_size * 256,
        num_hidden_layers=cfg.net_size * 4,
        hidden_dropout_prob=cfg.dropout,
        attention_probs_dropout_prob=cfg.dropout,
        input_feat_dim=6,
        torsion_feat_dim=6,
        max_position_embeddings=cfg.max_seq_len,
        relative_position=cfg.relative_position,
        condition_on_g_diag=cfg.metric_condition_model and cfg.metric_type == "kinetic_diag",
    )
    return BertForDiffusion(config=model_cfg)


def _build_bridge_models(cfg: EvalConfig) -> tuple[BertForDiffusion, BertForDiffusion]:
    bridge_feat_dim = 12 if cfg.bridge_coordinates == "extrinsic" else 6
    model_cfg = BertDiffusionConfig(
        num_attention_heads=cfg.net_size * 4,
        hidden_size=cfg.net_size * 128,
        intermediate_size=cfg.net_size * 256,
        num_hidden_layers=cfg.net_size * 4,
        hidden_dropout_prob=cfg.dropout,
        attention_probs_dropout_prob=cfg.dropout,
        input_feat_dim=bridge_feat_dim,
        torsion_feat_dim=6,
        max_position_embeddings=cfg.max_seq_len,
        relative_position=cfg.relative_position,
        condition_on_g_diag=False,
    )
    return BertForDiffusion(config=model_cfg), BertForDiffusion(config=model_cfg)


def _load_params(model: BertForDiffusion, cfg: EvalConfig):
    def _extract_params_tree(node):
        if not isinstance(node, Mapping):
            return node
        if "ema_params" in node:
            return _extract_params_tree(node["ema_params"])
        if "params" in node:
            return _extract_params_tree(node["params"])
        if "state" in node:
            return _extract_params_tree(node["state"])
        return node

    def _remap_legacy_attention_names(node):
        if not isinstance(node, Mapping):
            return node
        def _to_dense_qkv(param_tree):
            if not isinstance(param_tree, Mapping):
                return param_tree
            out_tree = dict(param_tree)
            k = out_tree.get("kernel", None)
            b = out_tree.get("bias", None)
            if hasattr(k, "ndim") and k.ndim == 3:
                out_tree["kernel"] = np.asarray(k).reshape(k.shape[0], k.shape[1] * k.shape[2])
            if hasattr(b, "ndim") and b.ndim == 2:
                out_tree["bias"] = np.asarray(b).reshape(b.shape[0] * b.shape[1])
            return out_tree

        def _to_dense_out(param_tree):
            if not isinstance(param_tree, Mapping):
                return param_tree
            out_tree = dict(param_tree)
            k = out_tree.get("kernel", None)
            if hasattr(k, "ndim") and k.ndim == 3:
                out_tree["kernel"] = np.asarray(k).reshape(k.shape[0] * k.shape[1], k.shape[2])
            return out_tree

        out = dict(node)
        base = out.get("base", out)
        if isinstance(base, Mapping):
            base_mut = dict(base)
            changed = False
            for k, v in list(base_mut.items()):
                if not (isinstance(k, str) and k.startswith("encoder_layer_")):
                    continue
                if not isinstance(v, Mapping):
                    continue
                layer = dict(v)
                sa = layer.get("self_attention", None)
                if isinstance(sa, Mapping):
                    if "self_attention_query" not in layer and "query" in sa:
                        layer["self_attention_query"] = _to_dense_qkv(sa["query"])
                    if "self_attention_key" not in layer and "key" in sa:
                        layer["self_attention_key"] = _to_dense_qkv(sa["key"])
                    if "self_attention_value" not in layer and "value" in sa:
                        layer["self_attention_value"] = _to_dense_qkv(sa["value"])
                    if "self_attention_output" not in layer and "out" in sa:
                        layer["self_attention_output"] = _to_dense_out(sa["out"])
                    layer.pop("self_attention", None)
                    base_mut[k] = layer
                    changed = True
            if changed:
                if "base" in out:
                    out["base"] = base_mut
                else:
                    out = base_mut
        return out

    def _normalize_params_tree(node):
        return _remap_legacy_attention_names(_extract_params_tree(node))

    b = Path(cfg.checkpoint_path).read_bytes()
    state_dict = flax.serialization.msgpack_restore(b)
    if isinstance(state_dict, dict):
        ema_params = state_dict.get("ema_params", None)
        if ema_params is not None:
            return _normalize_params_tree(ema_params)
        params = state_dict.get("params", None)
        if params is not None:
            return _normalize_params_tree(params)
        # Some legacy checkpoints nest params under "state".
        state_obj = state_dict.get("state", None)
        if isinstance(state_obj, dict):
            ema_params = state_obj.get("ema_params", None)
            if ema_params is not None:
                return _normalize_params_tree(ema_params)
            params = state_obj.get("params", None)
            if params is not None:
                return _normalize_params_tree(params)

    # Backward-compatible fallback for older checkpoints:
    # 1) full TrainState bytes
    # 2) raw params bytes
    d = 6 * (cfg.max_seq_len - 1)
    sample_x = jnp.zeros((1, d), dtype=jnp.float32)
    sample_mask = jnp.ones((1, d), dtype=jnp.float32)
    state = create_train_state(model, jax.random.PRNGKey(0), sample_x, sample_mask)
    try:
        loaded = flax.serialization.from_bytes(state, b)
        params = loaded.ema_params if getattr(loaded, "ema_params", None) is not None else loaded.params
        return _normalize_params_tree(params)
    except ValueError:
        raw = flax.serialization.msgpack_restore(b)
        raw = _normalize_params_tree(raw)
        target = state.params
        if isinstance(raw, Mapping):
            # Common legacy mismatch: checkpoint params saved without the outer "base"
            # scope while current model expects params["base"][...].
            if "base" in target and "base" not in raw:
                try:
                    return flax.serialization.from_state_dict(target, {"base": raw})
                except ValueError:
                    pass
            if "base" not in target and "base" in raw:
                try:
                    return flax.serialization.from_state_dict(target, raw["base"])
                except ValueError:
                    pass
            # Try direct state-dict restore as final structured fallback.
            try:
                return flax.serialization.from_state_dict(target, raw)
            except ValueError as exc:
                raise ValueError(
                    "Could not map checkpoint params to evaluation model structure. "
                    "Try passing matching --net-size/--max-seq-len/--relative-position or inspect checkpoint keys."
                ) from exc
        # Last fallback: treat as raw bytes for params-only checkpoints.
        return flax.serialization.from_bytes(target, b)


def _extract_eval_params(state_dict):
    ema_params = state_dict.get("ema_params", None)
    if ema_params is not None:
        return ema_params
    params = state_dict.get("params", None)
    if params is not None:
        return params
    return state_dict


def _load_bridge_params(cfg: EvalConfig):
    b = Path(cfg.checkpoint_path).read_bytes()
    state_dict = flax.serialization.msgpack_restore(b)
    if not (isinstance(state_dict, dict) and "state_f" in state_dict and "state_b" in state_dict):
        raise ValueError(
            "Bridge evaluation expects checkpoint with {'state_f', 'state_b'}; got incompatible format."
        )
    return _extract_eval_params(state_dict["state_f"]), _extract_eval_params(state_dict["state_b"])


def _sample_batch(
    params,
    model: BertForDiffusion,
    mask: jnp.ndarray,
    cfg: EvalConfig,
    rng: jax.Array,
) -> jnp.ndarray:
    return sample_intrinsic_batch(
        params=params,
        model=model,
        mask=mask,
        rng=rng,
        n_steps=cfg.n_steps,
        eps=cfg.eps,
        beta_0=cfg.beta_0,
        beta_f=cfg.beta_f,
        metric_type=cfg.metric_type,
        metric_cutoff=cfg.metric_cutoff,
        metric_eps=cfg.metric_eps,
        metric_normalize=cfg.metric_normalize,
        metric_clamp_min=cfg.metric_clamp_min,
        metric_clamp_max=cfg.metric_clamp_max,
        metric_anneal=cfg.metric_anneal,
        metric_anneal_data_lambda=cfg.metric_anneal_data_lambda,
        metric_anneal_prior_lambda=cfg.metric_anneal_prior_lambda,
        metric_anneal_power=cfg.metric_anneal_power,
        pc_corrector_steps=cfg.pc_corrector_steps,
        pc_corrector_step_scale=cfg.pc_corrector_step_scale,
        pc_corrector_noise_scale=cfg.pc_corrector_noise_scale,
    )


def _sample_batch_bridge(
    *,
    params_f,
    params_b,
    model_f: BertForDiffusion,
    model_b: BertForDiffusion,
    cfg: EvalConfig,
    mask: jnp.ndarray,
    rng: jax.Array,
) -> jnp.ndarray:
    manifold_dim = 6 * (cfg.max_seq_len - 1)
    manifold = (
        ExtrinsicMaskedTorus(manifold_dim)
        if cfg.bridge_coordinates == "extrinsic"
        else IntrinsicMaskedTorus(manifold_dim)
    )
    mix = DiffusionMixture(
        manifold=manifold,
        beta_schedule=LinearBetaSchedule(
            tf=1.0,
            t0=0.0,
            beta_0=cfg.bridge_beta_0,
            beta_f=cfg.bridge_beta_f,
        ),
        prior_type="unif",
        drift_scale=1.0,
        mix_type="log",
    )
    return sample_bridge_pc_batch(
        params_f=params_f,
        params_b=params_b,
        model_apply_f=model_f.apply,
        model_apply_b=model_b.apply,
        mix=mix,
        mask=mask,
        rng=rng,
        n_steps=cfg.n_steps,
        eps=cfg.eps,
        use_pode=cfg.bridge_use_pode,
    )


def main() -> None:
    cfg = _load_config_from_checkpoint_sidecar(EvalConfig())
    if cfg.training_objective == "bridge_matching":
        model_f, model_b = _build_bridge_models(cfg)
        params_f, params_b = _load_bridge_params(cfg)
        coord_for_decode = cfg.bridge_coordinates
    else:
        model = _build_score_model(cfg)
        params = _load_params(model, cfg)
        coord_for_decode = "intrinsic"

    out_dir = Path(cfg.out_dir)
    pdb_dir = out_dir / "sampled_pdb"
    fixed_pdb_dir = out_dir / "sampled_pdb_fixed"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_pdb_samples:
        pdb_dir.mkdir(parents=True, exist_ok=True)
        fixed_pdb_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    lengths = []
    for length in range(cfg.from_length, cfg.to_length + 1):
        lengths.extend([length] * cfg.samples_per_length)
    lengths_arr = np.asarray(lengths, dtype=np.int32)

    rng = jax.random.PRNGKey(cfg.seed)
    decoded: list[np.ndarray] = []
    pdb_paths: list[Path] = []
    saved = 0
    d = 6 * (cfg.max_seq_len - 1)
    d_sample = (
        d * 2
        if cfg.training_objective == "bridge_matching" and cfg.bridge_coordinates == "extrinsic"
        else d
    )

    if cfg.training_objective == "bridge_matching":
        sample_batch_compiled = jax.jit(
            lambda mask_in, rng_in: _sample_batch_bridge(
                params_f=params_f,
                params_b=params_b,
                model_f=model_f,
                model_b=model_b,
                cfg=cfg,
                mask=mask_in,
                rng=rng_in,
            )
        )
    else:
        sample_batch_compiled = jax.jit(
            lambda mask_in, rng_in: _sample_batch(
                params=params,
                model=model,
                mask=mask_in,
                cfg=cfg,
                rng=rng_in,
            )
        )

    total_batches = (len(lengths_arr) + cfg.batch_size - 1) // cfg.batch_size
    print(
        f"sampler_cfg metric={cfg.metric_type} n_steps={cfg.n_steps} "
        f"pc_steps={cfg.pc_corrector_steps} pc_step_scale={cfg.pc_corrector_step_scale} "
        f"pc_noise_scale={cfg.pc_corrector_noise_scale}"
    )
    print(
        f"sampling_start total_samples={len(lengths_arr)} batch_size={cfg.batch_size} total_batches={total_batches}"
    )
    for batch_idx, start in enumerate(range(0, len(lengths_arr), cfg.batch_size), start=1):
        batch_lengths = lengths_arr[start : start + cfg.batch_size]
        b = len(batch_lengths)
        mask = np.zeros((cfg.batch_size, d), dtype=np.float32)
        for i, length_i in enumerate(batch_lengths):
            mask[i, : (length_i - 1) * 6] = 1.0

        print(f"sampling_batch {batch_idx}/{total_batches} size={b}")
        rng, step_rng = jax.random.split(rng)
        x = sample_batch_compiled(jnp.asarray(mask), step_rng)
        x_np = np.asarray(jax.device_get(x), dtype=np.float32)[:b]
        if x_np.shape[-1] != d_sample:
            raise ValueError(
                f"Unexpected sampled shape {x_np.shape}; expected trailing dim {d_sample}"
            )

        for i in range(b):
            length_i = int(batch_lengths[i])
            angles = _decode_sample_to_angles(
                x_np[i],
                length_i,
                coordinate_system=coord_for_decode,
                n_feats=6,
            )
            decoded.append(angles)
            if cfg.save_raw_angles:
                np.save(out_dir / f"sample_len{length_i}_{saved:06d}.npy", angles)
            if cfg.save_pdb_samples:
                pdb_path = pdb_dir / f"rec_prot_{length_i}_sample_{saved:06d}.pdb"
                _angles_to_backbone_pdb(angles, pdb_path)
                pdb_paths.append(pdb_path)
            saved += 1

    if cfg.save_raw_angles:
        np.savez_compressed(
            out_dir / "samples.npz",
            lengths=lengths_arr,
            samples=np.array(decoded, dtype=object),
        )

    generated_stacked = np.concatenate(decoded, axis=0)
    test_values_stacked = _build_test_reference(cfg)
    metrics = _compute_and_save_metrics(generated_stacked, test_values_stacked, plots_dir=plots_dir)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if cfg.save_pdb_samples:
        print(f"saved_pdb_samples count={len(pdb_paths)} dir={pdb_dir}")

    run = None
    if cfg.wandb_mode != "disabled":
        upload_pdb_paths = pdb_paths
        if cfg.save_pdb_samples and len(pdb_paths) > 0:
            fixed_paths: list[Path] = []
            fixed_ok = 0
            for p in pdb_paths:
                dst = fixed_pdb_dir / p.name
                if _fix_pdb_file(p, dst):
                    fixed_paths.append(dst)
                    fixed_ok += 1
                else:
                    fixed_paths.append(p)
            upload_pdb_paths = fixed_paths
            print(f"pdbfix_before_wandb fixed={fixed_ok}/{len(pdb_paths)}")

        run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            mode=cfg.wandb_mode,
            config=cfg.model_dump(),
        )
        run.log(metrics)
        run.log(
            {
                "angles_distribution": [
                    wandb.Image(str(plots_dir / "angles_distribution.png")),
                    wandb.Image(str(plots_dir / "angles_cdf.png")),
                ],
                "ramachandran_diagrams": [
                    wandb.Image(str(plots_dir / "ramachandran_test.png")),
                    wandb.Image(str(plots_dir / "ramachandran_generated.png")),
                ],
            }
        )
        keep = min(cfg.sample_upload_count, len(upload_pdb_paths))
        if keep > 0:
            idx = np.linspace(0, len(upload_pdb_paths) - 1, num=keep, dtype=int)
            run.log(
                {
                    "generated_proteins": [wandb.Molecule(str(upload_pdb_paths[i])) for i in idx],
                }
            )
        if cfg.upload_samples_dir_artifact:
            artifact = wandb.Artifact(name="samples", type="dataset")
            if cfg.save_pdb_samples and fixed_pdb_dir.exists():
                artifact.add_dir(str(fixed_pdb_dir))
            else:
                artifact.add_dir(str(pdb_dir))
            run.log_artifact(artifact)
        run.finish()

    print(f"saved_samples={saved}")
    print(f"out_dir={out_dir}")
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()

"""Basic JAX evaluation/sampling script (intrinsic flat/kinetic)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flax.serialization
import jax
import jax.numpy as jnp
from jax import lax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pydantic_settings import BaseSettings, SettingsConfigDict

from diffgeo.angles_and_coords import angles_tensor_to_coords
from diffgeo.kinetic_metric import compute_kinetic_metric_diag
from foldingdiff.bert_for_diffusion import BertDiffusionConfig, BertForDiffusion
from foldingdiff.dataset import CathCanonicalAnglesOnlyDataset
from utils.train import create_train_state


FT_NAMES = CathCanonicalAnglesOnlyDataset.feature_names["angles"]
FT_NAME_MAP = {
    "phi": "phi",
    "psi": "psi",
    "omega": "omega",
    "tau": "tau",
    "CA:C:1N": "CA-C-N",
    "C:1N:1CA": "C-N-CA",
}


def _wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


def _to_angles_lengths(x: jnp.ndarray, mask: jnp.ndarray, n_feats: int = 6) -> tuple[jnp.ndarray, jnp.ndarray]:
    bsz, d = x.shape
    pad_minus_1 = d // n_feats
    pad = pad_minus_1 + 1
    angles = jnp.zeros((bsz, pad, n_feats), dtype=x.dtype).at[:, :pad_minus_1, :].set(
        x.reshape(bsz, pad_minus_1, n_feats)
    )
    lengths = jnp.clip((jnp.sum(mask, axis=-1) / n_feats).astype(jnp.int32) + 1, a_min=1, a_max=pad)
    return angles, lengths


def _sigma2_linear(t: jnp.ndarray, beta_0: float, beta_f: float) -> jnp.ndarray:
    return beta_0 * t + 0.5 * (beta_f - beta_0) * (t * t)


def _beta_t(t: jnp.ndarray, beta_0: float, beta_f: float) -> jnp.ndarray:
    return beta_0 + (beta_f - beta_0) * t


def _anneal_lambda_from_sigma2(
    sigma2: jnp.ndarray,
    sigma2_max: jnp.ndarray,
    enabled: bool,
    data_lambda: float,
    prior_lambda: float,
    power: float,
) -> jnp.ndarray:
    if not enabled:
        return jnp.ones_like(sigma2)
    u = jnp.clip(sigma2 / jnp.clip(sigma2_max, a_min=1e-8), a_min=0.0, a_max=1.0)
    u = jnp.power(u, power)
    return jnp.clip(data_lambda + (prior_lambda - data_lambda) * u, a_min=0.0, a_max=1.0)


def _decode_sample_to_angles(x: np.ndarray, length: int, n_feats: int = 6) -> np.ndarray:
    n = int((length - 1) * n_feats)
    vals = x[:n]
    vals = np.pad(vals, (1, n_feats - 1), mode="constant", constant_values=0.0)
    return vals.reshape(-1, n_feats).astype(np.float32, copy=False)


def _kl_from_empirical(sampled: np.ndarray, reference: np.ndarray, nbins: int = 200) -> float:
    hist_s, edges = np.histogram(sampled, bins=nbins, range=(-math.pi, math.pi), density=False)
    hist_r, _ = np.histogram(reference, bins=edges, density=False)
    p = hist_s.astype(np.float64) + 1.0
    q = hist_r.astype(np.float64) + 1.0
    p /= np.sum(p)
    q /= np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _plot_distribution_overlap(sampled: np.ndarray, reference: np.ndarray, title: str, ax: Any, cumulative: bool = False) -> None:
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
    n_limit = cfg.eval_test_limit if cfg.eval_test_limit and cfg.eval_test_limit > 0 else len(test_ds)
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
    metric_type: Literal["kinetic_diag", "flat_torus"] = "kinetic_diag"

    n_steps: int = 1000
    eps: float = 1e-3
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


def _build_model(cfg: EvalConfig) -> BertForDiffusion:
    model_cfg = BertDiffusionConfig(
        num_attention_heads=cfg.net_size * 4,
        hidden_size=cfg.net_size * 128,
        intermediate_size=cfg.net_size * 256,
        num_hidden_layers=cfg.net_size * 4,
        hidden_dropout_prob=cfg.dropout,
        attention_probs_dropout_prob=cfg.dropout,
        input_feat_dim=6,
        torsion_feat_dim=6,
        condition_on_g_diag=cfg.metric_condition_model and cfg.metric_type == "kinetic_diag",
    )
    return BertForDiffusion(config=model_cfg)


def _load_params(model: BertForDiffusion, cfg: EvalConfig):
    d = 6 * (cfg.max_seq_len - 1)
    sample_x = jnp.zeros((1, d), dtype=jnp.float32)
    sample_mask = jnp.ones((1, d), dtype=jnp.float32)
    state = create_train_state(model, jax.random.PRNGKey(0), sample_x, sample_mask)
    b = Path(cfg.checkpoint_path).read_bytes()
    loaded = flax.serialization.from_bytes(state, b)
    return loaded.params


def _sample_batch(
    params,
    model: BertForDiffusion,
    mask: jnp.ndarray,
    cfg: EvalConfig,
    rng: jax.Array,
) -> jnp.ndarray:
    bsz, _ = mask.shape
    rng, rng_init = jax.random.split(rng)
    x0 = (jax.random.uniform(rng_init, mask.shape, minval=-jnp.pi, maxval=jnp.pi) * mask).astype(jnp.float32)

    ts = jnp.linspace(1.0 - cfg.eps, 0.0 + cfg.eps, cfg.n_steps, dtype=jnp.float32)
    sigma2_max = jnp.clip(_sigma2_linear(jnp.ones((bsz,), dtype=jnp.float32), cfg.beta_0, cfg.beta_f), a_min=1e-8)

    def _body(k: int, carry: tuple[jnp.ndarray, jax.Array]) -> tuple[jnp.ndarray, jax.Array]:
        x, rng_loop = carry
        t = ts[k]
        t_next = ts[k + 1]
        dt = jnp.abs(t - t_next)
        vec_t = jnp.full((bsz,), t, dtype=jnp.float32)
        beta_t = _beta_t(vec_t, cfg.beta_0, cfg.beta_f)
        sigma2_t = jnp.clip(_sigma2_linear(vec_t, cfg.beta_0, cfg.beta_f), a_min=1e-8)

        g_diag = None
        sigma_diag = jnp.ones_like(mask)
        if cfg.metric_type == "kinetic_diag":
            angles, lengths = _to_angles_lengths(x, mask)
            g_diag = compute_kinetic_metric_diag(
                angles_batch=angles,
                lengths=lengths,
                geo_mask=mask,
                cutoff=cfg.metric_cutoff,
                eps=cfg.metric_eps,
                normalize=cfg.metric_normalize,
                clamp_min=cfg.metric_clamp_min,
                clamp_max=cfg.metric_clamp_max,
            )
            lam = _anneal_lambda_from_sigma2(
                sigma2_t,
                sigma2_max,
                enabled=cfg.metric_anneal,
                data_lambda=cfg.metric_anneal_data_lambda,
                prior_lambda=cfg.metric_anneal_prior_lambda,
                power=cfg.metric_anneal_power,
            )
            g_diag = (1.0 - lam[:, None]) + lam[:, None] * g_diag
            g_diag = jnp.nan_to_num(g_diag, nan=1.0, posinf=1e6, neginf=1.0)
            sigma_diag = jnp.clip(1.0 / g_diag, a_min=1e-8)

        eps_pred = model.apply(
            {"params": params},
            inputs=x,
            timestep=vec_t,
            mask=mask,
            manifold=None,
            g_diag=g_diag,
            deterministic=True,
        )
        if cfg.metric_type == "kinetic_diag":
            score = -jnp.sqrt(jnp.clip(g_diag, a_min=1e-8)) * eps_pred / jnp.sqrt(2.0 * sigma2_t[:, None])
        else:
            score = -eps_pred / jnp.sqrt(2.0 * sigma2_t[:, None])

        rng_loop, rng_noise = jax.random.split(rng_loop)
        noise = jax.random.normal(rng_noise, x.shape, dtype=x.dtype) * mask
        drift = 2.0 * dt * beta_t[:, None] * sigma_diag * score
        diff = jnp.sqrt(2.0 * dt * beta_t)[:, None] * jnp.sqrt(sigma_diag) * noise
        x_next = _wrap_to_pi(x + drift + diff) * mask
        return (x_next, rng_loop)

    x_fin, _ = lax.fori_loop(0, cfg.n_steps - 1, _body, (x0, rng))
    return x_fin


def main() -> None:
    cfg = _load_config_from_checkpoint_sidecar(EvalConfig())
    model = _build_model(cfg)
    params = _load_params(model, cfg)

    out_dir = Path(cfg.out_dir)
    pdb_dir = out_dir / "sampled_pdb"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"sampling_start total_samples={len(lengths_arr)} batch_size={cfg.batch_size} total_batches={total_batches}")
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

        for i in range(b):
            length_i = int(batch_lengths[i])
            angles = _decode_sample_to_angles(x_np[i], length_i, n_feats=6)
            decoded.append(angles)
            np.save(out_dir / f"sample_len{length_i}_{saved:06d}.npy", angles)
            pdb_path = pdb_dir / f"rec_prot_{length_i}_sample_{saved:06d}.pdb"
            _angles_to_backbone_pdb(angles, pdb_path)
            pdb_paths.append(pdb_path)
            saved += 1

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

    run = None
    if cfg.wandb_mode != "disabled":
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
        keep = min(cfg.sample_upload_count, len(pdb_paths))
        if keep > 0:
            idx = np.linspace(0, len(pdb_paths) - 1, num=keep, dtype=int)
            run.log(
                {
                    "generated_proteins": [wandb.Molecule(str(pdb_paths[i])) for i in idx],
                }
            )
        if cfg.upload_samples_dir_artifact:
            artifact = wandb.Artifact(name="samples", type="dataset")
            artifact.add_dir(str(pdb_dir))
            run.log_artifact(artifact)
        run.finish()

    print(f"saved_samples={saved}")
    print(f"out_dir={out_dir}")
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()

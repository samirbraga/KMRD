from __future__ import annotations

import numpy as np

from evaluation.basic_evaluation import (
    FT_NAMES,
    _compute_and_save_metrics,
    _decode_sample_to_angles,
    _kl_from_empirical,
    _write_backbone_pdb,
)


def test_decode_sample_to_angles_shape_contract() -> None:
    length = 5
    d = 6 * (length - 1)
    x = np.linspace(-1.0, 1.0, d, dtype=np.float32)
    angles = _decode_sample_to_angles(x, length=length, n_feats=6)
    assert angles.shape == (length, 6)


def test_kl_from_empirical_is_finite_and_small_for_identical_samples() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(-np.pi, np.pi, size=2048).astype(np.float32)
    kl_same = _kl_from_empirical(a, a, nbins=120)
    b = rng.uniform(-np.pi, np.pi, size=2048).astype(np.float32)
    kl_diff = _kl_from_empirical(a, b, nbins=120)
    assert np.isfinite(kl_same)
    assert np.isfinite(kl_diff)
    assert kl_same <= kl_diff + 1e-6


def test_write_backbone_pdb_contract(tmp_path) -> None:
    n_res = 4
    coords = np.zeros((n_res * 3, 3), dtype=np.float32)
    out = tmp_path / "sample.pdb"
    _write_backbone_pdb(coords, out)
    text = out.read_text(encoding="utf-8").strip().splitlines()
    atom_lines = [ln for ln in text if ln.startswith("ATOM")]
    assert len(atom_lines) == n_res * 3
    assert text[-1] == "END"


def test_compute_and_save_metrics_outputs_files_and_keys(tmp_path) -> None:
    rng = np.random.default_rng(1)
    generated = rng.uniform(-np.pi, np.pi, size=(256, len(FT_NAMES))).astype(np.float32)
    reference = rng.uniform(-np.pi, np.pi, size=(256, len(FT_NAMES))).astype(np.float32)
    metrics = _compute_and_save_metrics(generated=generated, reference=reference, plots_dir=tmp_path)

    for name in FT_NAMES:
        assert f"kl_{name}" in metrics
        assert np.isfinite(metrics[f"kl_{name}"])
    assert "kl_mean" in metrics
    assert np.isfinite(metrics["kl_mean"])

    assert (tmp_path / "angles_distribution.png").exists()
    assert (tmp_path / "angles_cdf.png").exists()
    assert (tmp_path / "ramachandran_test.png").exists()
    assert (tmp_path / "ramachandran_generated.png").exists()


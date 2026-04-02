from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from tmtools import tm_align

from evaluation.basic_evaluation import _angles_to_backbone_pdb
from foldingdiff.dataset import CATH_DIR, _featurize_single_canonical_angles


def _kabsch_ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Expected (N,3), got {a.shape}")
    ac = a - np.mean(a, axis=0, keepdims=True)
    bc = b - np.mean(b, axis=0, keepdims=True)
    h = ac.T @ bc
    u, _, vt = np.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    a_aligned = ac @ r
    diff = a_aligned - bc
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _resname_to_aa(resname: str) -> str:
    key = resname.strip().lower()
    return protein_letters_3to1.get(key, "X")


def _read_backbone_ca_coords_and_seq(pdb_path: Path) -> tuple[np.ndarray, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.name, str(pdb_path))
    coords = []
    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
                    seq.append(_resname_to_aa(residue.resname))
            if coords:
                break
        if coords:
            break
    if not coords:
        raise RuntimeError(f"No CA atoms found in {pdb_path}")
    return np.asarray(coords, dtype=np.float64), "".join(seq)


def _cossin_roundtrip(angles: np.ndarray) -> np.ndarray:
    flat = angles[:-1].reshape(-1)
    cossin = np.stack([np.cos(flat), np.sin(flat)], axis=-1).reshape(-1)
    back = np.arctan2(cossin[1::2], cossin[0::2]).astype(np.float32, copy=False)
    rec = np.zeros_like(angles, dtype=np.float32)
    rec[:-1] = back.reshape(angles.shape[0] - 1, angles.shape[1])
    return rec


def _circ_abs_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def test_dataset_pdb_angles_cossin_pdb_roundtrip(tmp_path: Path) -> None:
    pdb_dir = CATH_DIR / "dompdb"
    if not pdb_dir.exists():
        pytest.skip(f"Local dataset not available: {pdb_dir}")

    candidates = sorted(glob.glob(str(pdb_dir / "*")))
    if not candidates:
        pytest.skip(f"No PDB files found in {pdb_dir}")

    feat = None
    source_pdb = None
    for p in candidates[:200]:
        cur = _featurize_single_canonical_angles(p)
        if cur is None:
            continue
        if int(cur["angles"].shape[0]) >= 50:
            feat = cur
            source_pdb = Path(p)
            break
    if feat is None or source_pdb is None:
        pytest.skip("Could not find a suitable dataset PDB sample with enough residues.")

    # Dataset featurizer may emit NaNs on boundary terms; dataset pipeline zero-fills those.
    angles_src = np.nan_to_num(feat["angles"], nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    angles_rec = _cossin_roundtrip(angles_src)

    # cossin inverse should recover all encoded rows (all except the final row).
    max_rt = float(np.max(_circ_abs_delta(angles_src[:-1], angles_rec[:-1])))
    assert max_rt < 1e-5

    ref_pdb = tmp_path / "reference_from_angles.pdb"
    _angles_to_backbone_pdb(angles_src, ref_pdb)

    rec_pdb = tmp_path / "reconstructed_from_cossin.pdb"
    _angles_to_backbone_pdb(angles_rec, rec_pdb)

    rec_feat = _featurize_single_canonical_angles(str(rec_pdb))
    if rec_feat is None:
        raise RuntimeError("Failed to featurize reconstructed PDB")
    angles_refeat = np.nan_to_num(
        rec_feat["angles"], nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)

    n = min(angles_src.shape[0], angles_refeat.shape[0])
    mean_circ_err = float(np.mean(_circ_abs_delta(angles_src[:n], angles_refeat[:n])))
    assert mean_circ_err < 0.15

    ca_src, _ = _read_backbone_ca_coords_and_seq(ref_pdb)
    ca_rec, _ = _read_backbone_ca_coords_and_seq(rec_pdb)
    n_ca = min(ca_src.shape[0], ca_rec.shape[0])
    ca_rmsd = _kabsch_ca_rmsd(ca_src[:n_ca], ca_rec[:n_ca])
    assert ca_rmsd < 1e-3

    # Compare with original PDB via TM-align, accounting for canonical-geometry gap.
    ca_orig, seq_orig = _read_backbone_ca_coords_and_seq(source_pdb)
    n_ref = min(ca_orig.shape[0], ca_src.shape[0])
    n_rec = min(ca_orig.shape[0], ca_rec.shape[0])
    tm_ref = tm_align(ca_orig[:n_ref], ca_src[:n_ref], seq_orig[:n_ref], "G" * n_ref)
    tm_rec = tm_align(ca_orig[:n_rec], ca_rec[:n_rec], seq_orig[:n_rec], "G" * n_rec)
    assert abs(float(tm_rec.tm_norm_chain1) - float(tm_ref.tm_norm_chain1)) < 1e-3
    assert abs(float(tm_rec.tm_norm_chain2) - float(tm_ref.tm_norm_chain2)) < 1e-3

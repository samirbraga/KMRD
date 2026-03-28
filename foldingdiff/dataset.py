"""Dataset utilities for canonical backbone angles only."""

from __future__ import annotations

import functools
import glob
import gzip
import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_angle, calc_dihedral

LOCAL_DATA_DIR = Path(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)
CATH_DIR = LOCAL_DATA_DIR / "cath"
ALPHAFOLD_DIR = LOCAL_DATA_DIR / "alphafold"

TRIM_STRATEGIES = Literal["leftalign", "discard"]


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    """Wrap radians into [-pi, pi]."""
    return ((values + np.pi) % (2.0 * np.pi)) - np.pi


def _wrapped_mean(values: np.ndarray) -> np.ndarray:
    """Circular mean per feature with NaN-aware reduction."""
    s = np.nanmean(np.sin(values), axis=0)
    c = np.nanmean(np.cos(values), axis=0)
    return np.arctan2(s, c)


def _maybe_zero_center(angles: np.ndarray, means: Optional[np.ndarray]) -> np.ndarray:
    """Apply wrapped zero-centering when means are provided."""
    if means is None:
        return angles
    return _wrap_to_pi(angles - means)


def _trim_to_pad(
    angles: np.ndarray, pad: int, trim_strategy: TRIM_STRATEGIES
) -> Tuple[np.ndarray, int]:
    """Trim angles according to strategy and return effective sequence length."""
    if angles.shape[0] > pad:
        if trim_strategy == "leftalign":
            angles = angles[:pad]
        else:
            raise ValueError(
                f"Found length {angles.shape[0]} > pad {pad} with trim_strategy={trim_strategy}"
            )
    return angles, int(min(pad, angles.shape[0]))


def _build_padded_outputs(
    angles: np.ndarray, length: int, pad: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build padded angles, cossin stream, and masks from preprocessed angles."""
    n_feats = angles.shape[1]
    padded_angles = np.pad(angles, ((0, pad - length), (0, 0)), mode="constant")

    flat = padded_angles[:-1].reshape(-1)
    cossin = np.stack([np.cos(flat), np.sin(flat)], axis=-1).reshape(-1)

    attn_mask = np.zeros((pad,), dtype=np.float32)
    attn_mask[:length] = 1.0

    geo_mask = np.zeros(((pad - 1) * n_feats,), dtype=np.float32)
    geo_mask[: (max(length - 1, 0) * n_feats)] = 1.0

    return (
        padded_angles.astype(np.float32, copy=False),
        cossin.astype(np.float32, copy=False),
        attn_mask,
        geo_mask,
    )


class CathCanonicalAnglesOnlyDataset:
    """
    CATH/AlphaFold dataset returning only canonical angles:
    phi, psi, omega, tau, CA:C:1N, C:1N:1CA
    """

    feature_names = {
        "angles": ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
    }
    feature_is_angular = {"angles": [True, True, True, True, True, True]}

    def __init__(
        self,
        pdbs: Union[Literal["cath", "alphafold"], str, List[str], Tuple[str, ...]] = "cath",
        split: Optional[Literal["train", "test", "validation"]] = None,
        pad: int = 128,
        min_length: int = 40,
        trim_strategy: TRIM_STRATEGIES = "leftalign",
        toy: int = 0,
        zero_center: bool = True,
        use_cache: bool = True,
        cache_dir: Path = Path(os.path.dirname(os.path.abspath(__file__))),
        cache_path: Optional[Path] = None,
    ) -> None:
        if pad <= min_length:
            raise ValueError(f"pad ({pad}) must be > min_length ({min_length})")
        if trim_strategy not in ("leftalign", "discard"):
            raise ValueError(f"Unknown trim strategy: {trim_strategy}")

        self.pad = pad
        self.min_length = min_length
        self.trim_strategy = trim_strategy
        self.pdbs_src = pdbs

        self.fnames = self._get_pdb_fnames(pdbs)
        self.cache_dir = Path(cache_dir)
        self.cache_path = Path(cache_path) if cache_path is not None else self._build_cache_path()

        work_fnames = self.fnames
        if toy:
            if isinstance(toy, bool):
                toy = 150
            work_fnames = work_fnames[:toy]
            logging.info("Loading toy dataset of %s structures", toy)

        self.structures: List[Dict[str, np.ndarray]]
        if use_cache and self.cache_path.exists() and not toy:
            logging.info("Loading dataset cache from %s", self.cache_path)
            with open(self.cache_path, "rb") as source:
                self.structures = pickle.load(source)
        else:
            self.structures = self._compute_featurization(work_fnames)
            if use_cache and not toy:
                logging.info("Saving dataset cache to %s", self.cache_path)
                os.makedirs(self.cache_path.parent, exist_ok=True)
                with open(self.cache_path, "wb") as sink:
                    pickle.dump(self.structures, sink)

        if self.min_length:
            orig_len = len(self.structures)
            self.structures = [
                s for s in self.structures if s["angles"].shape[0] >= self.min_length
            ]
            logging.info(
                "Removed %s/%s structures shorter than %s",
                orig_len - len(self.structures),
                orig_len,
                self.min_length,
            )

        if self.trim_strategy == "discard":
            orig_len = len(self.structures)
            self.structures = [s for s in self.structures if s["angles"].shape[0] <= self.pad]
            logging.info(
                "Removed %s/%s structures longer than %s",
                orig_len - len(self.structures),
                orig_len,
                self.pad,
            )

        self.rng = np.random.default_rng(seed=6489)
        self.rng.shuffle(self.structures)
        if split is not None:
            split_idx = int(len(self.structures) * 0.8)
            valid_size = int(len(self.structures) * 0.1)
            if split == "train":
                self.structures = self.structures[:split_idx]
            elif split == "validation":
                self.structures = self.structures[split_idx : split_idx + valid_size]
            elif split == "test":
                self.structures = self.structures[split_idx + valid_size :]
            else:
                raise ValueError(f"Unknown split: {split}")

        self.means: Optional[np.ndarray] = None
        if zero_center and self.structures:
            concat_angles = np.concatenate([s["angles"] for s in self.structures], axis=0)
            self.means = _wrapped_mean(concat_angles)

        self.all_lengths = [s["angles"].shape[0] for s in self.structures]
        self._length_rng = np.random.default_rng(seed=6489)

    @property
    def cache_fname(self) -> str:
        return str(self.cache_path)

    @functools.cached_property
    def filenames(self) -> List[str]:
        return [s["fname"] for s in self.structures]

    def _build_cache_path(self) -> Path:
        if isinstance(self.pdbs_src, str) and os.path.isdir(self.pdbs_src):
            src_key = os.path.basename(self.pdbs_src)
        else:
            src_key = str(self.pdbs_src)

        hash_md5 = hashlib.md5()
        for fname in self.fnames:
            hash_md5.update(os.path.basename(fname).encode())
        filename_hash = hash_md5.hexdigest()

        return self.cache_dir / f"cache_canonical_angles_only_{src_key}_{filename_hash}.pkl"

    def _get_pdb_fnames(
        self, pdbs: Union[Literal["cath", "alphafold"], str, List[str], Tuple[str, ...]]
    ) -> List[str]:
        if isinstance(pdbs, (list, tuple)):
            fnames = list(pdbs)
            for fname in fnames:
                if not os.path.isfile(fname):
                    raise FileNotFoundError(f"Given file does not exist: {fname}")
            return fnames

        if os.path.isfile(pdbs):
            return [str(pdbs)]

        if os.path.isdir(pdbs):
            fnames: List[str] = []
            for ext in (".pdb", ".pdb.gz"):
                fnames.extend(glob.glob(os.path.join(pdbs, f"*{ext}")))
            if not fnames:
                raise FileNotFoundError(f"No PDB files found in {pdbs}")
            return sorted(fnames)

        if pdbs == "cath":
            fnames = glob.glob(os.path.join(CATH_DIR, "dompdb", "*"))
            if not fnames:
                raise FileNotFoundError(f"No files found in {CATH_DIR}/dompdb")
            return sorted(fnames)

        if pdbs == "alphafold":
            fnames = glob.glob(os.path.join(ALPHAFOLD_DIR, "*.pdb.gz"))
            if not fnames:
                raise FileNotFoundError(f"No files found in {ALPHAFOLD_DIR}")
            return sorted(fnames)

        raise ValueError(f"Unknown pdb set: {pdbs}")

    def _read_structure(self, fname: str):
        parser = PDBParser(QUIET=True)
        if fname.endswith(".gz"):
            with gzip.open(fname, "rt") as handle:
                return parser.get_structure(os.path.basename(fname), handle)
        return parser.get_structure(os.path.basename(fname), fname)

    @staticmethod
    def _extract_chain_residues(structure) -> List:
        residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " ":
                        continue
                    if all(atom in residue for atom in ("N", "CA", "C")):
                        residues.append(residue)
                if residues:
                    return residues
        return residues

    def _angles_from_residues(self, residues: Sequence) -> np.ndarray:
        n = len(residues)
        if n == 0:
            return np.empty((0, 6), dtype=np.float32)

        arr = np.full((n, 6), np.nan, dtype=np.float64)
        for i in range(n):
            r_i = residues[i]
            n_i = r_i["N"].get_vector()
            ca_i = r_i["CA"].get_vector()
            c_i = r_i["C"].get_vector()

            if i > 0:
                c_prev = residues[i - 1]["C"].get_vector()
                arr[i, 0] = calc_dihedral(c_prev, n_i, ca_i, c_i)  # phi

            if i < n - 1:
                n_next = residues[i + 1]["N"].get_vector()
                ca_next = residues[i + 1]["CA"].get_vector()
                c_next = residues[i + 1]["C"].get_vector()
                arr[i, 1] = calc_dihedral(n_i, ca_i, c_i, n_next)  # psi
                arr[i, 2] = calc_dihedral(ca_i, c_i, n_next, ca_next)  # omega
                # Match the legacy PyTorch featurization, which stores tau for
                # residue i+1 at row i, leaving the final row empty.
                arr[i, 3] = calc_angle(n_next, ca_next, c_next)  # tau
                arr[i, 4] = calc_angle(ca_i, c_i, n_next)  # CA:C:1N
                arr[i, 5] = calc_angle(c_i, n_next, ca_next)  # C:1N:1CA

        return _wrap_to_pi(arr).astype(np.float32, copy=False)

    def _compute_featurization(self, fnames: Sequence[str]) -> List[Dict[str, np.ndarray]]:
        structures: List[Dict[str, np.ndarray]] = []
        for fname in fnames:
            try:
                structure = self._read_structure(fname)
                residues = self._extract_chain_residues(structure)
                angles = self._angles_from_residues(residues)
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to parse %s: %s", fname, exc)
                continue

            if angles.shape[0] == 0:
                continue
            structures.append({"angles": angles, "fname": fname})
        return structures

    def sample_length(self, n: int = 1) -> Union[int, List[int]]:
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if n == 1:
            return int(self._length_rng.choice(self.all_lengths))
        return self._length_rng.choice(self.all_lengths, size=n, replace=True).tolist()

    def get_masked_means(self) -> Optional[np.ndarray]:
        return None if self.means is None else np.copy(self.means)

    def set_masked_means(self, mean_values: np.ndarray) -> None:
        if self.means is None:
            raise NotImplementedError("zero_center=False, means are not initialized")
        mean_values = np.asarray(mean_values)
        if mean_values.shape != self.means.shape:
            raise ValueError(f"Expected means shape {self.means.shape}, got {mean_values.shape}")
        self.means = mean_values.copy()

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(
        self, index: int, ignore_zero_center: bool = False
    ) -> Dict[str, np.ndarray]:
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        angles = self.structures[index]["angles"].copy()
        means = None if ignore_zero_center else self.means
        angles = _maybe_zero_center(angles, means)

        np.nan_to_num(angles, copy=False, nan=0.0)
        angles, length = _trim_to_pad(
            angles=angles,
            pad=self.pad,
            trim_strategy=self.trim_strategy,
        )
        padded_angles, cossin, attn_mask, geo_mask = _build_padded_outputs(
            angles=angles,
            length=length,
            pad=self.pad,
        )

        return {
            "angles": padded_angles,
            "lengths": np.int64(length),
            "cossin": cossin,
            "attn_mask": attn_mask,
            "geo_mask": geo_mask,
        }

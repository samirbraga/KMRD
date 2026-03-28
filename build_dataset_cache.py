"""Precompute and cache FoldingDiff canonical-angle dataset features."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from foldingdiff.dataset import CathCanonicalAnglesOnlyDataset


class BuildDatasetConfig(BaseSettings, cli_parse_args=True):
    model_config = SettingsConfigDict(cli_kebab_case=True)

    pdbs: str = "cath"
    pad: int = 128
    min_length: int = 0
    trim_strategy: str = "leftalign"
    zero_center: bool = False
    num_workers: int = 8
    use_cache: bool = True
    split: str | None = None
    toy: int = 0


def main() -> None:
    cfg = BuildDatasetConfig()
    ds = CathCanonicalAnglesOnlyDataset(
        pdbs=cfg.pdbs,
        split=cfg.split,
        pad=cfg.pad,
        min_length=cfg.min_length,
        trim_strategy=cfg.trim_strategy,  # type: ignore[arg-type]
        toy=cfg.toy,
        zero_center=cfg.zero_center,
        use_cache=cfg.use_cache,
        num_workers=cfg.num_workers,
    )
    print(f"built_len={len(ds)}")
    print(f"cache_path={ds.cache_fname}")


if __name__ == "__main__":
    main()


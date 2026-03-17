from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridging.MD.paths import GENERATED_DIR
from bridging.utils.affinity import to_float


DEFAULT_MMGBSA_DIR = GENERATED_DIR / "MMGBSA"


def _norm_path_text(path_like) -> str:
    return str(path_like).replace("\\", "/").lower()


def default_mmgbsa_path(dataset_path: Path) -> Path:
    return DEFAULT_MMGBSA_DIR / f"{dataset_path.stem}_mmgbsa_estimates.csv"


def _load_baseline_map(
    dataset_path: Path,
    path: Path,
    value_cols: list[str],
) -> dict[int, float]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"]

    if "dataset" in df.columns:
        target = _norm_path_text(dataset_path)
        ds_norm = df["dataset"].astype(str).map(_norm_path_text)
        matched = ds_norm == target
        if matched.any():
            df = df[matched]

    if "row_index" not in df.columns:
        return {}

    value_col = None
    for col in value_cols:
        if col in df.columns:
            value_col = col
            break
    if value_col is None:
        return {}

    out = {}
    for row in df.to_dict("records"):
        idx = to_float(row.get("row_index"))
        val = to_float(row.get(value_col))
        if idx is None or val is None:
            continue
        out[int(idx)] = float(val)
    return out


def load_mmgbsa_map(dataset_path: Path, mmgbsa_path: str | Path | None = None) -> dict[int, float]:
    path = Path(mmgbsa_path) if mmgbsa_path else default_mmgbsa_path(dataset_path)
    return _load_baseline_map(
        dataset_path=dataset_path,
        path=path,
        value_cols=["delta_g_kcal_mol", "mmgbsa_estimate", "Baseline_dG"],
    )

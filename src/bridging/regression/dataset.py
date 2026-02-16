from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridging.MD.paths import GENERATED_DIR
from bridging.ml.dataset import collect_feature_files
from bridging.utils.affinity import experimental_delta_g_kcalmol, split_name, to_float
from bridging.utils.dataset_rows import row_pdb_id, row_temperature_k
from bridging.utils.table import first_nonempty, normalized_lookup

from .config import DEFAULT_FEATURE_FILENAME, DEFAULT_MMGBSA_DIR, DEFAULT_PRODIGY_DIR


def _norm_lookup(row: dict) -> dict[str, str]:
    return normalized_lookup(row)


def _first_value(row: dict, lookup: dict[str, str], aliases: list[str]):
    return first_nonempty(row, lookup, aliases)


def _parse_pdb_id(row: dict) -> str | None:
    return row_pdb_id(row)


def _parse_temp_k(row: dict) -> float | None:
    return row_temperature_k(row)


def _parse_complex_id(row: dict) -> str | None:
    lookup = _norm_lookup(row)
    value = _first_value(row, lookup, ["complexid", "complexpdb"])
    if value is None:
        return None
    return str(value).strip()


def _parse_chain_value(row: dict, aliases: list[str]) -> str | None:
    lookup = _norm_lookup(row)
    value = _first_value(row, lookup, aliases)
    if value is None:
        return None
    return str(value).strip()


def _row_baseline_prodigy(row: dict) -> float | None:
    lookup = _norm_lookup(row)
    value = _first_value(
        row,
        lookup,
        [
            "baselinedg",
            "prodigyestimate",
            "prodigydeltag",
            "deltagkcalmolpred",
            "predicteddg",
        ],
    )
    return to_float(value)


def _norm_path_text(path_like) -> str:
    return str(path_like).replace("\\", "/").lower()


def default_prodigy_path(dataset_path: Path) -> Path:
    return DEFAULT_PRODIGY_DIR / f"{dataset_path.stem}_prodigy_estimates.csv"


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


def load_prodigy_map(dataset_path: Path, prodigy_path: str | Path | None = None) -> dict[int, float]:
    path = Path(prodigy_path) if prodigy_path else default_prodigy_path(dataset_path)
    return _load_baseline_map(
        dataset_path=dataset_path,
        path=path,
        value_cols=["delta_g_kcal_mol", "prodigy_estimate", "Baseline_dG"],
    )


def load_mmgbsa_map(dataset_path: Path, mmgbsa_path: str | Path | None = None) -> dict[int, float]:
    path = Path(mmgbsa_path) if mmgbsa_path else default_mmgbsa_path(dataset_path)
    return _load_baseline_map(
        dataset_path=dataset_path,
        path=path,
        value_cols=["delta_g_kcal_mol", "mmgbsa_estimate", "Baseline_dG"],
    )


def _feature_path_score(path: Path, dataset_stem: str | None) -> tuple[int, float]:
    text = str(path).replace("\\", "/").lower()
    score = 0
    if dataset_stem and dataset_stem.lower() in text:
        score += 10
    if "/md_datasets/" in text:
        score += 2
    return score, path.stat().st_mtime


def collect_feature_index(
    features: list[str] | None = None,
    *,
    feature_filename: str = DEFAULT_FEATURE_FILENAME,
    dataset_stem: str | None = None,
):
    paths = []
    if features:
        for item in features:
            paths.extend(collect_feature_files(item, filename=feature_filename))
    else:
        roots = [
            GENERATED_DIR / "MD",
            GENERATED_DIR / "MD_datasets",
        ]
        for root in roots:
            if not root.exists():
                continue
            paths.extend([str(p) for p in root.rglob(f"features/{feature_filename}")])

    index = {}
    duplicates = {}
    for raw in sorted(set(paths)):
        path = Path(raw)
        if not path.exists():
            continue
        if path.parent.name != "features":
            continue
        pdb_id = path.parent.parent.name.upper()
        if len(pdb_id) != 4:
            continue

        if pdb_id not in index:
            index[pdb_id] = path
            continue

        prev = index[pdb_id]
        duplicates[pdb_id] = duplicates.get(pdb_id, 1) + 1
        prev_score = _feature_path_score(prev, dataset_stem)
        new_score = _feature_path_score(path, dataset_stem)
        if new_score > prev_score:
            index[pdb_id] = path

    return index, duplicates


def build_records(
    dataset_path: str | Path,
    feature_index: dict[str, Path],
    prodigy_map: dict[int, float],
    mmgbsa_map: dict[int, float] | None = None,
):
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)
    mmgbsa_map = mmgbsa_map or {}
    rows = []
    for row_index, row in enumerate(df.to_dict("records")):
        pdb_id = _parse_pdb_id(row)
        split = split_name(row.get("split", "train"))
        experimental = experimental_delta_g_kcalmol(row)
        baseline = prodigy_map.get(row_index)
        if baseline is None:
            baseline = _row_baseline_prodigy(row)
        mmgbsa = mmgbsa_map.get(row_index)

        feature_path = feature_index.get((pdb_id or "").upper())
        rows.append(
            {
                "row_index": row_index,
                "dataset": str(dataset_path),
                "split": split,
                "pdb_id": pdb_id,
                "complex_id": _parse_complex_id(row),
                "ligand_chains": _parse_chain_value(row, ["ligandchains", "chains1"]),
                "receptor_chains": _parse_chain_value(row, ["receptorchains", "chains2"]),
                "temperature_k": _parse_temp_k(row),
                "experimental_delta_g": experimental,
                "prodigy_estimate": baseline,
                "mmgbsa_estimate": mmgbsa,
                "feature_path": str(feature_path) if feature_path else None,
                "feature_available": feature_path is not None,
                "experimental_available": experimental is not None,
                "prodigy_available": baseline is not None,
                "mmgbsa_available": mmgbsa is not None,
            }
        )
    return pd.DataFrame(rows)

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from bridging.MD.paths import GENERATED_DIR
from bridging.ml.dataset import collect_feature_files
from bridging.utils.dataset_rows import row_pdb_id
from bridging.utils.table import first_nonempty, normalize_column_name, normalized_lookup

from .config import DEFAULT_FEATURE_FILENAME, DEFAULT_PRODIGY_DIR

R_KCAL_PER_MOL_K = 0.00198720425864083


def _norm_col(name: str) -> str:
    return normalize_column_name(name)


def _norm_lookup(row: dict) -> dict[str, str]:
    return normalized_lookup(row)


def _first_value(row: dict, lookup: dict[str, str], aliases: list[str]):
    return first_nonempty(row, lookup, aliases)


def _to_float(value):
    if value is None:
        return None
    try:
        x = float(value)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def _parse_pdb_id(row: dict) -> str | None:
    return row_pdb_id(row)


def _parse_split(row: dict) -> str:
    lookup = _norm_lookup(row)
    value = _first_value(row, lookup, ["split"])
    if value is None:
        return "train"
    s = str(value).strip().lower()
    if s in {"test", "val", "valid", "validation"}:
        return "test"
    return "train"


def _parse_temp_k(row: dict) -> float | None:
    lookup = _norm_lookup(row)
    temp_k = _first_value(row, lookup, ["tempk", "temperaturek", "temperaturekelvin"])
    tk = _to_float(temp_k)
    if tk is not None:
        return tk
    temp_c = _first_value(row, lookup, ["tempc", "temperaturec", "temperaturecelsius"])
    tc = _to_float(temp_c)
    if tc is not None:
        return tc + 273.15
    return None


def _experimental_from_kd(row: dict) -> float | None:
    lookup = _norm_lookup(row)
    kd_val = _first_value(row, lookup, ["kdm", "kd"])
    kd = _to_float(kd_val)
    if kd is None or kd <= 0:
        return None
    temp_k = _parse_temp_k(row) or 298.15
    return R_KCAL_PER_MOL_K * float(temp_k) * math.log(kd)


def _parse_experimental_dg(row: dict) -> float | None:
    lookup = _norm_lookup(row)
    value = _first_value(
        row,
        lookup,
        [
            "deltagkcal",
            "experimentaldg",
            "dgkcalmol",
            "dgkcalmol",
            "bindingaffinity",
            "dGkcalmol",
        ],
    )
    out = _to_float(value)
    if out is not None:
        return out
    return _experimental_from_kd(row)


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
    return _to_float(value)


def _norm_path_text(path_like) -> str:
    return str(path_like).replace("\\", "/").lower()


def default_prodigy_path(dataset_path: Path) -> Path:
    return DEFAULT_PRODIGY_DIR / f"{dataset_path.stem}_prodigy_estimates.csv"


def load_prodigy_map(dataset_path: Path, prodigy_path: str | Path | None = None) -> dict[int, float]:
    path = Path(prodigy_path) if prodigy_path else default_prodigy_path(dataset_path)
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
    for col in ["delta_g_kcal_mol", "prodigy_estimate", "Baseline_dG"]:
        if col in df.columns:
            value_col = col
            break
    if value_col is None:
        return {}

    out = {}
    for row in df.to_dict("records"):
        idx = _to_float(row.get("row_index"))
        val = _to_float(row.get(value_col))
        if idx is None or val is None:
            continue
        out[int(idx)] = float(val)
    return out


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
            GENERATED_DIR / "MD_PPB_TCR_pMHC",
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


def build_records(dataset_path: str | Path, feature_index: dict[str, Path], prodigy_map: dict[int, float]):
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)
    rows = []
    for row_index, row in enumerate(df.to_dict("records")):
        pdb_id = _parse_pdb_id(row)
        split = _parse_split(row)
        experimental = _parse_experimental_dg(row)
        baseline = prodigy_map.get(row_index)
        if baseline is None:
            baseline = _row_baseline_prodigy(row)

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
                "feature_path": str(feature_path) if feature_path else None,
                "feature_available": feature_path is not None,
                "experimental_available": experimental is not None,
                "prodigy_available": baseline is not None,
            }
        )
    return pd.DataFrame(rows)

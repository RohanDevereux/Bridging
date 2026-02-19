from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bridging.utils.dataset_rows import parse_chain_group, row_chain_groups, row_pdb_id, row_temperature_k
from bridging.utils.table import normalize_column_name


def _lookup(columns: list[str]) -> dict[str, str]:
    return {normalize_column_name(c): c for c in columns}


def _first_col(lookup: dict[str, str], aliases: list[str]) -> str | None:
    for alias in aliases:
        col = lookup.get(alias)
        if col is not None:
            return col
    return None


def _norm_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).strip().lower().split())


def _is_wildtype_mutation(value) -> bool:
    text = _norm_text(value)
    if text in {"", "nan", "none", "na"}:
        return True
    return text in {"wt", "wildtype", "wild-type"}


def _norm_chain_group(value) -> str:
    return ",".join(parse_chain_group(value))


def _pair_key(
    row: dict,
    ligand_name_col: str | None,
    receptor_name_col: str | None,
) -> str | None:
    lig_name = _norm_text(row.get(ligand_name_col)) if ligand_name_col else ""
    rec_name = _norm_text(row.get(receptor_name_col)) if receptor_name_col else ""
    if not lig_name or not rec_name:
        return None
    return f"{lig_name}||{rec_name}"


def _parse_size(size_text: str | None, total: int) -> int:
    if size_text is None:
        return -1
    s = str(size_text).strip()
    if not s:
        return -1
    if "." in s:
        frac = float(s)
        if not (0.0 < frac < 1.0):
            raise ValueError(f"Fraction must be in (0,1). Got {frac}.")
        return int(round(total * frac))
    return int(s)


def _resolve_split_targets(total: int, train_size: str | None, test_size: str | None) -> tuple[int, int]:
    train_target = _parse_size(train_size, total)
    test_target = _parse_size(test_size, total)

    if train_target < 0 and test_target < 0:
        return -1, -1
    if train_target < 0:
        train_target = total - test_target
    if test_target < 0:
        test_target = total - train_target

    if train_target < 0 or test_target < 0:
        raise ValueError("Resolved negative split targets.")
    if train_target + test_target > total:
        raise ValueError(f"train_size + test_size exceeds n={total}.")

    remainder = total - (train_target + test_target)
    if remainder > 0:
        train_target += remainder
    return int(train_target), int(test_target)


def _split_random(df: pd.DataFrame, train_target: int, test_target: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_target < 0 and test_target < 0:
        return df.copy(), df.iloc[0:0].copy()
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(df))
    test_idx = idx[:test_target]
    train_idx = idx[test_target : test_target + train_target]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _split_stratified(
    df: pd.DataFrame,
    train_target: int,
    test_target: int,
    seed: int,
    stratify_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stratify_col not in df.columns:
        raise KeyError(f"stratify column not found: {stratify_col}")
    if train_target < 0 and test_target < 0:
        return df.copy(), df.iloc[0:0].copy()

    values = df[stratify_col].fillna("NA").astype(str)
    groups: dict[str, list[int]] = {}
    for idx, value in enumerate(values):
        groups.setdefault(value, []).append(idx)

    total = len(df)
    desired: dict[str, int] = {}
    frac_parts: list[tuple[float, str]] = []
    assigned = 0
    for key, indices in groups.items():
        n = len(indices)
        raw = n * (test_target / total)
        base = int(math.floor(raw))
        base = min(base, n)
        desired[key] = base
        assigned += base
        frac_parts.append((raw - base, key))

    remaining = test_target - assigned
    if remaining > 0:
        frac_parts.sort(reverse=True)
        for _, key in frac_parts:
            if remaining <= 0:
                break
            cap = len(groups[key]) - desired[key]
            if cap <= 0:
                continue
            desired[key] += 1
            remaining -= 1

    rng = np.random.default_rng(int(seed))
    train_idx: list[int] = []
    test_idx: list[int] = []
    for key, indices in groups.items():
        perm = np.array(indices, dtype=int)
        rng.shuffle(perm)
        n_test = desired.get(key, 0)
        test_idx.extend(perm[:n_test].tolist())
        train_idx.extend(perm[n_test:].tolist())
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a broad PPB-style dataset: keep all subgroups, keep rows even without temperature, "
            "drop mutants, enforce unique ligand/receptor pairs and unique PDB IDs."
        )
    )
    parser.add_argument("--in-csv", required=True, help="Input CSV (e.g., rawData/PPB-Affinity.csv)")
    parser.add_argument("--out-csv", required=True, help="Filtered output CSV")
    parser.add_argument("--report-csv", required=True, help="Row-level keep/drop report CSV")
    parser.add_argument("--train-csv", default="", help="Optional train split CSV")
    parser.add_argument("--test-csv", default="", help="Optional test split CSV")
    parser.add_argument("--train-size", default="0.8", help="Train split size (count or fraction)")
    parser.add_argument("--test-size", default="0.2", help="Test split size (count or fraction)")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--stratify-by", default="Subgroup", help="Split stratification column; empty disables")
    parser.add_argument("--col-mutations", default="Mutations")
    parser.add_argument("--col-ligand-name", default="Ligand Name")
    parser.add_argument("--col-receptor-name", default="Receptor Name")
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    report_csv = Path(args.report_csv)
    train_csv = Path(args.train_csv) if args.train_csv else None
    test_csv = Path(args.test_csv) if args.test_csv else None

    df = pd.read_csv(in_csv).copy()
    cols = _lookup(list(df.columns))

    mut_col = _first_col(cols, [normalize_column_name(args.col_mutations)])
    lig_name_col = _first_col(cols, [normalize_column_name(args.col_ligand_name)])
    rec_name_col = _first_col(cols, [normalize_column_name(args.col_receptor_name)])

    records = df.to_dict("records")
    for row in records:
        row["_pdb_id"] = row_pdb_id(row)
        chains_1, chains_2 = row_chain_groups(row)
        row["_chains_1"] = chains_1
        row["_chains_2"] = chains_2
        row["_pair_key"] = _pair_key(row, lig_name_col, rec_name_col)
        row["_temp_k"] = row_temperature_k(row)
        row["_is_wt"] = _is_wildtype_mutation(row.get(mut_col)) if mut_col else True
        row["_has_temp"] = 1 if row["_temp_k"] is not None else 0

    # Prefer rows with reported temperature when resolving duplicates.
    rows_df = pd.DataFrame(records).sort_values(["_has_temp"], ascending=False, kind="mergesort")

    report_rows: list[dict] = []
    seen_pdb: set[str] = set()
    seen_pair: set[str] = set()
    kept_rows: list[dict] = []

    for row in rows_df.to_dict("records"):
        pdb_id = row.get("_pdb_id")
        pair_key = str(row.get("_pair_key", ""))

        if not row.get("_is_wt", False):
            report_rows.append({"pdb": pdb_id, "reason": "mutant_dropped"})
            continue
        if not pdb_id:
            report_rows.append({"pdb": "", "reason": "missing_or_invalid_pdb"})
            continue

        c1 = parse_chain_group(row.get("_chains_1"))
        c2 = parse_chain_group(row.get("_chains_2"))
        if not c1 or not c2:
            report_rows.append({"pdb": pdb_id, "reason": "missing_chain_groups"})
            continue

        if pair_key and pair_key in seen_pair:
            report_rows.append({"pdb": pdb_id, "reason": "duplicate_ligand_receptor_pair"})
            continue
        if pdb_id in seen_pdb:
            report_rows.append({"pdb": pdb_id, "reason": "duplicate_pdb"})
            continue

        if pair_key:
            seen_pair.add(pair_key)
        seen_pdb.add(pdb_id)
        out_row = dict(row)
        out_row["PDB"] = pdb_id
        out_row["Temp_K"] = row.get("_temp_k")
        kept_rows.append(out_row)
        report_rows.append({"pdb": pdb_id, "reason": "KEEP"})

    out_df = pd.DataFrame(kept_rows)
    drop_cols = [c for c in out_df.columns if c.startswith("_")]
    if drop_cols:
        out_df = out_df.drop(columns=drop_cols)

    if out_df.empty:
        raise RuntimeError("No rows left after filtering.")

    train_target, test_target = _resolve_split_targets(len(out_df), args.train_size, args.test_size)
    if train_target >= 0 or test_target >= 0:
        stratify_col = str(args.stratify_by).strip()
        if stratify_col and stratify_col in out_df.columns:
            train_df, test_df = _split_stratified(out_df, train_target, test_target, args.split_seed, stratify_col)
        else:
            train_df, test_df = _split_random(out_df, train_target, test_target, args.split_seed)

        test_ids = set(test_df["PDB"].astype(str).str.upper())
        out_df["split"] = out_df["PDB"].astype(str).str.upper().map(lambda x: "test" if x in test_ids else "train")
        if train_csv is not None:
            train_csv.parent.mkdir(parents=True, exist_ok=True)
            out_df[out_df["split"] == "train"].to_csv(train_csv, index=False)
        if test_csv is not None:
            test_csv.parent.mkdir(parents=True, exist_ok=True)
            out_df[out_df["split"] == "test"].to_csv(test_csv, index=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_csv, index=False)

    print(
        f"[DONE] input={len(df)} kept={len(out_df)} "
        f"unique_pdb={out_df['PDB'].nunique()} unique_pair={len(seen_pair)} out={out_csv}"
    )
    print(f"[REPORT] {report_csv}")
    if "split" in out_df.columns:
        print(f"[SPLIT] {out_df['split'].value_counts().to_dict()}")
    if train_csv:
        print(f"[TRAIN] {train_csv}")
    if test_csv:
        print(f"[TEST] {test_csv}")


if __name__ == "__main__":
    main()

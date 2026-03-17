from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from bridging.utils.dataset_rows import parse_chain_group, row_temperature_k


def _norm_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _norm_subgroup(value) -> str:
    text = _norm_text(value)
    if text in {"", "nan", "none"}:
        return "OTHER"
    return str(value).strip()


def _chain_key(value) -> str:
    return ",".join(parse_chain_group(value))


def _parse_size(size_text: str | None, total: int) -> int:
    if size_text is None:
        return -1
    s = str(size_text).strip()
    if not s:
        return -1
    if re.match(r"^\d+(\.\d+)?$", s) is None:
        raise ValueError(f"Bad size '{size_text}'. Use integer (e.g. 100) or fraction (e.g. 0.8).")
    if "." in s:
        frac = float(s)
        if not (0.0 < frac < 1.0):
            raise ValueError(f"Fraction size must be in (0,1). Got {frac}.")
        return int(round(frac * total))
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
        raise ValueError("Resolved negative split sizes. Check train-size/test-size.")
    if train_target + test_target > total:
        raise ValueError(
            f"train-size + test-size = {train_target + test_target} exceeds n={total}"
        )

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
    if remaining > 0:
        raise RuntimeError("Could not allocate stratified test set sizes.")

    rng = np.random.default_rng(int(seed))
    test_indices = []
    train_indices = []
    for key, indices in groups.items():
        perm = np.array(indices, dtype=int)
        rng.shuffle(perm)
        n_test = desired.get(key, 0)
        test_indices.extend(perm[:n_test].tolist())
        train_indices.extend(perm[n_test:].tolist())

    train = df.iloc[np.array(train_indices, dtype=int)].copy()
    test = df.iloc[np.array(test_indices, dtype=int)].copy()
    return train, test


def _dedup_by_pdb(df: pd.DataFrame, col_pdb: str) -> tuple[pd.DataFrame, int]:
    if col_pdb not in df.columns:
        raise KeyError(f"Missing PDB column: {col_pdb}")
    out = df.copy()
    out[col_pdb] = out[col_pdb].astype(str).str.strip().str.upper()
    before = len(out)
    out = out.drop_duplicates(subset=[col_pdb], keep="first").copy()
    return out, before - len(out)


def _dedup_exact_pairs(
    df: pd.DataFrame,
    *,
    col_ligand_name: str,
    col_receptor_name: str,
) -> tuple[pd.DataFrame, int]:
    if col_ligand_name not in df.columns or col_receptor_name not in df.columns:
        raise KeyError(
            f"Pair dedup requires columns: {col_ligand_name!r}, {col_receptor_name!r}"
        )
    out = df.copy()
    pair_key = out[col_ligand_name].map(_norm_text) + "||" + out[col_receptor_name].map(_norm_text)
    before = len(out)
    out = out.loc[~pair_key.duplicated(keep="first")].copy()
    return out, before - len(out)


def _build_feature_table(
    df: pd.DataFrame,
    col_ligand_name: str,
    col_receptor_name: str,
    col_ligand_chains: str,
    col_receptor_chains: str,
    col_subgroup: str,
) -> pd.DataFrame:
    out = df.copy()

    empty = pd.Series([""] * len(out), index=out.index)
    lig_name = out[col_ligand_name] if col_ligand_name in out.columns else empty
    rec_name = out[col_receptor_name] if col_receptor_name in out.columns else empty
    lig_chain = out[col_ligand_chains] if col_ligand_chains in out.columns else empty
    rec_chain = out[col_receptor_chains] if col_receptor_chains in out.columns else empty

    out["_ligand_key"] = lig_name.map(_norm_text)
    out["_receptor_key"] = rec_name.map(_norm_text)
    out["_ligand_chain_key"] = lig_chain.map(_chain_key)
    out["_receptor_chain_key"] = rec_chain.map(_chain_key)

    # fall back to chain identity when protein name is blank
    out["_ligand_key"] = np.where(
        out["_ligand_key"] == "",
        out["_ligand_chain_key"],
        out["_ligand_key"],
    )
    out["_receptor_key"] = np.where(
        out["_receptor_key"] == "",
        out["_receptor_chain_key"],
        out["_receptor_key"],
    )

    out["_pair_key"] = out["_ligand_key"] + "||" + out["_receptor_key"]
    out["_chain_pair_key"] = out["_ligand_chain_key"] + "||" + out["_receptor_chain_key"]
    out["_subgroup_key"] = (
        out[col_subgroup].map(_norm_subgroup) if col_subgroup in out.columns else "OTHER"
    )
    return out


def _select_diverse_subset(
    df: pd.DataFrame,
    target_size: int,
    seed: int,
    *,
    w_ligand: float,
    w_receptor: float,
    w_pair: float,
    w_chain: float,
    w_subgroup: float,
    rarity_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_size <= 0:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()
    if target_size >= len(df):
        selected = df.copy()
        selected["selection_rank"] = np.arange(1, len(selected) + 1)
        selected["diversity_score"] = np.nan
        return selected, selected.copy()

    feature_cols = {
        "ligand": "_ligand_key",
        "receptor": "_receptor_key",
        "pair": "_pair_key",
        "chain": "_chain_pair_key",
        "subgroup": "_subgroup_key",
    }
    weights = {
        "ligand": float(w_ligand),
        "receptor": float(w_receptor),
        "pair": float(w_pair),
        "chain": float(w_chain),
        "subgroup": float(w_subgroup),
    }

    pool_counts = {
        name: Counter(df[col].tolist()) for name, col in feature_cols.items()
    }
    selected_counts = {
        name: defaultdict(int) for name in feature_cols
    }

    rng = np.random.default_rng(int(seed))
    remaining = list(range(len(df)))
    selected_indices: list[int] = []
    selected_scores: list[float] = []

    for _ in range(target_size):
        best_score = None
        best_rows: list[int] = []

        for idx in remaining:
            score = 0.0
            for feat_name, feat_col in feature_cols.items():
                key = df.iloc[idx][feat_col]
                weight = weights[feat_name]
                if weight == 0.0:
                    continue
                score += weight / (1.0 + selected_counts[feat_name][key])
                if rarity_weight > 0.0:
                    score += float(rarity_weight) * weight / float(pool_counts[feat_name][key])

            if best_score is None or score > best_score + 1e-12:
                best_score = score
                best_rows = [idx]
            elif abs(score - best_score) <= 1e-12:
                best_rows.append(idx)

        chosen = best_rows[int(rng.integers(len(best_rows)))]
        selected_indices.append(chosen)
        selected_scores.append(float(best_score))
        remaining.remove(chosen)

        for feat_name, feat_col in feature_cols.items():
            key = df.iloc[chosen][feat_col]
            selected_counts[feat_name][key] += 1

    selected = df.iloc[selected_indices].copy()
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["diversity_score"] = np.array(selected_scores, dtype=float)
    report = selected.copy()
    return selected, report


def main():
    ap = argparse.ArgumentParser(
        description="Select a max-diversity subset of PPB rows with a simple greedy novelty score."
    )
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--report-csv", default="", help="Optional report with selection rank and score")
    ap.add_argument(
        "--target-size",
        required=True,
        help="Count (e.g. 100) or fraction of input (e.g. 0.7).",
    )
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train-csv", default="", help="Optional train split file")
    ap.add_argument("--test-csv", default="", help="Optional test split file")
    ap.add_argument("--train-size", default="0.8", help="Split size: count or fraction")
    ap.add_argument("--test-size", default="0.2", help="Split size: count or fraction")
    ap.add_argument("--stratify-by", default="", help="Optional split stratification column")

    ap.add_argument("--require-temperature", action="store_true")
    ap.add_argument("--temp-min-k", type=float, default=None)
    ap.add_argument("--temp-max-k", type=float, default=None)

    ap.add_argument("--col-pdb", default="PDB")
    ap.add_argument("--col-ligand-name", default="Ligand Name")
    ap.add_argument("--col-receptor-name", default="Receptor Name")
    ap.add_argument("--col-ligand-chains", default="Ligand Chains")
    ap.add_argument("--col-receptor-chains", default="Receptor Chains")
    ap.add_argument("--col-subgroup", default="Subgroup")
    ap.add_argument(
        "--dedup-exact-pairs",
        action="store_true",
        help="Keep only first row per exact (Ligand Name, Receptor Name) pair.",
    )

    ap.add_argument("--w-ligand", type=float, default=1.0)
    ap.add_argument("--w-receptor", type=float, default=1.0)
    ap.add_argument("--w-pair", type=float, default=2.0)
    ap.add_argument("--w-chain", type=float, default=1.0)
    ap.add_argument("--w-subgroup", type=float, default=0.5)
    ap.add_argument("--rarity-weight", type=float, default=0.25)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    report_csv = Path(args.report_csv) if args.report_csv else None
    train_csv = Path(args.train_csv) if args.train_csv else None
    test_csv = Path(args.test_csv) if args.test_csv else None

    df = pd.read_csv(in_csv)
    if args.col_pdb not in df.columns:
        raise KeyError(f"Missing PDB column: {args.col_pdb}")

    # keep first row per PDB so scoring is over unique complexes
    df, dropped_dupes = _dedup_by_pdb(df, args.col_pdb)

    if args.require_temperature or args.temp_min_k is not None or args.temp_max_k is not None:
        keep_mask = []
        for row in df.to_dict("records"):
            t_k = row_temperature_k(row)
            keep = True
            if args.require_temperature and t_k is None:
                keep = False
            if keep and args.temp_min_k is not None and (t_k is None or t_k < args.temp_min_k):
                keep = False
            if keep and args.temp_max_k is not None and (t_k is None or t_k > args.temp_max_k):
                keep = False
            keep_mask.append(keep)
        df = df.loc[np.array(keep_mask, dtype=bool)].copy()

    dropped_pair_dupes = 0
    if args.dedup_exact_pairs:
        df, dropped_pair_dupes = _dedup_exact_pairs(
            df,
            col_ligand_name=args.col_ligand_name,
            col_receptor_name=args.col_receptor_name,
        )

    df = _build_feature_table(
        df,
        args.col_ligand_name,
        args.col_receptor_name,
        args.col_ligand_chains,
        args.col_receptor_chains,
        args.col_subgroup,
    )

    target_size = _parse_size(args.target_size, len(df))
    if target_size <= 0:
        raise ValueError(f"target-size resolved to {target_size}; must be >=1")
    target_size = min(target_size, len(df))

    selected_df, report_df = _select_diverse_subset(
        df,
        target_size,
        args.seed,
        w_ligand=args.w_ligand,
        w_receptor=args.w_receptor,
        w_pair=args.w_pair,
        w_chain=args.w_chain,
        w_subgroup=args.w_subgroup,
        rarity_weight=args.rarity_weight,
    )

    helper_cols = [
        "_ligand_key",
        "_receptor_key",
        "_pair_key",
        "_ligand_chain_key",
        "_receptor_chain_key",
        "_chain_pair_key",
        "_subgroup_key",
    ]

    out_cols = [c for c in selected_df.columns if c not in helper_cols]
    selected_out = selected_df[out_cols].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_out.to_csv(out_csv, index=False)

    if report_csv is not None:
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_csv, index=False)

    if train_csv is not None or test_csv is not None:
        train_target, test_target = _resolve_split_targets(
            len(selected_out), args.train_size, args.test_size
        )
        if args.stratify_by.strip():
            train_df, test_df = _split_stratified(
                selected_out, train_target, test_target, args.seed, args.stratify_by.strip()
            )
        else:
            train_df, test_df = _split_random(selected_out, train_target, test_target, args.seed)

        test_ids = set(test_df[args.col_pdb].astype(str).str.upper())
        selected_out["split"] = selected_out[args.col_pdb].astype(str).str.upper().map(
            lambda x: "test" if x in test_ids else "train"
        )

        if train_csv is not None:
            train_csv.parent.mkdir(parents=True, exist_ok=True)
            selected_out[selected_out["split"] == "train"].to_csv(train_csv, index=False)
        if test_csv is not None:
            test_csv.parent.mkdir(parents=True, exist_ok=True)
            selected_out[selected_out["split"] == "test"].to_csv(test_csv, index=False)

        selected_out.to_csv(out_csv, index=False)

    subgroup_counts = {}
    if args.col_subgroup in selected_out.columns:
        subgroup_counts = selected_out[args.col_subgroup].fillna("OTHER").astype(str).value_counts().to_dict()

    print(
        f"[DIVERSE] input={len(pd.read_csv(in_csv))} deduped={len(df)} "
        f"dropped_pdb_dupes={dropped_dupes} dropped_exact_pair_dupes={dropped_pair_dupes} "
        f"selected={len(selected_out)} out={out_csv}"
    )
    if subgroup_counts:
        print(f"[DIVERSE] subgroup_counts={subgroup_counts}")
    if report_csv is not None:
        print(f"[REPORT] {report_csv}")
    if train_csv is not None:
        print(f"[TRAIN] {train_csv}")
    if test_csv is not None:
        print(f"[TEST] {test_csv}")


if __name__ == "__main__":
    main()

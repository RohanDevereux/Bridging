from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rcsbapi.data import DataQuery as Query

    RCSB_OK = True
except Exception:
    Query = None
    RCSB_OK = False


def _parse_chain_group(chain_group: str) -> list[str]:
    if chain_group is None:
        return []
    s = str(chain_group).strip()
    if not s or s.lower() == "nan":
        return []
    s = s.replace(";", ",").replace("/", ",")
    s = "".join(s.split())
    return [c for c in s.split(",") if c]


def _normalize_subgroup(x) -> str:
    if x is None:
        return "OTHER"
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return "OTHER"
    return s


def _parse_size(size_text: str | None, total: int) -> int:
    if size_text is None:
        return -1
    s = str(size_text).strip()
    if not s:
        return -1
    if re.match(r"^\d+(\.\d+)?$", s) is None:
        raise ValueError(f"Bad size '{size_text}'. Use integer (e.g. 200) or fraction (e.g. 0.8).")
    if "." in s:
        frac = float(s)
        if not (0.0 < frac < 1.0):
            raise ValueError(f"Fraction size must be in (0,1). Got {frac}.")
        return int(round(frac * total))
    return int(s)


def _dedup_by_pdb(df: pd.DataFrame, col_pdb: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if col_pdb not in df.columns:
        raise KeyError(f"Missing PDB column: {col_pdb}")
    df2 = df.copy()
    df2[col_pdb] = df2[col_pdb].astype(str).str.strip().str.upper()
    dup_mask = df2.duplicated(subset=[col_pdb], keep="first")
    dup_report = df2.loc[dup_mask].copy()
    dup_report["reason"] = "DUPLICATE_PDB_DROPPED"
    deduped = df2.loc[~dup_mask].copy()
    return deduped, dup_report


def _is_valid_pdb_id(pdb_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{4}", str(pdb_id).strip().upper()))


def _normalize_chain_key(chain_group: str) -> str:
    return ",".join(_parse_chain_group(chain_group))


def _size_key(pdb_id: str, ligand_chains: str, receptor_chains: str) -> str:
    pdb = str(pdb_id).strip().upper()
    return f"{pdb}|{_normalize_chain_key(ligand_chains)}|{_normalize_chain_key(receptor_chains)}"


def _to_int_or_none(value) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        out = int(round(float(value)))
        if out < 0:
            return None
        return out
    except Exception:
        return None


def _load_sizes_cache(path_like: str | Path | None) -> dict[str, dict]:
    if not path_like:
        return {}
    path = Path(path_like)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    out: dict[str, dict] = {}
    for row in df.to_dict("records"):
        key = row.get("size_key", None)
        if key is None or str(key).strip() == "":
            pdb = row.get("pdb_id", row.get("pdb", None))
            lig = row.get("ligand_chains", row.get("chains_1", ""))
            rec = row.get("receptor_chains", row.get("chains_2", ""))
            if pdb is None:
                continue
            key = _size_key(str(pdb), str(lig), str(rec))
        out[str(key)] = row
    return out


def _select_partner_lengths(size_row: dict | None, size_source: str) -> tuple[int | None, int | None, str | None]:
    if not size_row:
        return None, None, None

    seq_lig = _to_int_or_none(size_row.get("ligand_seq_len", size_row.get("ligand_total_residues", None)))
    seq_rec = _to_int_or_none(size_row.get("receptor_seq_len", size_row.get("receptor_total_residues", None)))
    mdl_lig = _to_int_or_none(size_row.get("ligand_modeled_residues", size_row.get("ligand_len", None)))
    mdl_rec = _to_int_or_none(size_row.get("receptor_modeled_residues", size_row.get("receptor_len", None)))

    if size_source == "rcsb":
        if seq_lig is not None and seq_rec is not None:
            return seq_lig, seq_rec, "rcsb_cache"
        return None, None, None

    if size_source == "modeled":
        if mdl_lig is not None and mdl_rec is not None:
            return mdl_lig, mdl_rec, "modeled_cache"
        return None, None, None

    # auto
    if seq_lig is not None and seq_rec is not None:
        return seq_lig, seq_rec, "rcsb_cache"
    if mdl_lig is not None and mdl_rec is not None:
        return mdl_lig, mdl_rec, "modeled_cache"
    return None, None, None


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
        raise ValueError("Resolved negative split sizes. Check train_size/test_size.")
    if train_target + test_target > total:
        raise ValueError(
            f"train_size + test_size = {train_target + test_target} exceeds n={total}"
        )

    remainder = total - (train_target + test_target)
    if remainder > 0:
        train_target += remainder

    return int(train_target), int(test_target)


def _random_split(
    df: pd.DataFrame, train_target: int, test_target: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_target < 0 and test_target < 0:
        return df.copy(), df.iloc[0:0].copy()

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(df))
    test_idx = idx[:test_target]
    train_idx = idx[test_target : test_target + train_target]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _stratified_split(
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
    groups = {}
    for idx, value in enumerate(values):
        groups.setdefault(value, []).append(idx)

    rng = np.random.default_rng(int(seed))
    total = len(df)
    if total == 0:
        return df.copy(), df.iloc[0:0].copy()

    desired = {}
    frac_parts = []
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

    test_indices = []
    train_indices = []
    for key, indices in groups.items():
        perm = np.array(indices, dtype=int)
        rng.shuffle(perm)
        n_test = desired.get(key, 0)
        test_indices.extend(perm[:n_test].tolist())
        train_indices.extend(perm[n_test:].tolist())

    # train gets all remaining rows by design
    if len(test_indices) != test_target:
        raise RuntimeError(
            f"Internal stratified split mismatch: expected test={test_target}, got {len(test_indices)}"
        )

    train = df.iloc[np.array(train_indices, dtype=int)].copy()
    test = df.iloc[np.array(test_indices, dtype=int)].copy()
    return train, test


def _stratified_sample(
    df: pd.DataFrame,
    target_size: int,
    seed: int,
    stratify_col: str,
) -> pd.DataFrame:
    if stratify_col not in df.columns:
        raise KeyError(f"stratify column not found: {stratify_col}")
    if target_size >= len(df):
        return df.copy()
    if target_size <= 0:
        return df.iloc[0:0].copy()

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
        raw = n * (target_size / total)
        base = int(math.floor(raw))
        base = min(base, n)
        desired[key] = base
        assigned += base
        frac_parts.append((raw - base, key))

    remaining = target_size - assigned
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
        raise RuntimeError("Could not allocate stratified sample sizes.")

    rng = np.random.default_rng(int(seed))
    keep_indices: list[int] = []
    for key, indices in groups.items():
        perm = np.array(indices, dtype=int)
        rng.shuffle(perm)
        keep_indices.extend(perm[: desired.get(key, 0)].tolist())

    if len(keep_indices) != target_size:
        raise RuntimeError(
            f"Internal stratified sample mismatch: expected {target_size}, got {len(keep_indices)}"
        )
    return df.iloc[np.array(keep_indices, dtype=int)].copy()


def _load_ca_coords_by_chain(pdb_path: Path) -> dict[str, np.ndarray]:
    by_chain: dict[str, list[np.ndarray]] = {}
    seen: set[tuple[str, str, str]] = set()

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue

            chain = line[21].strip() or " "
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)

            try:
                x = float(line[30:38]) * 0.1
                y = float(line[38:46]) * 0.1
                z = float(line[46:54]) * 0.1
            except Exception:
                continue
            by_chain.setdefault(chain, []).append(np.array([x, y, z], dtype=np.float32))

    return {k: np.stack(v, axis=0) for k, v in by_chain.items() if v}


def _nth_min_interface_distance_nm(X: np.ndarray, Y: np.ndarray, n: int) -> float:
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return float("inf")

    diff = X[:, None, :] - Y[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    mins = np.sqrt(d2.min(axis=1))
    mins.sort()
    if mins.shape[0] < n:
        return float("inf")
    return float(mins[n - 1])


def partner_ca_stats(
    pdb_path: Path,
    ligand_chains: str,
    receptor_chains: str,
    n_interface: int,
) -> dict[str, float]:
    coords = _load_ca_coords_by_chain(pdb_path)
    lig = _parse_chain_group(ligand_chains)
    rec = _parse_chain_group(receptor_chains)

    X = (
        np.concatenate([coords.get(c, np.zeros((0, 3), dtype=np.float32)) for c in lig], axis=0)
        if lig
        else np.zeros((0, 3), dtype=np.float32)
    )
    Y = (
        np.concatenate([coords.get(c, np.zeros((0, 3), dtype=np.float32)) for c in rec], axis=0)
        if rec
        else np.zeros((0, 3), dtype=np.float32)
    )

    return {
        "n_ca_ligand": float(X.shape[0]),
        "n_ca_receptor": float(Y.shape[0]),
        "dN_ligand_nm": _nth_min_interface_distance_nm(X, Y, n_interface),
        "dN_receptor_nm": _nth_min_interface_distance_nm(Y, X, n_interface),
    }


def fetch_entry_polymer_entity_lengths(pdb_id: str):
    if not RCSB_OK:
        raise RuntimeError(
            "py-rcsb-api not available; install it or pass --no-rcsb / --min-partner-len 0."
        )
    pdb_id = pdb_id.strip().upper()
    q = Query(
        input_type="entries",
        input_ids=[pdb_id],
        return_data_list=[
            "polymer_entities.entity_poly.rcsb_sample_sequence_length",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.auth_asym_ids",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.entity_id",
        ],
    )
    result = q.exec()
    entries = result.get("data", {}).get("entries", [])
    if not entries:
        raise ValueError(f"No entry returned for pdb_id={pdb_id}")
    return entries[0].get("polymer_entities", []) or []


def partner_lengths_from_chain_groups_rcsb(
    pdb_id: str,
    ligand_chains: str,
    receptor_chains: str,
    cache: dict[str, dict[str, int]],
) -> dict[str, int | None]:
    pdb_id = pdb_id.strip().upper()
    cache_key = f"{pdb_id}|{ligand_chains}|{receptor_chains}"
    if cache_key in cache:
        return cache[cache_key]

    entities = fetch_entry_polymer_entity_lengths(pdb_id)
    chain_to_len: dict[str, int] = {}
    for pe in entities:
        seq_len = (pe.get("entity_poly", {}) or {}).get("rcsb_sample_sequence_length", None)
        if seq_len is None:
            continue
        try:
            seq_len = int(seq_len)
        except Exception:
            continue
        cont = pe.get("rcsb_polymer_entity_container_identifiers", {}) or {}
        auth_asym_ids = cont.get("auth_asym_ids", []) or []
        for ch in auth_asym_ids:
            chain_to_len[str(ch)] = seq_len

    lig = _parse_chain_group(ligand_chains)
    rec = _parse_chain_group(receptor_chains)

    def _sum_len(chains: list[str]) -> int:
        total = 0
        for ch in chains:
            if ch not in chain_to_len:
                raise KeyError(
                    f"Chain '{ch}' not found for {pdb_id}. "
                    f"Available chains: {sorted(chain_to_len.keys())[:20]}"
                )
            total += chain_to_len[ch]
        return total

    out = {
        "ligand_total_residues": _sum_len(lig) if lig else None,
        "receptor_total_residues": _sum_len(rec) if rec else None,
    }
    cache[cache_key] = out
    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Filter PPB-Affinity-style data by subgroup, size, and interface support; "
            "deduplicate by PDB; and optionally create train/test splits."
        )
    )
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True, help="Filtered dataset (with split column if split requested)")
    ap.add_argument("--report-csv", required=True, help="Report with reasons (KEEP / dropped)")
    ap.add_argument("--train-csv", default="", help="Optional train split CSV")
    ap.add_argument("--test-csv", default="", help="Optional test split CSV")
    ap.add_argument(
        "--target-total-size",
        default="",
        help="Optional cap on kept dataset size before split. Count (e.g. 100) or fraction (e.g. 0.5).",
    )
    ap.add_argument("--train-size", default="", help="Count (e.g. 200) or fraction (e.g. 0.8)")
    ap.add_argument("--test-size", default="", help="Count (e.g. 50) or fraction (e.g. 0.2)")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--stratify-by", default="", help="Optional split stratification column, e.g. Subgroup")

    ap.add_argument("--pdb-root", required=True, help="Directory containing <PDB>.pdb files")
    ap.add_argument("--n-interface", type=int, default=64)
    ap.add_argument(
        "--max-dN-nm",
        type=float,
        default=1.0,
        help="Require Nth-closest CA distance <= this (nm). Use 'inf' to disable.",
    )
    ap.add_argument(
        "--min-partner-len",
        type=int,
        default=0,
        help="RCSB total residue count filter per partner; 0 disables.",
    )
    ap.add_argument("--keep-subgroups", default="*", help="Comma list (e.g. OTHER,TCR-pMHC) or *")
    ap.add_argument("--no-rcsb", action="store_true", help="Disable RCSB partner length checks")
    ap.add_argument("--rcsb-cache-json", default="rcsb_lengths_cache.json")
    ap.add_argument("--sizes-cache-csv", default="", help="Optional prefetched partner sizes CSV")
    ap.add_argument(
        "--size-source",
        choices=["auto", "rcsb", "modeled"],
        default="auto",
        help="Which partner length source to use for --min-partner-len threshold.",
    )
    ap.add_argument("--limit", type=int, help="Optional row limit for debugging")

    ap.add_argument("--col-pdb", default="PDB")
    ap.add_argument("--col-ligand", default="Ligand Chains")
    ap.add_argument("--col-receptor", default="Receptor Chains")
    ap.add_argument("--col-subgroup", default="Subgroup")
    ap.add_argument("--col-pdb-path", default="pdb_path")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    report_csv = Path(args.report_csv)
    train_csv = Path(args.train_csv) if args.train_csv else None
    test_csv = Path(args.test_csv) if args.test_csv else None
    pdb_root = Path(args.pdb_root)

    df0 = pd.read_csv(in_csv)
    if args.limit is not None:
        df0 = df0.head(int(args.limit)).copy()
    if args.col_pdb not in df0.columns:
        raise KeyError(f"Missing PDB column: {args.col_pdb}")
    df0[args.col_pdb] = df0[args.col_pdb].astype(str).str.strip().str.upper()

    keep_set = None
    if str(args.keep_subgroups).strip() != "*":
        keep_set = {s.strip() for s in str(args.keep_subgroups).split(",") if s.strip()}

    cache: dict[str, dict[str, int]] = {}
    cache_path = Path(args.rcsb_cache_json)
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    sizes_cache = _load_sizes_cache(args.sizes_cache_csv)
    if args.sizes_cache_csv:
        print(f"[FILTER] loaded size cache entries={len(sizes_cache)} from {args.sizes_cache_csv}")
    can_query_rcsb = (not args.no_rcsb) and RCSB_OK
    if args.min_partner_len > 0 and (not args.no_rcsb) and (not RCSB_OK):
        if len(sizes_cache) == 0:
            raise RuntimeError(
                "min-partner-len > 0 requested but py-rcsb-api is not available and no --sizes-cache-csv was provided. "
                "Install dependencies (`pip install -e .`) or rerun with --no-rcsb / --sizes-cache-csv / --min-partner-len 0."
            )
        print("[WARN] py-rcsb-api unavailable; using --sizes-cache-csv and/or modeled lengths only.")

    report_rows = []

    kept_rows = []
    total = len(df0)
    for i, (_, row) in enumerate(df0.iterrows(), start=1):
        pdb_id = str(row[args.col_pdb]).strip().upper()
        ligand = str(row.get(args.col_ligand, ""))
        receptor = str(row.get(args.col_receptor, ""))
        subgroup = _normalize_subgroup(row.get(args.col_subgroup, None))

        if i % 250 == 0 or i == total:
            print(f"[FILTER] {i}/{total}")

        if keep_set is not None and subgroup not in keep_set:
            report_rows.append({"pdb": pdb_id, "reason": f"subgroup_filtered:{subgroup}", "subgroup": subgroup})
            continue

        if not _is_valid_pdb_id(pdb_id):
            report_rows.append({"pdb": pdb_id, "reason": "invalid_pdb_id", "subgroup": subgroup})
            continue

        pdb_path = None
        if args.col_pdb_path in row.index:
            pdb_path_raw = row.get(args.col_pdb_path, None)
            if pd.notna(pdb_path_raw) and str(pdb_path_raw).strip():
                pdb_path = Path(str(pdb_path_raw))
                if not pdb_path.is_absolute():
                    pdb_path = pdb_root / pdb_path
        if pdb_path is None:
            pdb_path = pdb_root / f"{pdb_id}.pdb"

        if not pdb_path.exists():
            report_rows.append(
                {"pdb": pdb_id, "reason": f"missing_pdb:{pdb_path}", "subgroup": subgroup, "pdb_path": str(pdb_path)}
            )
            continue

        try:
            stats = partner_ca_stats(pdb_path, ligand, receptor, args.n_interface)
        except Exception as exc:
            report_rows.append(
                {
                    "pdb": pdb_id,
                    "reason": f"ca_parse_error:{type(exc).__name__}",
                    "detail": str(exc)[:300],
                    "subgroup": subgroup,
                    "pdb_path": str(pdb_path),
                }
            )
            continue

        if (
            stats["n_ca_ligand"] < args.n_interface
            or stats["n_ca_receptor"] < args.n_interface
        ):
            report_rows.append(
                {
                    "pdb": pdb_id,
                    "reason": "too_few_CA",
                    "subgroup": subgroup,
                    "pdb_path": str(pdb_path),
                    **stats,
                }
            )
            continue

        if np.isfinite(args.max_dN_nm):
            if stats["dN_ligand_nm"] > args.max_dN_nm or stats["dN_receptor_nm"] > args.max_dN_nm:
                report_rows.append(
                    {
                        "pdb": pdb_id,
                        "reason": "interface_too_small_for_N",
                        "subgroup": subgroup,
                        "pdb_path": str(pdb_path),
                        **stats,
                    }
                )
                continue

        lig_len = None
        rec_len = None
        size_source = None
        if args.min_partner_len > 0:
            cache_key = _size_key(pdb_id, ligand, receptor)
            lig_len, rec_len, size_source = _select_partner_lengths(
                sizes_cache.get(cache_key), args.size_source
            )

            if (lig_len is None or rec_len is None) and can_query_rcsb:
                try:
                    lens = partner_lengths_from_chain_groups_rcsb(
                        pdb_id, ligand, receptor, cache
                    )
                    lig_len = _to_int_or_none(lens.get("ligand_total_residues", None))
                    rec_len = _to_int_or_none(lens.get("receptor_total_residues", None))
                    size_source = "rcsb_live"
                except Exception as exc:
                    report_rows.append(
                        {
                            "pdb": pdb_id,
                            "reason": f"rcsb_error:{type(exc).__name__}",
                            "detail": str(exc)[:300],
                            "subgroup": subgroup,
                            "pdb_path": str(pdb_path),
                            **stats,
                        }
                    )
                    continue

            if (lig_len is None or rec_len is None) and args.size_source in {"auto", "modeled"}:
                lig_len = int(round(stats["n_ca_ligand"]))
                rec_len = int(round(stats["n_ca_receptor"]))
                size_source = size_source or "modeled_from_pdb"

            if lig_len is None or rec_len is None:
                report_rows.append(
                    {
                        "pdb": pdb_id,
                        "reason": "size_unavailable",
                        "subgroup": subgroup,
                        "pdb_path": str(pdb_path),
                        **stats,
                    }
                )
                continue

            if lig_len < args.min_partner_len or rec_len < args.min_partner_len:
                report_rows.append(
                    {
                        "pdb": pdb_id,
                        "reason": "too_small_partner_len",
                        "ligand_len": lig_len,
                        "receptor_len": rec_len,
                        "size_source": size_source,
                        "subgroup": subgroup,
                        "pdb_path": str(pdb_path),
                        **stats,
                    }
                )
                continue

        report_rows.append(
            {
                "pdb": pdb_id,
                "reason": "KEEP",
                "ligand_len": lig_len,
                "receptor_len": rec_len,
                "size_source": size_source,
                "subgroup": subgroup,
                "pdb_path": str(pdb_path),
                **stats,
            }
        )
        row_out = row.copy()
        row_out[args.col_subgroup] = subgroup
        kept_rows.append(row_out)

    if args.min_partner_len > 0 and not args.no_rcsb:
        try:
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception:
            pass

    report_df = pd.DataFrame(report_rows)

    filtered_df = pd.DataFrame(kept_rows)
    if filtered_df.empty:
        out_df = df0.iloc[0:0].copy()
        kept_duplicates = filtered_df
    else:
        out_df, kept_duplicates = _dedup_by_pdb(filtered_df, args.col_pdb)
        if len(kept_duplicates) > 0:
            for _, row in kept_duplicates.iterrows():
                report_rows.append(
                    {
                        "pdb": str(row[args.col_pdb]).strip().upper(),
                        "reason": "DUPLICATE_PDB_DROPPED_AFTER_FILTER",
                        "subgroup": _normalize_subgroup(row.get(args.col_subgroup, None)),
                    }
                )
            report_df = pd.DataFrame(report_rows)

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_csv, index=False)

    stratify_col = str(args.stratify_by).strip()

    pre_cap_count = len(out_df)
    cap_dropped = 0
    if str(args.target_total_size).strip():
        target_total = _parse_size(args.target_total_size, len(out_df))
        if target_total <= 0:
            raise ValueError(
                f"target-total-size must resolve to >=1. Got {args.target_total_size!r} for n={len(out_df)}"
            )
        if target_total < len(out_df):
            if stratify_col:
                out_df = _stratified_sample(out_df, target_total, args.split_seed, stratify_col)
            else:
                rng = np.random.default_rng(int(args.split_seed))
                idx = rng.permutation(len(out_df))[:target_total]
                out_df = out_df.iloc[idx].copy()
            cap_dropped = pre_cap_count - len(out_df)
            print(f"[CAP] target_total_size={target_total} kept={len(out_df)}")
        elif target_total > len(out_df):
            print(
                f"[WARN] target_total_size={target_total} but only {len(out_df)} rows available after filters; keeping all."
            )

    train_target, test_target = _resolve_split_targets(
        len(out_df), args.train_size, args.test_size
    )
    if train_target >= 0 or test_target >= 0:
        if stratify_col:
            train_df, test_df = _stratified_split(
                out_df, train_target, test_target, args.split_seed, stratify_col
            )
        else:
            train_df, test_df = _random_split(out_df, train_target, test_target, args.split_seed)

        test_ids = set(test_df[args.col_pdb].astype(str).str.upper())
        out_df["split"] = out_df[args.col_pdb].astype(str).str.upper().map(
            lambda x: "test" if x in test_ids else "train"
        )

        train_ids = set(train_df[args.col_pdb].astype(str).str.upper())
        overlap = train_ids.intersection(test_ids)
        if overlap:
            raise RuntimeError(f"Train/test overlap detected for PDB IDs: {sorted(list(overlap))[:10]}")

        if train_csv:
            train_csv.parent.mkdir(parents=True, exist_ok=True)
            out_df[out_df["split"] == "train"].to_csv(train_csv, index=False)
        if test_csv:
            test_csv.parent.mkdir(parents=True, exist_ok=True)
            out_df[out_df["split"] == "test"].to_csv(test_csv, index=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    unique_pdb_input = int(df0[args.col_pdb].nunique())
    candidates_after_filters = int(len(filtered_df))
    duplicates_dropped_after_filter = int(len(kept_duplicates))
    capped_dropped = int(cap_dropped)
    print(
        f"[DONE] kept {len(out_df)}/{len(df0)} "
        f"(input_unique_pdb={unique_pdb_input}, "
        f"candidates_after_filters={candidates_after_filters}, "
        f"duplicates_dropped_after_filter={duplicates_dropped_after_filter}, "
        f"capped_dropped={capped_dropped})"
    )
    if "split" in out_df.columns:
        split_counts = out_df["split"].value_counts().to_dict()
        print(f"[SPLIT] {split_counts}")
    print(f"[OUT] {out_csv}")
    print(f"[REPORT] {report_csv}")
    if train_csv:
        print(f"[TRAIN] {train_csv}")
    if test_csv:
        print(f"[TEST] {test_csv}")


if __name__ == "__main__":
    main()

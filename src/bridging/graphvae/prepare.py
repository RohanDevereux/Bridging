from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from bridging.MD.paths import PDB_CACHE_DIR
from bridging.utils.affinity import experimental_delta_g_kcalmol
from bridging.utils.dataset_rows import parse_chain_group, row_chain_groups, row_pdb_id

from .config import (
    DEEPRANK_NODE_FEATURES,
    DYNAMIC_EDGE_FEATURES_BASE,
    DYNAMIC_EDGE_FEATURES_WITH_DIST,
    DYNAMIC_NODE_FEATURES,
    STATIC_EDGE_FEATURES,
    STATIC_NODE_FEATURES,
)
from .chain_remap import build_raw_to_md_chain_map, load_chain_order, remap_query_pair
from .deeprank_adapter import (
    build_deeprank_hdf5,
    index_hdf5_entries,
    load_deeprank_graph,
    stage_query_pdbs,
    write_deeprank_index,
)
from .ids import canonical_complex_id, primary_chain
from .md_dynamics import compute_dynamic_features, load_full_md_trajectory
from .splits import make_train_val_test_split


def _fmt_seconds(seconds: float) -> str:
    s = int(max(0.0, float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _cached_raw_pdb_path(pdb_id: str, pdb_cache_root: Path) -> Path:
    return pdb_cache_root / f"{str(pdb_id).strip().upper()}.pdb"


def _select_complex_entries(dataset_csv: Path) -> tuple[list[dict], dict]:
    df = pd.read_csv(dataset_csv)
    seen: set[str] = set()
    rows: list[dict] = []
    report = {
        "rows_total": int(len(df)),
        "rows_missing_pdb": 0,
        "rows_missing_chain_groups": 0,
        "rows_missing_dg": 0,
        "rows_missing_primary_query_chain": 0,
        "rows_duplicate_complex_id": 0,
    }

    for row in df.to_dict("records"):
        complex_id = canonical_complex_id(row)
        if complex_id is None:
            report["rows_missing_pdb"] += 1
            continue
        left_raw, right_raw = row_chain_groups(row)
        if left_raw is None or right_raw is None:
            report["rows_missing_chain_groups"] += 1
            continue
        left_chains = parse_chain_group(left_raw)
        right_chains = parse_chain_group(right_raw)
        if not left_chains or not right_chains:
            report["rows_missing_chain_groups"] += 1
            continue

        dg = experimental_delta_g_kcalmol(row)
        if dg is None:
            report["rows_missing_dg"] += 1
            continue

        q1 = primary_chain(left_raw)
        q2 = primary_chain(right_raw)
        if q1 is None or q2 is None:
            report["rows_missing_primary_query_chain"] += 1
            continue

        if complex_id in seen:
            report["rows_duplicate_complex_id"] += 1
            continue
        seen.add(complex_id)

        rows.append(
            {
                "complex_id": complex_id,
                "pdb_id": row_pdb_id(row),
                "chains_1": "".join(left_chains),
                "chains_2": "".join(right_chains),
                "query_chain_1": q1,
                "query_chain_2": q2,
                "dG": float(dg),
            }
        )
    report["rows_selected_unique_complex"] = int(len(rows))
    return rows, report


def _resolve_hdf5_paths(
    *,
    entries: list[dict],
    md_root: Path,
    pdb_cache_root: Path,
    out_dir: Path,
    graph_source: str,
    build_deeprank: bool,
    deeprank_hdf5: list[Path] | None,
    deeprank_prefix: Path | None,
    influence_radius: float,
    max_edge_length: float | None,
    overwrite: bool,
) -> tuple[list[Path], dict[str, dict], list[dict], dict]:
    if graph_source not in {"raw_pdb", "md_topology_protein"}:
        raise ValueError(f"Unsupported graph_source={graph_source}")

    source_report = {
        "graph_source": graph_source,
        "n_entries": int(len(entries)),
        "n_md_topology_sources": 0,
        "n_raw_pdb_sources": 0,
        "n_raw_pdb_cache_hits": 0,
        "n_raw_pdb_cache_miss": 0,
        "n_raw_pdb_sources_missing": 0,
        "n_md_topology_missing": 0,
        "n_query_chain_remapped": 0,
        "n_query_chain_fallback": 0,
        "chain_remap_examples": [],
    }
    chain_map_cache: dict[str, tuple[dict[str, str], list[str], dict]] = {}

    for rec in entries:
        raw_pdb_path = _cached_raw_pdb_path(rec["pdb_id"], pdb_cache_root)
        raw_exists = raw_pdb_path.exists()
        if raw_exists:
            source_report["n_raw_pdb_cache_hits"] += 1
        else:
            source_report["n_raw_pdb_cache_miss"] += 1
        rec["pdb_path"] = str(raw_pdb_path)
        rec["query_model_id"] = rec["complex_id"]
        rec["query_source_pdb_path"] = str(raw_pdb_path)

        if graph_source != "md_topology_protein":
            if raw_exists:
                source_report["n_raw_pdb_sources"] += 1
            else:
                source_report["n_raw_pdb_sources_missing"] += 1
            continue

        md_pdb_path = md_root / str(rec["pdb_id"]) / "topology_protein.pdb"
        if not md_pdb_path.exists():
            source_report["n_md_topology_missing"] += 1
            continue

        source_report["n_md_topology_sources"] += 1
        rec["query_source_pdb_path"] = str(md_pdb_path)
        md_chain_order = load_chain_order(md_pdb_path)
        if not md_chain_order:
            source_report["n_md_topology_missing"] += 1
            continue

        if raw_exists:
            cache_key = str(rec["pdb_id"])
            if cache_key not in chain_map_cache:
                chain_map_cache[cache_key] = build_raw_to_md_chain_map(raw_pdb_path, md_pdb_path)
            chain_map, _md_chain_order, map_report = chain_map_cache[cache_key]
        else:
            chain_map = {}
            map_report = {
                "n_raw_chains": 0,
                "n_md_chains": int(len(md_chain_order)),
                "n_mapped": 0,
                "direct_chain_id_overlap": [],
                "mapping_scores": {},
                "note": "raw_pdb_missing_used_md_chain_fallback",
            }

        q1_old = str(rec["query_chain_1"]).strip().upper()
        q2_old = str(rec["query_chain_2"]).strip().upper()
        q1_new, q2_new = remap_query_pair(
            query_chain_1=q1_old,
            query_chain_2=q2_old,
            chain_map=chain_map,
            md_chain_order=md_chain_order,
        )
        rec["query_chain_1"] = q1_new
        rec["query_chain_2"] = q2_new
        if q1_new != q1_old or q2_new != q2_old:
            source_report["n_query_chain_remapped"] += 1
        if q1_old not in chain_map or q2_old not in chain_map:
            source_report["n_query_chain_fallback"] += 1
        if len(source_report["chain_remap_examples"]) < 30 and (q1_new != q1_old or q2_new != q2_old):
            source_report["chain_remap_examples"].append(
                {
                    "complex_id": rec["complex_id"],
                    "query_old": [q1_old, q2_old],
                    "query_new": [q1_new, q2_new],
                    "map_report": map_report,
                }
            )

    if build_deeprank:
        pdb_stage = out_dir / "deeprank_stage_pdb"
        stage_inputs = [rec for rec in entries if Path(str(rec.get("query_source_pdb_path", rec["pdb_path"]))).exists()]
        staged = stage_query_pdbs(stage_inputs, pdb_stage, overwrite=overwrite)
        for rec in staged:
            rec["query_model_id"] = Path(rec["query_pdb_path"]).stem
        prefix = deeprank_prefix if deeprank_prefix else (out_dir / "deeprank_graphs")
        hdf5_paths = build_deeprank_hdf5(
            staged_entries=staged,
            out_prefix=prefix,
            influence_radius=influence_radius,
            max_edge_length=max_edge_length,
        )
        entries = staged
    else:
        if not deeprank_hdf5:
            raise ValueError("Provide --deep-rank-hdf5 when --build-deeprank is not set")
        hdf5_paths = [Path(p) for p in deeprank_hdf5]

    index = index_hdf5_entries(hdf5_paths)
    write_deeprank_index(index, out_dir / "deeprank_index.json")
    source_report["n_deeprank_stage_inputs"] = int(len(entries) if build_deeprank else 0)
    source_report["n_deeprank_indexed_entries"] = int(len(index))
    return hdf5_paths, index, entries, source_report


def build_prepared_dataset(
    *,
    dataset_csv: Path,
    md_root: Path,
    out_dir: Path,
    pdb_cache_root: Path,
    graph_source: str,
    build_deeprank: bool,
    deeprank_hdf5: list[Path] | None,
    deeprank_prefix: Path | None,
    influence_radius: float,
    max_edge_length: float | None,
    train_fraction: float,
    val_fraction: float,
    split_seed: int,
    frames_per_complex: int,
    include_dynamic_dist_stats: bool,
    require_all_protein_nodes: bool,
    overwrite: bool,
    progress_every: int = 25,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, select_report = _select_complex_entries(dataset_csv)
    split_map = make_train_val_test_split(
        [e["complex_id"] for e in entries],
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=split_seed,
    )
    hdf5_paths, deeprank_index, entries, source_report = _resolve_hdf5_paths(
        entries=entries,
        md_root=md_root,
        pdb_cache_root=pdb_cache_root,
        out_dir=out_dir,
        graph_source=graph_source,
        build_deeprank=build_deeprank,
        deeprank_hdf5=deeprank_hdf5,
        deeprank_prefix=deeprank_prefix,
        influence_radius=influence_radius,
        max_edge_length=max_edge_length,
        overwrite=overwrite,
    )

    edge_dyn_names = list(DYNAMIC_EDGE_FEATURES_WITH_DIST if include_dynamic_dist_stats else DYNAMIC_EDGE_FEATURES_BASE)
    node_feature_names = list(STATIC_NODE_FEATURES) + list(DYNAMIC_NODE_FEATURES)
    edge_feature_names = list(STATIC_EDGE_FEATURES) + edge_dyn_names

    records = []
    missing_graph = []
    missing_md = []
    partial_node_coverage = []
    traj_cache = {}
    total_entries = int(len(entries))
    processed = 0
    loop_t0 = time.perf_counter()

    print(
        f"[PREP] start total_entries={total_entries} graph_source={graph_source} "
        f"frames_per_complex={frames_per_complex} progress_every={progress_every}"
    )

    def _log_progress() -> None:
        if progress_every <= 0:
            return
        if processed % progress_every != 0 and processed != total_entries:
            return
        elapsed = time.perf_counter() - loop_t0
        rate = (processed / elapsed) if elapsed > 0 else 0.0
        eta = ((total_entries - processed) / rate) if rate > 0 else 0.0
        pct = (100.0 * processed / total_entries) if total_entries > 0 else 100.0
        print(
            f"[PREP] progress {processed}/{total_entries} ({pct:.1f}%) "
            f"records={len(records)} missing_md={len(missing_md)} "
            f"missing_graph={len(missing_graph)} partial={len(partial_node_coverage)} "
            f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)} "
            f"rate={rate*3600:.1f}/h"
        )

    for rec in entries:
        processed += 1
        pdb_id = rec["pdb_id"]
        md_dir = md_root / str(pdb_id)
        done = md_dir / "DONE"
        if not done.exists():
            missing_md.append(rec["complex_id"])
            _log_progress()
            continue

        model_id = rec.get("query_model_id", rec["complex_id"])
        graph_ref = deeprank_index.get(model_id)
        if graph_ref is None:
            missing_graph.append(rec["complex_id"])
            _log_progress()
            continue

        graph = load_deeprank_graph(
            hdf5_path=Path(graph_ref["hdf5_path"]),
            entry_name=graph_ref["entry_name"],
            node_feature_names=list(DEEPRANK_NODE_FEATURES),
            edge_feature_names=list(STATIC_EDGE_FEATURES),
        )

        if pdb_id not in traj_cache:
            traj_cache[pdb_id] = load_full_md_trajectory(md_dir, max_frames=frames_per_complex)
        dyn = compute_dynamic_features(
            traj=traj_cache[pdb_id],
            node_chain_id=graph["node_chain_id"],
            node_position=graph["node_position"],
            edge_index=graph["edge_index"],
            include_distance_stats=include_dynamic_dist_stats,
        )
        graph_node_keys = {
            (str(c).strip().upper(), int(p))
            for c, p in zip(graph["node_chain_id"], graph["node_position"])
        }
        protein_node_keys = set(dyn["protein_residue_keys"])
        missing_nodes = sorted(protein_node_keys - graph_node_keys)
        if missing_nodes:
            partial_node_coverage.append(
                {
                    "complex_id": rec["complex_id"],
                    "missing_count": int(len(missing_nodes)),
                    "missing_sample": [
                        {"chain": c, "resseq": int(p)}
                        for c, p in missing_nodes[:20]
                    ],
                }
            )
            if require_all_protein_nodes:
                raise RuntimeError(
                    f"Graph missing protein residues for {rec['complex_id']}: "
                    f"missing={len(missing_nodes)}. "
                    "Increase influence radius / adjust DeepRank query or pass "
                    "--allow-partial-node-coverage."
                )

        node_features = torch.as_tensor(
            pd.concat(
                [
                    pd.DataFrame(graph["node_features"], columns=list(DEEPRANK_NODE_FEATURES)),
                    pd.DataFrame(
                        dyn["node_structural_context"],
                        columns=["n_same_chain_8A", "n_other_chain_8A"],
                    ),
                    pd.DataFrame(dyn["node_dynamic"], columns=list(DYNAMIC_NODE_FEATURES)),
                ],
                axis=1,
            ).to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        edge_features = torch.as_tensor(
            pd.concat(
                [
                    pd.DataFrame(graph["edge_features"], columns=list(STATIC_EDGE_FEATURES)),
                    pd.DataFrame(dyn["edge_dynamic"], columns=edge_dyn_names),
                ],
                axis=1,
            ).to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)

        records.append(
            {
                "complex_id": rec["complex_id"],
                "pdb_id": rec["pdb_id"],
                "split": split_map[rec["complex_id"]],
                "dG": float(rec["dG"]),
                "node_feature_names": node_feature_names,
                "edge_feature_names": edge_feature_names,
                "node_features": node_features,
                "edge_features": edge_features,
                "edge_index": edge_index,
                "node_chain_id": graph["node_chain_id"],
                "node_position": graph["node_position"],
                "deeprank_hdf5": graph_ref["hdf5_path"],
                "deeprank_entry_name": graph_ref["entry_name"],
            }
        )

        _log_progress()

    records_path = out_dir / "graph_records.pt"
    torch.save(records, records_path)
    split_path = out_dir / "splits.json"
    split_path.write_text(json.dumps(split_map, indent=2), encoding="utf-8")

    report = {
        "dataset_csv": str(dataset_csv),
        "md_root": str(md_root),
        "graph_source": graph_source,
        "progress_every": int(progress_every),
        "records_path": str(records_path),
        "splits_path": str(split_path),
        "deeprank_hdf5_paths": [str(p) for p in hdf5_paths],
        "graph_source_report": source_report,
        "select_report": select_report,
        "n_records": int(len(records)),
        "n_missing_graph": int(len(missing_graph)),
        "n_missing_md": int(len(missing_md)),
        "n_partial_node_coverage": int(len(partial_node_coverage)),
        "missing_graph_complex_ids": missing_graph[:100],
        "missing_md_complex_ids": missing_md[:100],
        "partial_node_coverage": partial_node_coverage[:100],
        "node_feature_names": node_feature_names,
        "edge_feature_names": edge_feature_names,
        "split_counts": {
            "train": int(sum(1 for r in records if r["split"] == "train")),
            "val": int(sum(1 for r in records if r["split"] == "val")),
            "test": int(sum(1 for r in records if r["split"] == "test")),
        },
    }
    report_path = out_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare graph dataset for masked Graph-VAE experiments (S and SD) using "
            "DeepRank2 static features and MD-derived dynamic features."
        )
    )
    parser.add_argument("--dataset", required=True, help="PPB-derived CSV with affinity labels/chains.")
    parser.add_argument("--md-root", required=True, help="Per-PDB MD directories containing traj_full/topology_full.")
    parser.add_argument("--out-dir", required=True, help="Output directory for prepared graph dataset.")
    parser.add_argument("--pdb-cache-root", default=str(PDB_CACHE_DIR))
    parser.add_argument(
        "--graph-source",
        choices=["raw_pdb", "md_topology_protein"],
        default="md_topology_protein",
        help="Source PDB used to build DeepRank graphs. Use md_topology_protein to match saved MD ID space.",
    )
    parser.add_argument("--build-deeprank", action="store_true", help="Generate DeepRank2 HDF5 from staged PDBs.")
    parser.add_argument("--deep-rank-hdf5", nargs="*", help="Existing DeepRank2 HDF5 file(s).")
    parser.add_argument("--deeprank-prefix", help="Output prefix for DeepRank2 generated HDF5.")
    parser.add_argument(
        "--influence-radius",
        type=float,
        default=1_000_000.0,
        help="DeepRank influence radius in Angstrom. Large finite default approximates whole-complex coverage.",
    )
    parser.add_argument("--max-edge-length", type=float)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--frames-per-complex", type=int, default=120)
    parser.add_argument("--include-dynamic-dist-stats", action="store_true")
    parser.add_argument("--allow-partial-node-coverage", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N processed complexes.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_prepared_dataset(
        dataset_csv=Path(args.dataset),
        md_root=Path(args.md_root),
        out_dir=Path(args.out_dir),
        pdb_cache_root=Path(args.pdb_cache_root).expanduser(),
        graph_source=str(args.graph_source),
        build_deeprank=bool(args.build_deeprank),
        deeprank_hdf5=[Path(p) for p in args.deep_rank_hdf5] if args.deep_rank_hdf5 else None,
        deeprank_prefix=Path(args.deeprank_prefix) if args.deeprank_prefix else None,
        influence_radius=float(args.influence_radius),
        max_edge_length=float(args.max_edge_length) if args.max_edge_length is not None else None,
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        frames_per_complex=int(args.frames_per_complex),
        include_dynamic_dist_stats=bool(args.include_dynamic_dist_stats),
        require_all_protein_nodes=not bool(args.allow_partial_node_coverage),
        overwrite=bool(args.overwrite),
        progress_every=int(args.progress_every),
    )
    print(f"[PREP] records={report['n_records']} train={report['split_counts']['train']} val={report['split_counts']['val']} test={report['split_counts']['test']}")
    print(f"[PREP] report={Path(args.out_dir) / 'prepare_report.json'}")


if __name__ == "__main__":
    main()

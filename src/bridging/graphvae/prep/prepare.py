from __future__ import annotations

import argparse
import contextlib
from collections import OrderedDict
import json
import os
import re
import signal
import shutil
import time
from pathlib import Path

import pandas as pd
import torch

from bridging.MD.paths import PDB_CACHE_DIR
from bridging.MD.prefetch_pdbs import ensure_pdb_cached
from bridging.utils.affinity import experimental_delta_g_kcalmol
from bridging.utils.dataset_rows import parse_chain_group, row_chain_groups, row_pdb_id

from ..common.config import (
    DEEPRANK_NODE_FEATURES,
    DYNAMIC_EDGE_FEATURES_BASE,
    DYNAMIC_EDGE_FEATURES_WITH_DIST,
    DYNAMIC_NODE_FEATURES,
    DYNAMIC_NODE_INPUT_FEATURES,
    FORCE_NODE_INPUT_FEATURES,
    NODE_IDENTITY_FEATURES,
    STATIC_EDGE_FEATURES,
    STATIC_NODE_FEATURES,
    TORSION_NODE_INPUT_FEATURES,
)
from ..common.chain_remap import build_raw_to_md_chain_map, load_chain_order, remap_query_pair
from .deeprank_adapter import (
    build_deeprank_hdf5,
    index_hdf5_entries,
    load_deeprank_graph,
    stage_query_pdbs,
    write_deeprank_index,
)
from .force_features import assess_force_query_compatibility, compute_node_interchain_force_features
from ..common.ids import canonical_complex_id, primary_chain, sanitize_filename_token
from .md_dynamics import (
    compute_dynamic_features,
    compute_node_torsion_sincos_features,
    load_full_md_trajectory,
    load_protein_md_trajectory,
)
from ..common.splits import make_train_val_test_split


def _fmt_seconds(seconds: float) -> str:
    s = int(max(0.0, float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


class ComplexProcessingTimeoutError(TimeoutError):
    pass


@contextlib.contextmanager
def _complex_timeout(timeout_seconds: float) -> None:
    timeout_seconds = float(timeout_seconds)
    if timeout_seconds <= 0:
        yield
        return
    if os.name == "nt" or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum, frame):
        raise ComplexProcessingTimeoutError(
            f"Timed out after {timeout_seconds:.1f} seconds while preparing complex."
        )

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _checkpoint_index_from_name(path: Path) -> int:
    m = re.fullmatch(r"records_(\d{5})\.pt", path.name)
    if m is None:
        return -1
    return int(m.group(1))


def _sorted_checkpoint_shards(checkpoint_dir: Path) -> list[Path]:
    shards = [p for p in checkpoint_dir.glob("records_*.pt") if p.is_file()]
    return sorted(shards, key=_checkpoint_index_from_name)


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


def _multichain_complexes(entries: list[dict]) -> list[str]:
    out = []
    for rec in entries:
        left = str(rec.get("chains_1", "")).strip()
        right = str(rec.get("chains_2", "")).strip()
        if len(left) > 1 or len(right) > 1:
            out.append(str(rec["complex_id"]))
    return out


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
    deeprank_query_mode: str,
    deeprank_cpu_count: int | None,
    overwrite: bool,
    progress_every: int,
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
        "n_query_chain_unresolved": 0,
        "n_stage_skipped_not_done": 0,
        "n_stage_skipped_missing_source_pdb": 0,
        "chain_remap_examples": [],
    }
    chain_map_cache: dict[tuple[str, str, str], tuple[dict[str, str], list[str], dict]] = {}

    total_entries = int(len(entries))
    for i, rec in enumerate(entries, start=1):
        raw_pdb_path = _cached_raw_pdb_path(rec["pdb_id"], pdb_cache_root)
        raw_exists = raw_pdb_path.exists()
        if raw_exists:
            source_report["n_raw_pdb_cache_hits"] += 1
        else:
            source_report["n_raw_pdb_cache_miss"] += 1
        rec["pdb_path"] = str(raw_pdb_path)
        rec["query_model_id"] = sanitize_filename_token(rec["complex_id"])
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

        q1_old = str(rec["query_chain_1"]).strip().upper()
        q2_old = str(rec["query_chain_2"]).strip().upper()

        if raw_exists:
            cache_key = (str(rec["pdb_id"]), q1_old, q2_old)
            if cache_key not in chain_map_cache:
                chain_map_cache[cache_key] = build_raw_to_md_chain_map(
                    raw_pdb_path,
                    md_pdb_path,
                    query_chains=[q1_old, q2_old],
                )
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

        try:
            q1_new, q2_new = remap_query_pair(
                query_chain_1=q1_old,
                query_chain_2=q2_old,
                chain_map=chain_map,
                md_chain_order=md_chain_order,
                strict=raw_exists,
            )
        except Exception:
            source_report["n_query_chain_unresolved"] += 1
            source_report["n_query_chain_fallback"] += 1
            if len(source_report["chain_remap_examples"]) < 30:
                source_report["chain_remap_examples"].append(
                    {
                        "complex_id": rec["complex_id"],
                        "query_old": [q1_old, q2_old],
                        "query_new": None,
                        "map_report": map_report,
                    }
                )
            if progress_every > 0 and (i % progress_every == 0 or i == total_entries):
                print(
                    f"[PREP] source_resolve {i}/{total_entries} "
                    f"md_sources={source_report['n_md_topology_sources']} "
                    f"md_missing={source_report['n_md_topology_missing']} "
                    f"remapped={source_report['n_query_chain_remapped']} "
                    f"fallback={source_report['n_query_chain_fallback']}",
                    flush=True,
                )
            continue
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
        if progress_every > 0 and (i % progress_every == 0 or i == total_entries):
            print(
                f"[PREP] source_resolve {i}/{total_entries} "
                f"md_sources={source_report['n_md_topology_sources']} "
                f"md_missing={source_report['n_md_topology_missing']} "
                f"remapped={source_report['n_query_chain_remapped']} "
                f"fallback={source_report['n_query_chain_fallback']}",
                flush=True,
            )

    if build_deeprank:
        pdb_stage = out_dir / "deeprank_stage_pdb"
        stage_inputs = []
        for rec in entries:
            done_path = md_root / str(rec["pdb_id"]) / "DONE"
            if not done_path.exists():
                source_report["n_stage_skipped_not_done"] += 1
                continue
            source_pdb = Path(str(rec.get("query_source_pdb_path", rec["pdb_path"])))
            if not source_pdb.exists():
                source_report["n_stage_skipped_missing_source_pdb"] += 1
                continue
            stage_inputs.append(rec)
        print(
            f"[PREP] deeprank stage_inputs={len(stage_inputs)} "
            f"skipped_not_done={source_report['n_stage_skipped_not_done']} "
            f"skipped_missing_source_pdb={source_report['n_stage_skipped_missing_source_pdb']}",
            flush=True,
        )
        staged = stage_query_pdbs(stage_inputs, pdb_stage, overwrite=overwrite)
        for rec in staged:
            rec["query_model_id"] = Path(rec["query_pdb_path"]).stem
        prefix = deeprank_prefix if deeprank_prefix else (out_dir / "deeprank_graphs")
        dr_t0 = time.perf_counter()
        print(f"[PREP] deeprank build start prefix={prefix}", flush=True)
        hdf5_paths = build_deeprank_hdf5(
            staged_entries=staged,
            out_prefix=prefix,
            influence_radius=influence_radius,
            max_edge_length=max_edge_length,
            query_mode=deeprank_query_mode,
            cpu_count=deeprank_cpu_count,
        )
        print(
            f"[PREP] deeprank build done n_hdf5={len(hdf5_paths)} "
            f"elapsed={_fmt_seconds(time.perf_counter() - dr_t0)}",
            flush=True,
        )
        entries = staged
    else:
        if not deeprank_hdf5:
            raise ValueError("Provide --deep-rank-hdf5 when --build-deeprank is not set")
        hdf5_paths = [Path(p) for p in deeprank_hdf5]

    index = index_hdf5_entries(hdf5_paths)
    write_deeprank_index(index, out_dir / "deeprank_index.json")
    print(
        f"[PREP] deeprank index hdf5_files={len(hdf5_paths)} entries={len(index)}",
        flush=True,
    )
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
    deeprank_query_mode: str,
    deeprank_cpu_count: int | None,
    train_fraction: float,
    val_fraction: float,
    split_seed: int,
    frames_per_complex: int,
    traj_cache_size: int,
    checkpoint_every: int,
    include_dynamic_dist_stats: bool,
    require_all_protein_nodes: bool,
    overwrite: bool,
    progress_every: int = 25,
    per_complex_timeout_minutes: float = 30.0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "graph_records.pt"
    split_path = out_dir / "splits.json"
    report_path = out_dir / "prepare_report.json"
    for stale in (records_path, split_path, report_path):
        if stale.exists():
            stale.unlink()
    entries, select_report = _select_complex_entries(dataset_csv)
    if require_all_protein_nodes and build_deeprank and deeprank_query_mode == "ppi":
        multi = _multichain_complexes(entries)
        if multi:
            raise RuntimeError(
                "Strict full-protein node coverage is incompatible with DeepRank single-pair query "
                "for multichain complexes when --deeprank-query-mode=ppi. "
                f"multichain_complexes={len(multi)} sample={multi[:10]}. "
                "Use --deeprank-query-mode=full_complex for all-residue coverage, "
                "or use --allow-partial-node-coverage."
            )
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
        deeprank_query_mode=deeprank_query_mode,
        deeprank_cpu_count=deeprank_cpu_count,
        overwrite=overwrite,
        progress_every=progress_every,
    )

    edge_dyn_names = list(DYNAMIC_EDGE_FEATURES_WITH_DIST if include_dynamic_dist_stats else DYNAMIC_EDGE_FEATURES_BASE)
    node_feature_names = list(STATIC_NODE_FEATURES) + list(DYNAMIC_NODE_INPUT_FEATURES)
    edge_feature_names = list(STATIC_EDGE_FEATURES) + edge_dyn_names

    checkpoint_every = max(1, int(checkpoint_every))
    checkpoint_dir = out_dir / "checkpoints"
    if overwrite and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resumed_records = 0
    resume_skipped = 0
    completed_complex_ids: set[str] = set()
    shard_paths = _sorted_checkpoint_shards(checkpoint_dir)
    next_shard_idx = 0
    if shard_paths and not overwrite:
        for shard in shard_paths:
            chunk = torch.load(shard, map_location="cpu")
            if not isinstance(chunk, list):
                raise RuntimeError(f"Invalid checkpoint shard format: {shard}")
            resumed_records += int(len(chunk))
            for row in chunk:
                cid = str(row.get("complex_id", "")).strip()
                if cid:
                    completed_complex_ids.add(cid)
        next_shard_idx = max(_checkpoint_index_from_name(p) for p in shard_paths) + 1
        print(
            f"[PREP] resume shards={len(shard_paths)} resumed_records={resumed_records} "
            f"completed_complexes={len(completed_complex_ids)} checkpoint_dir={checkpoint_dir}",
            flush=True,
        )

    records_buffer: list[dict] = []
    records_new_count = 0
    missing_graph = []
    missing_md = []
    bad_md = []
    partial_node_coverage = []
    torsion_feature_fallbacks = []
    force_feature_fallbacks = []
    force_query_incompatible = []
    complex_timeouts = []
    traj_cache: OrderedDict[str, object] = OrderedDict()
    traj_cache_size = max(0, int(traj_cache_size))
    per_complex_timeout_minutes = max(0.0, float(per_complex_timeout_minutes))
    per_complex_timeout_seconds = per_complex_timeout_minutes * 60.0
    total_entries = int(len(entries))
    processed = 0
    loop_t0 = time.perf_counter()

    print(
        f"[PREP] start total_entries={total_entries} graph_source={graph_source} "
        f"frames_per_complex={frames_per_complex} progress_every={progress_every} "
        f"checkpoint_every={checkpoint_every} checkpoint_dir={checkpoint_dir} "
        f"traj_cache_size={traj_cache_size} "
        f"per_complex_timeout_minutes={per_complex_timeout_minutes:.1f}"
    )

    def _flush_records_buffer() -> None:
        nonlocal records_buffer, next_shard_idx
        if not records_buffer:
            return
        shard_path = checkpoint_dir / f"records_{next_shard_idx:05d}.pt"
        torch.save(records_buffer, shard_path)
        next_shard_idx += 1
        records_buffer = []
        total_saved = resumed_records + records_new_count
        print(
            f"[PREP] checkpoint saved={shard_path.name} records_total={total_saved}",
            flush=True,
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
            f"records={resumed_records + records_new_count} missing_md={len(missing_md)} "
            f"missing_graph={len(missing_graph)} bad_md={len(bad_md)} "
            f"partial={len(partial_node_coverage)} timed_out={len(complex_timeouts)} "
            f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)} "
            f"rate={rate*3600:.1f}/h"
        )

    for rec in entries:
        processed += 1
        complex_id = str(rec["complex_id"])
        if complex_id in completed_complex_ids:
            resume_skipped += 1
            _log_progress()
            continue
        pdb_id = rec["pdb_id"]
        md_dir = md_root / str(pdb_id)
        done = md_dir / "DONE"
        if not done.exists():
            missing_md.append(complex_id)
            _log_progress()
            continue

        model_id = rec.get("query_model_id", complex_id)
        graph_ref = deeprank_index.get(model_id)
        if graph_ref is None:
            missing_graph.append(complex_id)
            _log_progress()
            continue

        current_step = "load_deeprank_graph"
        try:
            with _complex_timeout(per_complex_timeout_seconds):
                graph = load_deeprank_graph(
                    hdf5_path=Path(graph_ref["hdf5_path"]),
                    entry_name=graph_ref["entry_name"],
                    node_feature_names=list(DEEPRANK_NODE_FEATURES),
                    edge_feature_names=list(STATIC_EDGE_FEATURES),
                )

                current_step = "load_full_md_trajectory"
                try:
                    if pdb_id in traj_cache:
                        traj = traj_cache.pop(pdb_id)
                        traj_cache[pdb_id] = traj
                    else:
                        traj = load_full_md_trajectory(md_dir, max_frames=frames_per_complex)
                        if traj_cache_size > 0:
                            traj_cache[pdb_id] = traj
                            if len(traj_cache) > traj_cache_size:
                                traj_cache.popitem(last=False)
                    current_step = "compute_dynamic_features"
                    dyn = compute_dynamic_features(
                        traj=traj,
                        node_chain_id=graph["node_chain_id"],
                        node_position=graph["node_position"],
                        edge_index=graph["edge_index"],
                        include_distance_stats=include_dynamic_dist_stats,
                    )
                except Exception as exc:
                    bad_md.append({"complex_id": complex_id, "pdb_id": pdb_id, "error": repr(exc)})
                    _log_progress()
                    continue

                torsion_features = torch.zeros(
                    (len(graph["node_chain_id"]), len(TORSION_NODE_INPUT_FEATURES)),
                    dtype=torch.float32,
                )
                current_step = "compute_torsion_features"
                try:
                    torsion_arr, _torsion_stats = compute_node_torsion_sincos_features(
                        traj=traj,
                        node_chain_id=graph["node_chain_id"],
                        node_position=graph["node_position"],
                    )
                    torsion_features = torch.as_tensor(torsion_arr, dtype=torch.float32)
                except Exception as exc:
                    torsion_feature_fallbacks.append(
                        {"complex_id": complex_id, "pdb_id": pdb_id, "error": repr(exc)}
                    )

                current_step = "ensure_raw_pdb"
                raw_pdb_path = _cached_raw_pdb_path(str(pdb_id), pdb_cache_root)
                if not raw_pdb_path.exists():
                    raw_pdb_path, _ = ensure_pdb_cached(str(pdb_id), cache_dir=pdb_cache_root)

                current_step = "assess_force_query_compatibility"
                compat = assess_force_query_compatibility(
                    raw_pdb_path=raw_pdb_path,
                    protein_topology_pdb=md_dir / "topology_protein.pdb",
                    full_topology_pdb=md_dir / "topology_full.pdb",
                    ligand_group=str(rec["chains_1"]),
                    receptor_group=str(rec["chains_2"]),
                )
                if not bool(compat.get("compatible", False)):
                    force_query_incompatible.append(
                        {
                            "complex_id": complex_id,
                            "pdb_id": pdb_id,
                            "reason": str(compat.get("compatibility_reason", "")),
                            "missing_in_raw": list(compat.get("missing_in_raw", [])),
                            "missing_in_full": list(compat.get("missing_in_full", [])),
                            "missing_in_protein": list(compat.get("missing_in_protein", [])),
                            "raw_query_overlap": list(compat.get("raw_query_overlap", [])),
                        }
                    )
                    _log_progress()
                    continue

                current_step = "load_protein_md_trajectory"
                try:
                    traj_protein = load_protein_md_trajectory(md_dir, max_frames=frames_per_complex)
                    current_step = "compute_force_features"
                    force_arr, _force_stats = compute_node_interchain_force_features(
                        traj=traj_protein,
                        topology_pdb=md_dir / "topology_protein.pdb",
                        raw_pdb_path=raw_pdb_path,
                        ligand_group=str(rec["chains_1"]),
                        receptor_group=str(rec["chains_2"]),
                        node_chain_id=graph["node_chain_id"],
                        node_position=graph["node_position"],
                    )
                    force_features = torch.as_tensor(force_arr, dtype=torch.float32)
                except Exception as exc:
                    force_feature_fallbacks.append(
                        {
                            "complex_id": complex_id,
                            "pdb_id": pdb_id,
                            "error": repr(exc),
                        }
                    )
                    _log_progress()
                    continue

                current_step = "validate_node_coverage"
                graph_node_keys = {
                    (str(c).strip().upper(), int(p))
                    for c, p in zip(graph["node_chain_id"], graph["node_position"])
                }
                protein_node_keys = set(dyn["protein_residue_keys"])
                missing_nodes = sorted(protein_node_keys - graph_node_keys)
                if missing_nodes:
                    partial_node_coverage.append(
                        {
                            "complex_id": complex_id,
                            "missing_count": int(len(missing_nodes)),
                            "missing_sample": [
                                {"chain": c, "resseq": int(p)}
                                for c, p in missing_nodes[:20]
                            ],
                        }
                    )
                    if require_all_protein_nodes:
                        raise RuntimeError(
                            f"Graph missing protein residues for {complex_id}: "
                            f"missing={len(missing_nodes)}. "
                            "Increase influence radius / adjust DeepRank query or pass "
                            "--allow-partial-node-coverage."
                        )

                current_step = "assemble_features"
                node_features = torch.as_tensor(
                    pd.concat(
                        [
                            pd.DataFrame(graph["node_features"], columns=list(DEEPRANK_NODE_FEATURES)),
                            pd.DataFrame(
                                dyn["node_structural_context"],
                                columns=["n_same_chain_8A", "n_other_chain_8A"],
                            ),
                            pd.DataFrame(dyn["node_identity"], columns=list(NODE_IDENTITY_FEATURES)),
                            pd.DataFrame(dyn["node_dynamic"], columns=list(DYNAMIC_NODE_FEATURES)),
                            pd.DataFrame(torsion_features.cpu().numpy(), columns=list(TORSION_NODE_INPUT_FEATURES)),
                            pd.DataFrame(force_features.cpu().numpy(), columns=list(FORCE_NODE_INPUT_FEATURES)),
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
        except ComplexProcessingTimeoutError as exc:
            complex_timeouts.append(
                {
                    "complex_id": complex_id,
                    "pdb_id": pdb_id,
                    "timeout_minutes": float(per_complex_timeout_minutes),
                    "step": current_step,
                    "error": repr(exc),
                }
            )
            print(
                f"[PREP] timeout complex_id={complex_id} pdb_id={pdb_id} "
                f"step={current_step} timeout_minutes={per_complex_timeout_minutes:.1f}",
                flush=True,
            )
            _log_progress()
            continue

        records_buffer.append(
            {
                "complex_id": complex_id,
                "pdb_id": rec["pdb_id"],
                "split": split_map[complex_id],
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
        records_new_count += 1
        if len(records_buffer) >= checkpoint_every:
            _flush_records_buffer()

        _log_progress()

    _flush_records_buffer()
    final_shards = _sorted_checkpoint_shards(checkpoint_dir)
    records: list[dict] = []
    for shard in final_shards:
        chunk = torch.load(shard, map_location="cpu")
        if not isinstance(chunk, list):
            raise RuntimeError(f"Invalid checkpoint shard format: {shard}")
        records.extend(chunk)

    torch.save(records, records_path)
    split_path.write_text(json.dumps(split_map, indent=2), encoding="utf-8")
    split_counts = {
        "train": int(sum(1 for r in records if r["split"] == "train")),
        "val": int(sum(1 for r in records if r["split"] == "val")),
        "test": int(sum(1 for r in records if r["split"] == "test")),
    }

    report = {
        "dataset_csv": str(dataset_csv),
        "md_root": str(md_root),
        "graph_source": graph_source,
        "progress_every": int(progress_every),
        "checkpoint_every": int(checkpoint_every),
        "per_complex_timeout_minutes": float(per_complex_timeout_minutes),
        "checkpoint_dir": str(checkpoint_dir),
        "records_path": str(records_path),
        "splits_path": str(split_path),
        "deeprank_hdf5_paths": [str(p) for p in hdf5_paths],
        "graph_source_report": source_report,
        "select_report": select_report,
        "n_records": int(len(records)),
        "n_resume_records": int(resumed_records),
        "n_resume_skipped": int(resume_skipped),
        "n_checkpoint_shards": int(len(final_shards)),
        "n_missing_graph": int(len(missing_graph)),
        "n_missing_md": int(len(missing_md)),
        "n_bad_md": int(len(bad_md)),
        "n_partial_node_coverage": int(len(partial_node_coverage)),
        "n_torsion_feature_fallbacks": int(len(torsion_feature_fallbacks)),
        "n_force_query_incompatible": int(len(force_query_incompatible)),
        "n_force_feature_fallbacks": int(len(force_feature_fallbacks)),
        "n_complex_timeouts": int(len(complex_timeouts)),
        "missing_graph_complex_ids": missing_graph[:100],
        "missing_md_complex_ids": missing_md[:100],
        "bad_md": bad_md[:100],
        "partial_node_coverage": partial_node_coverage[:100],
        "torsion_feature_fallbacks": torsion_feature_fallbacks[:100],
        "force_query_incompatible": force_query_incompatible[:100],
        "force_feature_fallbacks": force_feature_fallbacks[:100],
        "complex_timeouts": complex_timeouts[:100],
        "node_feature_names": node_feature_names,
        "edge_feature_names": edge_feature_names,
        "split_counts": split_counts,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if len(records) < 1:
        raise RuntimeError(
            "No prepared graph records were produced. "
            f"missing_md={len(missing_md)} missing_graph={len(missing_graph)} bad_md={len(bad_md)}"
        )
    if min(split_counts.values()) < 1:
        raise RuntimeError(
            "Prepared records produced empty split after MD/graph filtering. "
            f"split_counts={split_counts}. "
            "Adjust split seed/fractions or increase available DONE+graph overlap."
        )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare graph dataset for masked Graph-VAE experiments (S and SD) using "
            "DeepRank2 static features together with MD-derived dynamic, torsion, and "
            "inter-chain force node features."
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
        "--deeprank-query-mode",
        choices=["full_complex", "ppi"],
        default="full_complex",
        help="DeepRank query type: full_complex includes all protein residues; ppi uses one chain pair.",
    )
    parser.add_argument(
        "--deeprank-cpu-count",
        type=int,
        default=4,
        help="CPU workers used by DeepRank QueryCollection.process.",
    )
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
    parser.add_argument(
        "--traj-cache-size",
        type=int,
        default=1,
        help="Max number of loaded MD trajectories kept in memory during prepare.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write checkpoint shard every N newly prepared records.",
    )
    parser.add_argument(
        "--include-dynamic-dist-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-partial-node-coverage", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N processed complexes.")
    parser.add_argument(
        "--per-complex-timeout-minutes",
        type=float,
        default=30.0,
        help="Abort and skip any individual complex that takes longer than this many minutes. Use 0 to disable.",
    )
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
        deeprank_query_mode=str(args.deeprank_query_mode),
        deeprank_cpu_count=int(args.deeprank_cpu_count) if args.deeprank_cpu_count is not None else None,
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        frames_per_complex=int(args.frames_per_complex),
        traj_cache_size=int(args.traj_cache_size),
        checkpoint_every=int(args.checkpoint_every),
        include_dynamic_dist_stats=bool(args.include_dynamic_dist_stats),
        require_all_protein_nodes=not bool(args.allow_partial_node_coverage),
        overwrite=bool(args.overwrite),
        progress_every=int(args.progress_every),
        per_complex_timeout_minutes=float(args.per_complex_timeout_minutes),
    )
    print(f"[PREP] records={report['n_records']} train={report['split_counts']['train']} val={report['split_counts']['val']} test={report['split_counts']['test']}")
    print(f"[PREP] report={Path(args.out_dir) / 'prepare_report.json'}")


if __name__ == "__main__":
    main()

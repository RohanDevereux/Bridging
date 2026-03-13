from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bridging.MD.paths import PDB_CACHE_DIR
from bridging.utils.dataset_rows import parse_chain_group, row_chain_groups

from .ids import canonical_complex_id

SUBGROUP_ORDER = ("antibody_antigen", "tcr_pmhc", "other")
PPB_INTERFACE_CA_CUTOFF_ANGSTROM = 10.0
PPB_INTERFACE_PATCH_SIZE = 128
SUPPORTED_INTERFACE_POLICIES = ("ppb10_patch", "closest_pair_patch")
DEFAULT_INTERFACE_POLICY = "closest_pair_patch"


def normalize_subgroup_label(value) -> str:
    text = "" if value is None else str(value).strip().lower()
    if "antibody" in text:
        return "antibody_antigen"
    if "tcr" in text or "pmhc" in text:
        return "tcr_pmhc"
    return "other"


def load_complex_metadata(dataset_csv: Path) -> dict[str, dict]:
    df = pd.read_csv(dataset_csv)
    metadata: dict[str, dict] = {}
    for row in df.to_dict("records"):
        complex_id = canonical_complex_id(row)
        if not complex_id or complex_id in metadata:
            continue
        left_raw, right_raw = row_chain_groups(row)
        subgroup_raw = None
        for key, value in row.items():
            if str(key).strip().lower() == "subgroup":
                subgroup_raw = value
                break
        metadata[complex_id] = {
            "chains_1": tuple(parse_chain_group(left_raw)),
            "chains_2": tuple(parse_chain_group(right_raw)),
            "subgroup": normalize_subgroup_label(subgroup_raw),
        }
    return metadata


def subgroup_map_from_metadata(metadata: dict[str, dict]) -> dict[str, str]:
    return {cid: str(meta.get("subgroup", "other")) for cid, meta in metadata.items()}


def _partner_labels(node_chain_id: list[str], meta: dict) -> np.ndarray:
    left = {str(c).strip().upper() for c in meta.get("chains_1", ())}
    right = {str(c).strip().upper() for c in meta.get("chains_2", ())}
    labels = np.full((len(node_chain_id),), -1, dtype=np.int64)
    for i, chain in enumerate(node_chain_id):
        chain_norm = str(chain).strip().upper()
        in_left = chain_norm in left
        in_right = chain_norm in right
        if in_left and not in_right:
            labels[i] = 0
        elif in_right and not in_left:
            labels[i] = 1
    return labels


def _load_residue_anchor_coords_by_chain_resseq(
    pdb_path: Path,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], np.ndarray]]:
    ca_coords: dict[tuple[str, int], np.ndarray] = {}
    anchor_coords: dict[tuple[str, int], np.ndarray] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in {"CA", "CB"}:
                continue
            chain = (line[21].strip() or " ").upper()
            resseq_text = line[22:26].strip()
            if not resseq_text:
                continue
            try:
                resseq = int(resseq_text)
                xyz = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=np.float32,
                )
            except Exception:
                continue
            key = (chain, resseq)
            if atom_name == "CA":
                if key not in ca_coords:
                    ca_coords[key] = xyz
                if key not in anchor_coords:
                    anchor_coords[key] = xyz
            elif atom_name == "CB":
                anchor_coords[key] = xyz
    return ca_coords, anchor_coords


def _graph_inter_partner_edge_mask(edge_index: torch.Tensor, partner_labels: np.ndarray) -> np.ndarray:
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    return (
        (partner_labels[src] >= 0)
        & (partner_labels[dst] >= 0)
        & (partner_labels[src] != partner_labels[dst])
    )


def _mapped_partner_residue_coords(
    record: dict,
    *,
    partner_labels: np.ndarray,
    pdb_cache_root: Path,
) -> tuple[dict | None, dict]:
    pdb_id = str(record.get("pdb_id", "")).strip().upper()
    if not pdb_id:
        return None, {"status": "missing_pdb_id"}

    raw_pdb_path = pdb_cache_root / f"{pdb_id}.pdb"
    if not raw_pdb_path.exists():
        return None, {"status": "missing_raw_pdb", "raw_pdb_path": str(raw_pdb_path)}

    ca_lookup, anchor_lookup = _load_residue_anchor_coords_by_chain_resseq(raw_pdb_path)
    node_chain_id = list(record["node_chain_id"])
    node_position = [int(x) for x in record["node_position"]]
    node_ca_xyz = np.full((len(node_chain_id), 3), np.nan, dtype=np.float32)
    node_anchor_xyz = np.full((len(node_chain_id), 3), np.nan, dtype=np.float32)
    node_has_ca = np.zeros((len(node_chain_id),), dtype=bool)
    node_has_anchor = np.zeros((len(node_chain_id),), dtype=bool)

    for i, (chain_id, resseq) in enumerate(zip(node_chain_id, node_position)):
        if partner_labels[i] < 0:
            continue
        key = (str(chain_id).strip().upper(), int(resseq))
        ca_xyz = ca_lookup.get(key)
        anchor_xyz = anchor_lookup.get(key)
        if ca_xyz is not None:
            node_ca_xyz[i] = ca_xyz
            node_has_ca[i] = True
        if anchor_xyz is not None:
            node_anchor_xyz[i] = anchor_xyz
            node_has_anchor[i] = True

    left_idx = np.flatnonzero((partner_labels == 0) & node_has_ca)
    right_idx = np.flatnonzero((partner_labels == 1) & node_has_ca)
    info = {
        "raw_pdb_path": str(raw_pdb_path),
        "n_mapped_partner_1_ca_nodes": int(left_idx.size),
        "n_mapped_partner_2_ca_nodes": int(right_idx.size),
        "n_mapped_anchor_nodes": int(np.sum((partner_labels >= 0) & node_has_anchor)),
    }
    anchor_idx = np.flatnonzero((partner_labels >= 0) & node_has_anchor)

    info = {
        **info,
        "n_partner_1_ca_nodes": int(left_idx.size),
        "n_partner_2_ca_nodes": int(right_idx.size),
        "n_anchor_nodes": int(anchor_idx.size),
    }
    if left_idx.size == 0 or right_idx.size == 0:
        info["status"] = "missing_ca_mapping"
        return None, info

    if anchor_idx.size == 0:
        info["status"] = "missing_anchor_mapping"
        return None, info

    return {
        "node_ca_xyz": node_ca_xyz,
        "node_anchor_xyz": node_anchor_xyz,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "anchor_idx": anchor_idx,
    }, info


def _patch_nodes_from_core(
    *,
    core_nodes: np.ndarray,
    node_anchor_xyz: np.ndarray,
    anchor_idx: np.ndarray,
) -> np.ndarray:
    core_xyz = node_anchor_xyz[core_nodes]
    patch_xyz = node_anchor_xyz[anchor_idx]
    patch_diff = patch_xyz[:, None, :] - core_xyz[None, :, :]
    patch_d2 = np.einsum("ijk,ijk->ij", patch_diff, patch_diff, dtype=np.float64)
    patch_dist = np.sqrt(patch_d2, dtype=np.float64)
    patch_order = np.argsort(np.min(patch_dist, axis=1))
    patch_size = min(int(PPB_INTERFACE_PATCH_SIZE), int(anchor_idx.size))
    return anchor_idx[patch_order[:patch_size]].astype(np.int64, copy=False)


def _ppb_interface_patch_nodes(
    record: dict,
    *,
    partner_labels: np.ndarray,
    pdb_cache_root: Path,
) -> tuple[np.ndarray | None, dict]:
    mapped, info = _mapped_partner_residue_coords(
        record,
        partner_labels=partner_labels,
        pdb_cache_root=pdb_cache_root,
    )
    if mapped is None:
        return None, info

    node_ca_xyz = mapped["node_ca_xyz"]
    node_anchor_xyz = mapped["node_anchor_xyz"]
    left_idx = mapped["left_idx"]
    right_idx = mapped["right_idx"]
    anchor_idx = mapped["anchor_idx"]

    X = node_ca_xyz[left_idx]
    Y = node_ca_xyz[right_idx]
    diff = X[:, None, :] - Y[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff, dtype=np.float64)
    dist = np.sqrt(d2, dtype=np.float64)
    info["closest_pair_ca_distance_angstrom"] = float(np.min(dist))

    left_core = left_idx[np.min(dist, axis=1) < PPB_INTERFACE_CA_CUTOFF_ANGSTROM]
    right_core = right_idx[np.min(dist, axis=0) < PPB_INTERFACE_CA_CUTOFF_ANGSTROM]
    if left_core.size == 0 and right_core.size == 0:
        info["status"] = "no_ppb_interface_residues"
        info["n_interface_core_nodes"] = 0
        return None, info

    core_nodes = np.unique(np.concatenate([left_core, right_core]).astype(np.int64, copy=False))
    patch_nodes = _patch_nodes_from_core(
        core_nodes=core_nodes,
        node_anchor_xyz=node_anchor_xyz,
        anchor_idx=anchor_idx,
    )

    info["status"] = "ok"
    info["interface_source"] = "ppb10_patch"
    info["n_interface_core_nodes"] = int(core_nodes.size)
    info["n_patch_nodes"] = int(patch_nodes.size)
    info["interface_patch_size_limit"] = int(PPB_INTERFACE_PATCH_SIZE)
    return patch_nodes, info


def _closest_pair_patch_nodes(
    record: dict,
    *,
    partner_labels: np.ndarray,
    pdb_cache_root: Path,
) -> tuple[np.ndarray | None, dict]:
    mapped, info = _mapped_partner_residue_coords(
        record,
        partner_labels=partner_labels,
        pdb_cache_root=pdb_cache_root,
    )
    if mapped is None:
        return None, info

    node_ca_xyz = mapped["node_ca_xyz"]
    node_anchor_xyz = mapped["node_anchor_xyz"]
    left_idx = mapped["left_idx"]
    right_idx = mapped["right_idx"]
    anchor_idx = mapped["anchor_idx"]

    X = node_ca_xyz[left_idx]
    Y = node_ca_xyz[right_idx]
    diff = X[:, None, :] - Y[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff, dtype=np.float64)
    flat_idx = int(np.argmin(d2))
    left_pos, right_pos = np.unravel_index(flat_idx, d2.shape)
    core_nodes = np.asarray([left_idx[left_pos], right_idx[right_pos]], dtype=np.int64)
    patch_nodes = _patch_nodes_from_core(
        core_nodes=core_nodes,
        node_anchor_xyz=node_anchor_xyz,
        anchor_idx=anchor_idx,
    )
    info["status"] = "ok"
    info["interface_source"] = "closest_pair_patch"
    info["n_interface_core_nodes"] = int(core_nodes.size)
    info["n_patch_nodes"] = int(patch_nodes.size)
    info["interface_patch_size_limit"] = int(PPB_INTERFACE_PATCH_SIZE)
    info["closest_pair_ca_distance_angstrom"] = float(np.sqrt(d2[left_pos, right_pos], dtype=np.float64))
    return patch_nodes, info


def _slice_record_to_nodes(
    record: dict,
    *,
    subgroup: str,
    keep_nodes: np.ndarray,
    status: str,
    inter_edge_count: int,
    extra_report: dict,
) -> tuple[dict, dict]:
    edge_index = record["edge_index"].long()
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    keep_idx = np.flatnonzero(keep_nodes)
    edge_keep = keep_nodes[src] & keep_nodes[dst]
    keep_idx_t = torch.as_tensor(keep_idx, dtype=torch.long)
    edge_keep_t = torch.as_tensor(edge_keep, dtype=torch.bool)
    kept_edge_index = edge_index[:, edge_keep_t]
    remap = torch.full((len(keep_nodes),), -1, dtype=torch.long)
    remap[keep_idx_t] = torch.arange(len(keep_idx), dtype=torch.long)
    if kept_edge_index.numel() > 0:
        kept_edge_index = remap[kept_edge_index]

    out = dict(record)
    out["node_features"] = record["node_features"][keep_idx_t].clone()
    out["edge_features"] = record["edge_features"][edge_keep_t].clone()
    out["edge_index"] = kept_edge_index.long().clone()
    out["node_chain_id"] = [record["node_chain_id"][int(i)] for i in keep_idx.tolist()]
    out["node_position"] = [record["node_position"][int(i)] for i in keep_idx.tolist()]
    out["subgroup"] = subgroup
    out["graph_view"] = "interface"
    out["n_nodes_full"] = int(len(record.get("node_chain_id", [])))
    out["n_edges_full"] = int(record["edge_index"].shape[1])
    out["n_nodes_view"] = int(len(keep_idx))
    out["n_edges_view"] = int(kept_edge_index.shape[1])

    report = {
        "complex_id": str(record.get("complex_id")),
        "graph_view": "interface",
        "subgroup": subgroup,
        "status": status,
        "n_nodes_full": int(len(record.get("node_chain_id", []))),
        "n_edges_full": int(record["edge_index"].shape[1]),
        "n_nodes_view": int(len(keep_idx)),
        "n_edges_view": int(kept_edge_index.shape[1]),
        "n_inter_partner_edges_full": int(inter_edge_count),
    }
    report.update(extra_report)
    return out, report


def _failed_view_row(
    record: dict,
    *,
    subgroup: str,
    status: str,
    inter_edge_count: int,
    extra_report: dict,
) -> dict:
    report = {
        "complex_id": str(record.get("complex_id")),
        "graph_view": "interface",
        "subgroup": subgroup,
        "status": status,
        "n_nodes_full": int(len(record.get("node_chain_id", []))),
        "n_edges_full": int(record["edge_index"].shape[1]),
        "n_nodes_view": 0,
        "n_edges_view": 0,
        "n_inter_partner_edges_full": int(inter_edge_count),
    }
    report.update(extra_report)
    return report


def build_graph_view_record(
    record: dict,
    *,
    meta: dict,
    graph_view: str,
    pdb_cache_root: Path | None = None,
    interface_policy: str = DEFAULT_INTERFACE_POLICY,
) -> tuple[dict | None, dict]:
    subgroup = str(meta.get("subgroup", "other"))
    if graph_view == "full":
        out = dict(record)
        out["subgroup"] = subgroup
        out["graph_view"] = "full"
        out["n_nodes_full"] = int(len(record.get("node_chain_id", [])))
        out["n_edges_full"] = int(record["edge_index"].shape[1])
        out["n_nodes_view"] = int(len(record.get("node_chain_id", [])))
        out["n_edges_view"] = int(record["edge_index"].shape[1])
        return out, {
            "complex_id": str(record.get("complex_id")),
            "graph_view": "full",
            "subgroup": subgroup,
            "status": "ok",
            "n_nodes_full": int(len(record.get("node_chain_id", []))),
            "n_edges_full": int(record["edge_index"].shape[1]),
            "n_nodes_view": int(len(record.get("node_chain_id", []))),
            "n_edges_view": int(record["edge_index"].shape[1]),
            "n_inter_partner_edges_full": -1,
        }
    if graph_view != "interface":
        raise ValueError(f"Unsupported graph_view={graph_view}")
    if interface_policy not in SUPPORTED_INTERFACE_POLICIES:
        raise ValueError(f"Unsupported interface_policy={interface_policy}")

    node_chain_id = list(record["node_chain_id"])
    partner_labels = _partner_labels(node_chain_id, meta)
    edge_index = record["edge_index"].long()
    inter_mask = _graph_inter_partner_edge_mask(edge_index, partner_labels)
    inter_edge_count = int(np.sum(inter_mask))
    cache_root = Path(PDB_CACHE_DIR if pdb_cache_root is None else pdb_cache_root).expanduser()
    if interface_policy == "ppb10_patch":
        patch_nodes, patch_info = _ppb_interface_patch_nodes(
            record,
            partner_labels=partner_labels,
            pdb_cache_root=cache_root,
        )
    else:
        patch_nodes, patch_info = _closest_pair_patch_nodes(
            record,
            partner_labels=partner_labels,
            pdb_cache_root=cache_root,
        )
    patch_info = dict(patch_info)
    status = str(patch_info.pop("status", "ok"))
    patch_info["interface_policy"] = interface_policy
    if patch_nodes is None or patch_nodes.size == 0:
        return None, _failed_view_row(
            record,
            subgroup=subgroup,
            status=status,
            inter_edge_count=inter_edge_count,
            extra_report=patch_info,
        )

    keep_nodes = np.zeros((len(node_chain_id),), dtype=bool)
    keep_nodes[np.asarray(patch_nodes, dtype=np.int64)] = True
    return _slice_record_to_nodes(
        record,
        subgroup=subgroup,
        keep_nodes=keep_nodes,
        status=status,
        inter_edge_count=inter_edge_count,
        extra_report=patch_info,
    )


def materialize_graph_view_records(
    *,
    records_path: Path,
    dataset_csv: Path,
    graph_view: str,
    out_dir: Path,
    pdb_cache_root: Path | None = None,
    allowed_complex_ids: set[str] | None = None,
    interface_policy: str = DEFAULT_INTERFACE_POLICY,
) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_complex_metadata(dataset_csv)
    records = torch.load(records_path, map_location="cpu")
    transformed: list[dict] = []
    report_rows: list[dict] = []
    missing_meta: list[str] = []
    skipped_by_subset = 0

    for record in records:
        complex_id = str(record.get("complex_id"))
        if allowed_complex_ids is not None and complex_id not in allowed_complex_ids:
            skipped_by_subset += 1
            continue
        meta = metadata.get(complex_id)
        if meta is None:
            missing_meta.append(complex_id)
            continue
        viewed, row = build_graph_view_record(
            record,
            meta=meta,
            graph_view=graph_view,
            pdb_cache_root=pdb_cache_root,
            interface_policy=interface_policy,
        )
        report_rows.append(row)
        if viewed is not None:
            transformed.append(viewed)

    out_path = out_dir / f"graph_records_{graph_view}.pt"
    report_path = out_dir / f"graph_view_{graph_view}_report.json"
    torch.save(transformed, out_path)

    report = {
        "graph_view": graph_view,
        "interface_policy": interface_policy,
        "records_in": int(len(records)),
        "records_considered": int(len(records) - skipped_by_subset),
        "records_out": int(len(transformed)),
        "n_skipped_by_subset": int(skipped_by_subset),
        "missing_metadata": missing_meta[:100],
        "n_missing_metadata": int(len(missing_meta)),
        "n_failed_view": int(sum(1 for row in report_rows if row.get("status") != "ok")),
        "retained_complex_ids": [str(rec.get("complex_id")) for rec in transformed],
        "report_rows": report_rows[:200],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path, report

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bridging.utils.dataset_rows import parse_chain_group, row_chain_groups

from .ids import canonical_complex_id

SUBGROUP_ORDER = ("antibody_antigen", "tcr_pmhc", "other")


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


def _empty_interface_metrics(record: dict, subgroup: str) -> dict:
    return {
        "complex_id": str(record.get("complex_id")),
        "graph_view": "interface",
        "subgroup": subgroup,
        "status": "no_interface_edges",
        "n_nodes_full": int(len(record.get("node_chain_id", []))),
        "n_edges_full": int(record["edge_index"].shape[1]),
        "n_nodes_view": 0,
        "n_edges_view": 0,
        "n_inter_partner_edges_full": 0,
    }


def build_graph_view_record(record: dict, *, meta: dict, graph_view: str) -> tuple[dict | None, dict]:
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

    node_chain_id = list(record["node_chain_id"])
    partner_labels = _partner_labels(node_chain_id, meta)
    edge_index = record["edge_index"].long()
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    inter_mask = (
        (partner_labels[src] >= 0)
        & (partner_labels[dst] >= 0)
        & (partner_labels[src] != partner_labels[dst])
    )
    inter_edge_count = int(np.sum(inter_mask))
    if inter_edge_count < 1:
        return None, _empty_interface_metrics(record, subgroup=subgroup)

    keep_nodes = np.zeros((len(node_chain_id),), dtype=bool)
    keep_nodes[src[inter_mask]] = True
    keep_nodes[dst[inter_mask]] = True
    keep_idx = np.flatnonzero(keep_nodes)
    edge_keep = keep_nodes[src] & keep_nodes[dst]
    keep_idx_t = torch.as_tensor(keep_idx, dtype=torch.long)
    edge_keep_t = torch.as_tensor(edge_keep, dtype=torch.bool)
    kept_edge_index = edge_index[:, edge_keep_t]
    remap = torch.full((len(node_chain_id),), -1, dtype=torch.long)
    remap[keep_idx_t] = torch.arange(len(keep_idx), dtype=torch.long)
    kept_edge_index = remap[kept_edge_index]

    out = dict(record)
    out["node_features"] = record["node_features"][keep_idx_t].clone()
    out["edge_features"] = record["edge_features"][edge_keep_t].clone()
    out["edge_index"] = kept_edge_index.long().clone()
    out["node_chain_id"] = [record["node_chain_id"][int(i)] for i in keep_idx.tolist()]
    out["node_position"] = [record["node_position"][int(i)] for i in keep_idx.tolist()]
    out["subgroup"] = subgroup
    out["graph_view"] = "interface"
    out["n_nodes_full"] = int(len(node_chain_id))
    out["n_edges_full"] = int(record["edge_index"].shape[1])
    out["n_nodes_view"] = int(len(keep_idx))
    out["n_edges_view"] = int(kept_edge_index.shape[1])

    return out, {
        "complex_id": str(record.get("complex_id")),
        "graph_view": "interface",
        "subgroup": subgroup,
        "status": "ok",
        "n_nodes_full": int(len(node_chain_id)),
        "n_edges_full": int(record["edge_index"].shape[1]),
        "n_nodes_view": int(len(keep_idx)),
        "n_edges_view": int(kept_edge_index.shape[1]),
        "n_inter_partner_edges_full": inter_edge_count,
    }


def materialize_graph_view_records(
    *,
    records_path: Path,
    dataset_csv: Path,
    graph_view: str,
    out_dir: Path,
) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_complex_metadata(dataset_csv)
    records = torch.load(records_path, map_location="cpu")
    transformed: list[dict] = []
    report_rows: list[dict] = []
    missing_meta: list[str] = []

    for record in records:
        complex_id = str(record.get("complex_id"))
        meta = metadata.get(complex_id)
        if meta is None:
            missing_meta.append(complex_id)
            continue
        viewed, row = build_graph_view_record(record, meta=meta, graph_view=graph_view)
        report_rows.append(row)
        if viewed is not None:
            transformed.append(viewed)

    out_path = out_dir / f"graph_records_{graph_view}.pt"
    report_path = out_dir / f"graph_view_{graph_view}_report.json"
    torch.save(transformed, out_path)

    report = {
        "graph_view": graph_view,
        "records_in": int(len(records)),
        "records_out": int(len(transformed)),
        "missing_metadata": missing_meta[:100],
        "n_missing_metadata": int(len(missing_meta)),
        "n_failed_view": int(sum(1 for row in report_rows if row.get("status") != "ok")),
        "report_rows": report_rows[:200],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path, report

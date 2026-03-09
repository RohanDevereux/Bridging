from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from deeprank2.features import components, contact, exposure, surfacearea
from deeprank2.query import ProteinProteinInterfaceQuery, Query, QueryCollection
from deeprank2.utils.buildgraph import get_structure
from deeprank2.utils.graph import Graph
from pdb2sql import pdb2sql as pdb2sql_object

from .ids import sanitize_filename_token


def _as_path_list(paths: Any) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def _decode_chain_id(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip().upper()
    return str(x).strip().upper()


def _decode_name(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip()
    return str(x).strip()


def _parse_resseq_from_node_name(name: str) -> int:
    parts = str(name).strip().split()
    if len(parts) < 3:
        raise ValueError(f"Cannot parse DeepRank node name: {name!r}")
    last = parts[-1]
    m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", last)
    if m is None:
        raise ValueError(
            "Cannot parse residue number from DeepRank node name "
            f"{name!r}; expected trailing token like '<resseq>' or '<resseq><icode>'."
        )
    return int(m.group(1))


def _read_id_column(group: h5py.Group, name: str) -> list:
    if name not in group:
        raise KeyError(f"Missing required metadata '{name}' in HDF5 group '{group.name}'.")
    arr = np.asarray(group[name])
    if arr.ndim == 1:
        return arr.tolist()
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].tolist()
    raise ValueError(f"Unexpected shape for metadata '{name}' in '{group.name}': {arr.shape}")


def _read_feature_column(group: h5py.Group, name: str, n_rows: int) -> np.ndarray:
    m = re.fullmatch(r"([A-Za-z0-9]+)_(\d+)", name)
    base_name = m.group(1) if m is not None else name
    component = int(m.group(2)) if m is not None else None

    if base_name not in group:
        raise KeyError(f"Missing required feature '{base_name}' in HDF5 group '{group.name}' (requested '{name}').")
    arr = np.asarray(group[base_name])
    if arr.ndim == 1:
        if component is not None:
            raise ValueError(
                f"Feature '{base_name}' in '{group.name}' is 1D but component '{name}' was requested."
            )
    elif arr.ndim == 2:
        if component is None:
            if arr.shape[1] != 1:
                raise ValueError(
                    f"Feature '{base_name}' in '{group.name}' has shape={arr.shape}. "
                    "Request a specific component like '<name>_0'."
                )
            arr = arr[:, 0]
        else:
            if component >= arr.shape[1]:
                raise ValueError(
                    f"Requested component '{name}' is out of range for feature '{base_name}' "
                    f"with shape={arr.shape} in '{group.name}'."
                )
            arr = arr[:, component]
    else:
        raise ValueError(
            f"Feature '{base_name}' must be 1D or 2D in '{group.name}', found shape={arr.shape}."
        )
    out = arr.astype(np.float32, copy=False)
    if out.shape[0] != n_rows:
        raise ValueError(f"Feature {name} has incompatible length {out.shape[0]} != {n_rows}")
    return out


def _entry_model_id(entry_name: str) -> str:
    token = str(entry_name).split(":")[-1]
    return token.strip()


def stage_query_pdbs(entries: list[dict], pdb_stage_dir: Path, overwrite: bool = False) -> list[dict]:
    pdb_stage_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for row in entries:
        staged = pdb_stage_dir / f"{sanitize_filename_token(row['complex_id'])}.pdb"
        source_path = Path(row.get("query_source_pdb_path", row["pdb_path"]))
        if overwrite or not staged.exists():
            shutil.copy2(source_path, staged)
        rec = dict(row)
        rec["query_pdb_path"] = str(staged)
        out.append(rec)
    return out


class ProteinComplexQuery(Query):
    def get_query_id(self) -> str:
        return f"{self.resolution}-complex:{self.model_id}"

    def _build_helper(self) -> Graph:
        pdb = pdb2sql_object(self.pdb_path)
        try:
            structure = get_structure(pdb, self.model_id)
        finally:
            pdb._close()

        residues = []
        for chain in structure.chains:
            for residue in chain.residues:
                if residue.amino_acid is None:
                    continue
                if len(residue.atoms) < 1:
                    continue
                residues.append(residue)
        if not residues:
            raise ValueError(f"No protein residues found for full-complex graph: {self.pdb_path}")

        max_edge = float(self.max_edge_length) if self.max_edge_length is not None else 10.0
        graph = Graph.build_graph(nodes=residues, graph_id=self.get_query_id(), max_edge_length=max_edge)
        if len(graph.edges) < 1:
            raise ValueError(f"Graph has zero edges for full-complex query: {self.pdb_path}")

        all_atoms = [atom for residue in residues for atom in residue.atoms]
        graph.center = np.mean([atom.position for atom in all_atoms], axis=0)
        return graph


def build_deeprank_hdf5(
    *,
    staged_entries: list[dict],
    out_prefix: Path,
    influence_radius: float,
    max_edge_length: float | None,
    query_mode: str,
    cpu_count: int | None = None,
) -> list[Path]:
    if not staged_entries:
        raise ValueError("No staged entries available for DeepRank2 graph generation.")
    if shutil.which("msms") is None:
        raise RuntimeError(
            "msms executable not found on PATH. DeepRank exposure features (res_depth/hse) require msms."
        )
    if query_mode not in {"ppi", "full_complex"}:
        raise ValueError(f"Unsupported DeepRank query_mode={query_mode}. Use 'ppi' or 'full_complex'.")
    # DeepRank emits one warning per unknown atom for forcefield-derived terms.
    # This can produce multi-million-line logs on PPB and hide real failures.
    logging.getLogger("deeprank2.utils.parsing").setLevel(logging.ERROR)
    feature_modules = [components, contact, exposure, surfacearea]
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    query_collection = QueryCollection()
    for rec in staged_entries:
        if query_mode == "ppi":
            chain_ids = [rec["query_chain_1"], rec["query_chain_2"]]
            query = ProteinProteinInterfaceQuery(
                pdb_path=rec["query_pdb_path"],
                resolution="residue",
                chain_ids=chain_ids,
                targets={"dG": float(rec["dG"])},
                influence_radius=float(influence_radius),
                max_edge_length=(float(max_edge_length) if max_edge_length is not None else None),
            )
        else:
            query = ProteinComplexQuery(
                pdb_path=rec["query_pdb_path"],
                resolution="residue",
                chain_ids=[],
                targets={"dG": float(rec["dG"])},
                influence_radius=float(influence_radius),
                max_edge_length=(float(max_edge_length) if max_edge_length is not None else None),
            )
        query_collection.add(query)

    hdf5_paths = query_collection.process(
        prefix=str(out_prefix),
        feature_modules=feature_modules,
        cpu_count=cpu_count,
        log_error_traceback=True,
    )
    return _as_path_list(hdf5_paths)


def index_hdf5_entries(hdf5_paths: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in hdf5_paths:
        with h5py.File(path, "r") as h5:
            for key in h5.keys():
                model_id = _entry_model_id(key)
                if model_id not in out:
                    out[model_id] = {"hdf5_path": str(path), "entry_name": str(key)}
    return out


def load_deeprank_graph(
    *,
    hdf5_path: Path,
    entry_name: str,
    node_feature_names: list[str],
    edge_feature_names: list[str],
) -> dict:
    with h5py.File(hdf5_path, "r") as h5:
        entry = h5[entry_name]
        node_group = entry["node_features"]
        edge_group = entry["edge_features"]

        chain_vals = _read_id_column(node_group, "_chain_id")
        name_vals = _read_id_column(node_group, "_name")
        if len(chain_vals) != len(name_vals):
            raise ValueError(
                f"Node metadata length mismatch in '{entry_name}': "
                f"_chain_id={len(chain_vals)} _name={len(name_vals)}"
            )
        n_nodes = int(len(chain_vals))

        node_chain_id = [_decode_chain_id(v) for v in chain_vals]
        node_names = [_decode_name(v) for v in name_vals]
        node_position = [_parse_resseq_from_node_name(name) for name in node_names]

        edge_index_raw = np.asarray(edge_group["_index"]).astype(np.int64)
        if edge_index_raw.ndim != 2:
            raise ValueError(f"Unexpected edge index shape: {edge_index_raw.shape}")
        if edge_index_raw.shape[0] == 2:
            edge_index = edge_index_raw
            n_edges = int(edge_index_raw.shape[1])
        elif edge_index_raw.shape[1] == 2:
            edge_index = edge_index_raw.T
            n_edges = int(edge_index_raw.shape[0])
        else:
            raise ValueError(f"Unexpected edge index shape: {edge_index_raw.shape}")

        node_cols = [_read_feature_column(node_group, name, n_nodes) for name in node_feature_names]
        edge_cols = [_read_feature_column(edge_group, name, n_edges) for name in edge_feature_names]

        node_features = np.stack(node_cols, axis=1).astype(np.float32, copy=False)
        edge_features = np.stack(edge_cols, axis=1).astype(np.float32, copy=False)

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_index": edge_index.astype(np.int64, copy=False),
        "node_chain_id": node_chain_id,
        "node_position": node_position,
    }


def write_deeprank_index(index: dict[str, dict], out_path: Path) -> None:
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

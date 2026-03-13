from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .config import (
    DYNAMIC_EDGE_FEATURES_BASE,
    DYNAMIC_EDGE_FEATURES_WITH_DIST,
    DYNAMIC_NODE_INPUT_FEATURES,
    DYNAMIC_NODE_MASK_TARGETS,
    DYNAMIC_NODE_FEATURES,
    NODE_IDENTITY_FEATURES,
    STATIC_EDGE_MASK_TARGETS,
    STATIC_EDGE_FEATURES,
    STATIC_NODE_MASK_TARGETS,
    STATIC_NODE_FEATURES,
)

SUPPORTED_TARGET_POLICIES = (
    "shared_static",
    "shared_all",
    "all_inputs",
    "all_nonidentity",
    "mode_specific",
    "sd_dynamic_all",
    "sd_static_plus_dynamic_all",
    "sd_static_plus_dynamic_nodes",
    "sd_static_plus_dynamic_edges",
)
TRIVIAL_EDGE_TARGET_EXCLUDE = {"same_chain"}


@dataclass(frozen=True)
class FeatureSpec:
    mode: str
    target_policy: str
    node_input_names: list[str]
    edge_input_names: list[str]
    node_target_names: list[str]
    edge_target_names: list[str]
    node_target_idx: list[int]
    edge_target_idx: list[int]


def _idx(names: list[str], wanted: list[str]) -> list[int]:
    lookup = {n: i for i, n in enumerate(names)}
    out = [lookup[w] for w in wanted if w in lookup]
    if not out:
        raise ValueError(f"No requested features found. wanted={wanted}")
    return out


def _available(all_names: list[str], wanted: list[str] | tuple[str, ...]) -> list[str]:
    return [name for name in wanted if name in all_names]


def _dedupe(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _all_nonidentity_node_targets(node_input_names: list[str]) -> list[str]:
    return [name for name in node_input_names if name not in NODE_IDENTITY_FEATURES]


def _all_nontrivial_edge_targets(edge_input_names: list[str]) -> list[str]:
    return [name for name in edge_input_names if name not in TRIVIAL_EDGE_TARGET_EXCLUDE]


def _resolve_target_names(
    *,
    mode: str,
    target_policy: str,
    node_input_names: list[str],
    edge_input_names: list[str],
    dynamic_edge_names: list[str],
) -> tuple[list[str], list[str]]:
    static_node_subset = _available(node_input_names, STATIC_NODE_MASK_TARGETS)
    static_edge_subset = _available(edge_input_names, STATIC_EDGE_MASK_TARGETS)
    static_node_all = _available(node_input_names, STATIC_NODE_FEATURES)
    static_edge_all = _available(edge_input_names, STATIC_EDGE_FEATURES)
    dynamic_node_observed = _available(node_input_names, DYNAMIC_NODE_MASK_TARGETS)
    dynamic_node_all = _available(node_input_names, DYNAMIC_NODE_INPUT_FEATURES)
    dynamic_edge_all = list(dynamic_edge_names)

    if target_policy == "shared_static":
        return static_node_subset, static_edge_subset
    if target_policy == "shared_all":
        return static_node_all, static_edge_all
    if target_policy == "all_inputs":
        return list(node_input_names), list(edge_input_names)
    if target_policy == "all_nonidentity":
        return _all_nonidentity_node_targets(node_input_names), _all_nontrivial_edge_targets(edge_input_names)
    if target_policy == "mode_specific":
        if mode == "S":
            return static_node_subset, static_edge_subset
        return dynamic_node_observed, dynamic_edge_all
    if target_policy == "sd_dynamic_all":
        if mode == "S":
            return static_node_subset, static_edge_subset
        return dynamic_node_all, dynamic_edge_all
    if target_policy == "sd_static_plus_dynamic_all":
        if mode == "S":
            return static_node_subset, static_edge_subset
        return _dedupe(static_node_subset + dynamic_node_all), _dedupe(static_edge_subset + dynamic_edge_all)
    if target_policy == "sd_static_plus_dynamic_nodes":
        if mode == "S":
            return static_node_subset, static_edge_subset
        return _dedupe(static_node_subset + dynamic_node_all), static_edge_subset
    if target_policy == "sd_static_plus_dynamic_edges":
        if mode == "S":
            return static_node_subset, static_edge_subset
        return static_node_subset, _dedupe(static_edge_subset + dynamic_edge_all)
    raise ValueError(f"Unsupported target_policy={target_policy}")


def build_feature_spec(records: list[dict], mode: str, target_policy: str = "shared_static") -> FeatureSpec:
    if mode not in {"S", "SD"}:
        raise ValueError("mode must be S or SD")
    if target_policy not in SUPPORTED_TARGET_POLICIES:
        raise ValueError(f"Unsupported target_policy={target_policy}")
    if not records:
        raise ValueError("No records found.")

    node_names_all = list(records[0]["node_feature_names"])
    edge_names_all = list(records[0]["edge_feature_names"])

    if mode == "S":
        node_input_names = [n for n in STATIC_NODE_FEATURES if n in node_names_all]
        edge_input_names = [n for n in STATIC_EDGE_FEATURES if n in edge_names_all]
        dyn_edge = []
    else:
        dyn_edge = [n for n in DYNAMIC_EDGE_FEATURES_WITH_DIST if n in edge_names_all]
        if not dyn_edge:
            dyn_edge = [n for n in DYNAMIC_EDGE_FEATURES_BASE if n in edge_names_all]
        node_input_names = [n for n in (list(STATIC_NODE_FEATURES) + list(DYNAMIC_NODE_INPUT_FEATURES)) if n in node_names_all]
        edge_input_names = [n for n in (list(STATIC_EDGE_FEATURES) + dyn_edge) if n in edge_names_all]

    node_target_names, edge_target_names = _resolve_target_names(
        mode=mode,
        target_policy=target_policy,
        node_input_names=node_input_names,
        edge_input_names=edge_input_names,
        dynamic_edge_names=dyn_edge,
    )

    node_target_idx = _idx(node_input_names, node_target_names)
    edge_target_idx = _idx(edge_input_names, edge_target_names)
    return FeatureSpec(
        mode=mode,
        target_policy=target_policy,
        node_input_names=node_input_names,
        edge_input_names=edge_input_names,
        node_target_names=node_target_names,
        edge_target_names=edge_target_names,
        node_target_idx=node_target_idx,
        edge_target_idx=edge_target_idx,
    )


class FeatureStandardizer:
    def __init__(
        self,
        node_mean,
        node_std,
        edge_mean,
        edge_std,
        *,
        node_lower=None,
        node_upper=None,
        edge_lower=None,
        edge_upper=None,
        z_clip: float = 8.0,
    ):
        self.node_mean = np.asarray(node_mean, dtype=np.float32)
        self.node_std = np.asarray(node_std, dtype=np.float32)
        self.edge_mean = np.asarray(edge_mean, dtype=np.float32)
        self.edge_std = np.asarray(edge_std, dtype=np.float32)
        self.node_lower = np.asarray(
            self.node_mean if node_lower is None else node_lower,
            dtype=np.float32,
        )
        self.node_upper = np.asarray(
            self.node_mean if node_upper is None else node_upper,
            dtype=np.float32,
        )
        self.edge_lower = np.asarray(
            self.edge_mean if edge_lower is None else edge_lower,
            dtype=np.float32,
        )
        self.edge_upper = np.asarray(
            self.edge_mean if edge_upper is None else edge_upper,
            dtype=np.float32,
        )
        self.z_clip = float(z_clip)

    @classmethod
    def fit(cls, records: list[dict], spec: FeatureSpec) -> "FeatureStandardizer":
        node_cols = []
        edge_cols = []
        for rec in records:
            if rec["split"] != "train":
                continue
            node_df = _feature_matrix(rec["node_features"], rec["node_feature_names"], spec.node_input_names)
            edge_df = _feature_matrix(rec["edge_features"], rec["edge_feature_names"], spec.edge_input_names)
            node_cols.append(node_df)
            edge_cols.append(edge_df)
        if not node_cols or not edge_cols:
            raise ValueError("No train rows found for standardizer fitting.")
        node_cat = np.concatenate(node_cols, axis=0)
        edge_cat = np.concatenate(edge_cols, axis=0)
        node_mean = np.nanmean(node_cat, axis=0)
        edge_mean = np.nanmean(edge_cat, axis=0)
        node_std = np.nanstd(node_cat, axis=0)
        edge_std = np.nanstd(edge_cat, axis=0)
        node_lower = np.nanpercentile(node_cat, 0.5, axis=0)
        node_upper = np.nanpercentile(node_cat, 99.5, axis=0)
        edge_lower = np.nanpercentile(edge_cat, 0.5, axis=0)
        edge_upper = np.nanpercentile(edge_cat, 99.5, axis=0)
        node_std[node_std < 1e-6] = 1.0
        edge_std[edge_std < 1e-6] = 1.0
        return cls(
            node_mean=node_mean,
            node_std=node_std,
            edge_mean=edge_mean,
            edge_std=edge_std,
            node_lower=node_lower,
            node_upper=node_upper,
            edge_lower=edge_lower,
            edge_upper=edge_upper,
        )

    def transform_node(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, self.node_lower[None, :], self.node_upper[None, :])
        z = (clipped - self.node_mean[None, :]) / self.node_std[None, :]
        z = np.clip(z, -self.z_clip, self.z_clip)
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def transform_edge(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, self.edge_lower[None, :], self.edge_upper[None, :])
        z = (clipped - self.edge_mean[None, :]) / self.edge_std[None, :]
        z = np.clip(z, -self.z_clip, self.z_clip)
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def to_dict(self) -> dict:
        return {
            "node_mean": self.node_mean.tolist(),
            "node_std": self.node_std.tolist(),
            "edge_mean": self.edge_mean.tolist(),
            "edge_std": self.edge_std.tolist(),
            "node_lower": self.node_lower.tolist(),
            "node_upper": self.node_upper.tolist(),
            "edge_lower": self.edge_lower.tolist(),
            "edge_upper": self.edge_upper.tolist(),
            "z_clip": float(self.z_clip),
        }


def _feature_matrix(tensor_or_array, all_names: list[str], selected_names: list[str]) -> np.ndarray:
    arr = tensor_or_array.detach().cpu().numpy() if torch.is_tensor(tensor_or_array) else np.asarray(tensor_or_array)
    lookup = {n: i for i, n in enumerate(all_names)}
    cols = [lookup[n] for n in selected_names]
    return np.asarray(arr[:, cols], dtype=np.float32)


class PreparedGraphDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        *,
        split: str,
        spec: FeatureSpec,
        scaler: FeatureStandardizer,
    ):
        self.records = [r for r in records if r["split"] == split]
        self.spec = spec
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Data:
        rec = self.records[idx]
        x_raw = _feature_matrix(rec["node_features"], rec["node_feature_names"], self.spec.node_input_names)
        e_raw = _feature_matrix(rec["edge_features"], rec["edge_feature_names"], self.spec.edge_input_names)
        x = torch.as_tensor(self.scaler.transform_node(x_raw), dtype=torch.float32)
        edge_attr = torch.as_tensor(self.scaler.transform_edge(e_raw), dtype=torch.float32)
        edge_index = rec["edge_index"].long()
        y = torch.as_tensor([float(rec["dG"])], dtype=torch.float32)
        graph_index = torch.as_tensor([idx], dtype=torch.long)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            graph_index=graph_index,
        )
        data.complex_id = rec["complex_id"]
        return data

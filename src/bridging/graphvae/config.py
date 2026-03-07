from __future__ import annotations

from dataclasses import dataclass


DEEPRANK_NODE_FEATURES = ("bsa", "sasa", "res_depth", "hse")
STATIC_NODE_FEATURES = DEEPRANK_NODE_FEATURES + ("n_same_chain_8A", "n_other_chain_8A")
STATIC_EDGE_FEATURES = ("distance", "same_chain", "electrostatic", "vanderwaals")

DYNAMIC_NODE_FEATURES = (
    "dyn_log1p_water_count_5A_mean",
    "dyn_water_count_5A_std",
    "dyn_rmsf_ca",
)
DYNAMIC_EDGE_FEATURES_BASE = ("dyn_contact_freq_8A",)
DYNAMIC_EDGE_FEATURES_WITH_DIST = ("dyn_contact_freq_8A", "dyn_dist_mean", "dyn_dist_std")


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    latent_dim: int = 8
    hidden_dim: int = 128
    num_layers: int = 3
    mask_ratio: float = 0.30
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    max_epochs: int = 200
    patience: int = 25
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_anneal_fraction: float = 0.30
    corr_weight: float = 0.01
    seed: int = 2026

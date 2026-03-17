from __future__ import annotations

from dataclasses import dataclass


DEEPRANK_NODE_FEATURES = ("bsa", "sasa", "res_depth", "hse_0", "hse_1", "hse_2")
STATIC_EDGE_FEATURES = ("distance", "same_chain", "electrostatic", "vanderwaals")

AA_CODES = ("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")
AA_ONEHOT_FEATURES = tuple(f"aa_{aa}" for aa in AA_CODES)
AA_TYPE_FEATURES = ("aa_type_nonpolar", "aa_type_polar", "aa_type_charged")
PHYS_NODE_FEATURES = (
    "phys_sidechain_size",
    "phys_net_charge",
    "phys_hydrophobicity",
    "phys_alpha_propensity",
    "phys_beta_propensity",
)
TERMINUS_FEATURES = ("is_n_terminus", "is_c_terminus")
NODE_IDENTITY_FEATURES = AA_ONEHOT_FEATURES + AA_TYPE_FEATURES + PHYS_NODE_FEATURES + TERMINUS_FEATURES

STATIC_NODE_FEATURES = DEEPRANK_NODE_FEATURES + ("n_same_chain_8A", "n_other_chain_8A") + NODE_IDENTITY_FEATURES
STATIC_NODE_MASK_TARGETS = DEEPRANK_NODE_FEATURES + ("n_same_chain_8A", "n_other_chain_8A")
STATIC_EDGE_MASK_TARGETS = ("distance",)

DYNAMIC_NODE_FEATURES = (
    "dyn_log1p_water_count_5A_mean",
    "dyn_water_count_5A_std",
    "dyn_rmsf_ca",
)
TORSION_NODE_INPUT_FEATURES = (
    "dyn_sin_phi_mean",
    "dyn_cos_phi_mean",
    "dyn_sin_psi_mean",
    "dyn_cos_psi_mean",
)
FORCE_NODE_INPUT_FEATURES = (
    "dyn_coul_force_mean",
    "dyn_coul_force_std",
    "dyn_lj_force_mean",
    "dyn_lj_force_std",
)
DYNAMIC_NODE_MASK_TARGETS = DYNAMIC_NODE_FEATURES
DYNAMIC_NODE_INPUT_FEATURES = DYNAMIC_NODE_MASK_TARGETS + TORSION_NODE_INPUT_FEATURES + FORCE_NODE_INPUT_FEATURES
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

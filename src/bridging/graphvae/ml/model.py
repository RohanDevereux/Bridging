from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class MaskedGraphVAE(nn.Module):
    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        node_target_idx: list[int],
        edge_target_idx: list[int],
        latent_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        affinity_target_mean: float = 0.0,
        affinity_target_std: float = 1.0,
        logvar_clip: float = 8.0,
    ):
        super().__init__()
        self.node_in_dim = int(node_in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.latent_dim = int(latent_dim)
        self.logvar_clip = float(max(logvar_clip, 1.0))
        self.node_target_idx = sorted(int(i) for i in node_target_idx)
        self.edge_target_idx = sorted(int(i) for i in edge_target_idx)
        self.node_static_idx = [i for i in range(self.node_in_dim) if i not in self.node_target_idx]
        self.edge_static_idx = [i for i in range(self.edge_in_dim) if i not in self.edge_target_idx]

        self.node_target_dim = len(self.node_target_idx)
        self.edge_target_dim = len(self.edge_target_idx)
        if self.node_target_dim < 1 or self.edge_target_dim < 1:
            raise ValueError("At least one node and one edge target feature are required.")

        enc_node_dim = self.node_in_dim + self.node_target_dim
        enc_edge_dim = self.edge_in_dim + self.edge_target_dim

        self.node_proj = nn.Linear(enc_node_dim, hidden_dim)
        self.edge_proj = nn.Linear(enc_edge_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [GINEConv(_make_mlp(hidden_dim, hidden_dim, hidden_dim), edge_dim=hidden_dim) for _ in range(num_layers)]
        )
        self.mu_head = nn.Linear(2 * hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(2 * hidden_dim, latent_dim)
        self.affinity_head = _make_mlp(latent_dim, hidden_dim, 1)

        node_dec_in = latent_dim + len(self.node_static_idx) + 2 * self.node_target_dim
        edge_dec_in = latent_dim + len(self.edge_static_idx) + 2 * self.edge_target_dim
        self.node_decoder = _make_mlp(node_dec_in, hidden_dim, self.node_target_dim)
        self.edge_decoder = _make_mlp(edge_dec_in, hidden_dim, self.edge_target_dim)

        self.register_buffer("_node_target_idx_tensor", torch.tensor(self.node_target_idx, dtype=torch.long))
        self.register_buffer("_edge_target_idx_tensor", torch.tensor(self.edge_target_idx, dtype=torch.long))
        self.register_buffer("_node_static_idx_tensor", torch.tensor(self.node_static_idx, dtype=torch.long))
        self.register_buffer("_edge_static_idx_tensor", torch.tensor(self.edge_static_idx, dtype=torch.long))
        self.register_buffer("_affinity_target_mean", torch.tensor(float(affinity_target_mean), dtype=torch.float32))
        self.register_buffer(
            "_affinity_target_std",
            torch.tensor(max(float(affinity_target_std), 1e-6), dtype=torch.float32),
        )

    def _normalize_affinity(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self._affinity_target_mean) / self._affinity_target_std

    def _denormalize_affinity(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self._affinity_target_std + self._affinity_target_mean

    def _apply_masks(self, x: torch.Tensor, edge_attr: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, ...]:
        node_target = x.index_select(dim=1, index=self._node_target_idx_tensor)
        edge_target = edge_attr.index_select(dim=1, index=self._edge_target_idx_tensor)
        node_mask = torch.rand_like(node_target) < float(mask_ratio)
        edge_mask = torch.rand_like(edge_target) < float(mask_ratio)
        node_target_masked = node_target.masked_fill(node_mask, 0.0)
        edge_target_masked = edge_target.masked_fill(edge_mask, 0.0)

        x_masked = x.clone()
        edge_masked = edge_attr.clone()
        x_masked[:, self._node_target_idx_tensor] = node_target_masked
        edge_masked[:, self._edge_target_idx_tensor] = edge_target_masked
        return x_masked, edge_masked, node_target, edge_target, node_target_masked, edge_target_masked, node_mask, edge_mask

    def _encode(self, x_enc: torch.Tensor, edge_index: torch.Tensor, edge_enc: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.node_proj(x_enc))
        e = F.relu(self.edge_proj(edge_enc))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, e))
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.mu_head(pooled), self.logvar_head(pooled)

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.sum() * 0.0
        selected = (pred - target)[mask]
        if selected.numel() == 0:
            return pred.sum() * 0.0
        return torch.mean(selected * selected)

    @staticmethod
    def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        per_dim = -0.5 * (1.0 + logvar - mu * mu - torch.exp(logvar))
        return torch.mean(torch.mean(per_dim, dim=1))

    @staticmethod
    def _corr_penalty(mu: torch.Tensor) -> torch.Tensor:
        if mu.shape[0] < 2:
            return mu.sum() * 0.0
        centered = mu - torch.mean(mu, dim=0, keepdim=True)
        cov = (centered.T @ centered) / (mu.shape[0] - 1)
        offdiag = cov - torch.diag(torch.diag(cov))
        return torch.mean(offdiag * offdiag)

    def predict_affinity_from_mu(self, mu: torch.Tensor) -> torch.Tensor:
        pred_norm = self.affinity_head(mu).reshape(-1)
        return self._denormalize_affinity(pred_norm)

    def compute_loss(
        self,
        data,
        *,
        mask_ratio: float,
        beta: float,
        corr_weight: float,
        affinity_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor | None]:
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch
        if batch is None:
            batch = x.new_zeros((x.shape[0],), dtype=torch.long)

        (
            x_masked,
            edge_masked,
            node_target,
            edge_target,
            node_target_masked,
            edge_target_masked,
            node_mask,
            edge_mask,
        ) = self._apply_masks(x, edge_attr, mask_ratio=mask_ratio)

        x_enc = torch.cat([x_masked, node_mask.float()], dim=1)
        edge_enc = torch.cat([edge_masked, edge_mask.float()], dim=1)
        mu, logvar = self._encode(x_enc, edge_index, edge_enc, batch=batch)
        logvar = torch.clamp(logvar, min=-self.logvar_clip, max=self.logvar_clip)
        z = self._sample(mu, logvar)

        z_node = z[batch]
        edge_batch = batch[edge_index[0]] if edge_index.numel() > 0 else x.new_zeros((0,), dtype=torch.long)
        z_edge = z[edge_batch] if edge_index.numel() > 0 else edge_attr.new_zeros((0, z.shape[1]))

        node_static = x.index_select(dim=1, index=self._node_static_idx_tensor)
        edge_static = edge_attr.index_select(dim=1, index=self._edge_static_idx_tensor)
        node_dec_in = torch.cat([z_node, node_static, node_target_masked, node_mask.float()], dim=1)
        edge_dec_in = torch.cat([z_edge, edge_static, edge_target_masked, edge_mask.float()], dim=1)

        node_pred = self.node_decoder(node_dec_in)
        edge_pred = self.edge_decoder(edge_dec_in)

        node_recon = self._masked_mse(node_pred, node_target, node_mask)
        edge_recon = self._masked_mse(edge_pred, edge_target, edge_mask)
        recon = node_recon + edge_recon
        kl = self._kl(mu, logvar)
        corr = self._corr_penalty(mu)
        affinity_pred: torch.Tensor | None = None
        affinity_loss = recon.new_zeros(())
        if float(affinity_weight) > 0.0 and hasattr(data, "y") and data.y is not None and data.y.numel() > 0:
            y = data.y.reshape(-1).float()
            pred_norm = self.affinity_head(mu).reshape(-1)
            affinity_pred = self._denormalize_affinity(pred_norm)
            affinity_loss = F.mse_loss(pred_norm, self._normalize_affinity(y))
        total = recon + float(beta) * kl + float(corr_weight) * corr + float(affinity_weight) * affinity_loss

        parts = {
            "loss": float(total.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "node_recon": float(node_recon.detach().cpu()),
            "edge_recon": float(edge_recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "corr": float(corr.detach().cpu()),
            "affinity_loss": float(affinity_loss.detach().cpu()) if math.isfinite(float(affinity_loss.detach().cpu())) else float("nan"),
        }
        return total, parts, mu, affinity_pred

    def encode_mu(self, data) -> torch.Tensor:
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch
        if batch is None:
            batch = x.new_zeros((x.shape[0],), dtype=torch.long)
        node_mask = x.new_zeros((x.shape[0], self.node_target_dim))
        edge_mask = edge_attr.new_zeros((edge_attr.shape[0], self.edge_target_dim))
        x_enc = torch.cat([x, node_mask], dim=1)
        edge_enc = torch.cat([edge_attr, edge_mask], dim=1)
        mu, _ = self._encode(x_enc, edge_index, edge_enc, batch=batch)
        return mu

    def predict_affinity(self, data) -> torch.Tensor:
        mu = self.encode_mu(data)
        return self.predict_affinity_from_mu(mu)

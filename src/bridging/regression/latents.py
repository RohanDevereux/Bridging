from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bridging.ml.model import CVAE


def _resolve_device(device: str | None = None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cvae_checkpoint(checkpoint_path: str | Path, device: str | None = None):
    checkpoint_path = Path(checkpoint_path)
    dev = _resolve_device(device)
    state = torch.load(checkpoint_path, map_location=dev)
    cfg = state["config"]
    model = CVAE(
        in_channels=int(cfg["in_channels"]),
        img_size=int(cfg["img_size"]),
        latent_dim=int(cfg["latent_dim"]),
        base_channels=int(cfg["base_channels"]),
    ).to(dev)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, dev, cfg


@torch.no_grad()
def _encode_mu_batch(model: CVAE, x: torch.Tensor) -> torch.Tensor:
    h = model.enc(x)
    h = h.view(x.size(0), -1)
    cond_vec = model._cond(x.size(0), None)
    if cond_vec is not None:
        h = torch.cat([h, cond_vec], dim=1)
    h = F.leaky_relu(model.fc(h), 0.1, inplace=False)
    return model.mu(h)


@torch.no_grad()
def encode_mu_matrix(model: CVAE, feature_path: str | Path, device, batch_size: int = 256):
    arr = np.load(feature_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"{feature_path} expected shape (T,C,N,N), got {arr.shape}")
    total = int(arr.shape[0])
    chunks = []
    for start in range(0, total, max(1, int(batch_size))):
        stop = min(total, start + max(1, int(batch_size)))
        x = torch.from_numpy(arr[start:stop].astype(np.float32)).to(device, non_blocking=True)
        mu = _encode_mu_batch(model, x)
        chunks.append(mu.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, int(model.latent_dim)), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def mean_std_pool(mu: np.ndarray) -> np.ndarray:
    if mu.ndim != 2:
        raise ValueError(f"mu expected shape (T,d), got {mu.shape}")
    mean = mu.mean(axis=0)
    std = mu.std(axis=0, ddof=0)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def _npz_path(latents_dir: Path, row: dict) -> Path:
    pdb = (row.get("pdb_id") or "UNK").upper()
    idx = int(row["row_index"])
    return latents_dir / f"{idx:04d}_{pdb}.npz"


def build_latent_cache(
    model: CVAE,
    device,
    records,
    latents_dir: str | Path,
    *,
    batch_size: int = 256,
    overwrite: bool = False,
):
    latents_dir = Path(latents_dir)
    latents_dir.mkdir(parents=True, exist_ok=True)

    pooled_by_row = {}
    for row in records:
        feature_path = row.get("feature_path")
        if not feature_path:
            continue
        out_path = _npz_path(latents_dir, row)
        if out_path.exists() and not overwrite:
            cached = np.load(out_path, allow_pickle=True)
            mu = cached["mu"].astype(np.float32)
        else:
            mu = encode_mu_matrix(model, feature_path, device=device, batch_size=batch_size)
            np.savez_compressed(
                out_path,
                mu=mu.astype(np.float32),
                y=np.array([row.get("experimental_delta_g")], dtype=np.float32),
                complex_id=np.array([row.get("complex_id") or row.get("pdb_id")], dtype=object),
                pdb_id=np.array([row.get("pdb_id")], dtype=object),
                row_index=np.array([int(row["row_index"])], dtype=np.int64),
                split=np.array([row.get("split", "train")], dtype=object),
            )
        pooled_by_row[int(row["row_index"])] = mean_std_pool(mu)
    return pooled_by_row

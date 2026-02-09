import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bridging.utils.dataset_rows import row_pdb_id

from .config import (
    DEFAULT_FEATURE_GLOB,
    LATENT_DIM,
    BASE_CHANNELS,
    BATCH_SIZE,
    LR,
    EPOCHS,
    BETA_MAX,
    BETA_WARMUP_EPOCHS,
)
from .dataset import FeatureFrameDataset, collect_feature_files
from .model import CVAE, loss_fn


def _infer_shape(paths):
    arr = np.load(paths[0], mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"{paths[0]} expected shape (T,C,N,N), got {arr.shape}")
    _, c, n, _ = arr.shape
    return c, n


def _split_name(value) -> str:
    s = str(value).strip().lower()
    if s in {"test", "val", "valid", "validation"}:
        return "test"
    return "train"


def _dataset_split_pdbs(dataset_path: str | Path, split: str) -> set[str]:
    df = pd.read_csv(dataset_path)
    keep = []
    split_mode = str(split).strip().lower()
    for row in df.to_dict("records"):
        pdb = row_pdb_id(row)
        if not pdb:
            continue
        if split_mode != "all":
            row_split = _split_name(row.get("split", "train"))
            if row_split != split_mode:
                continue
        keep.append(str(pdb).upper())
    return set(keep)


def _feature_pdb_id(path_like: str) -> str | None:
    path = Path(path_like)
    # expected: .../<PDB>/features/<filename>.npy
    if path.parent.name != "features":
        return None
    pdb = path.parent.parent.name.strip().upper()
    if len(pdb) != 4:
        return None
    return pdb


def train(
    features,
    out_path,
    *,
    latent_dim=LATENT_DIM,
    base_channels=BASE_CHANNELS,
    batch_size=BATCH_SIZE,
    lr=LR,
    epochs=EPOCHS,
    beta_max=BETA_MAX,
    beta_warmup_epochs=BETA_WARMUP_EPOCHS,
    frame_stride=1,
    max_frames=None,
    num_workers=0,
    contact_weight=1.0,
    dist_weight=1.0,
    dataset_path=None,
    split="all",
    device=None,
):
    paths = collect_feature_files(features)
    if not paths:
        raise FileNotFoundError(f"No features found for: {features}")

    split_mode = str(split).strip().lower()
    if split_mode not in {"all", "train", "test"}:
        raise ValueError(f"Invalid split='{split}'. Expected one of: all, train, test.")

    if dataset_path:
        allowed = _dataset_split_pdbs(dataset_path, split_mode)
        before = len(paths)
        paths = [p for p in paths if (_feature_pdb_id(p) in allowed)]
        print(
            f"[ML] dataset_filter dataset={dataset_path} split={split_mode} "
            f"feature_files={len(paths)}/{before}"
        )
        if not paths:
            raise FileNotFoundError(
                f"No features left after dataset split filter: dataset={dataset_path} split={split_mode}"
            )

    in_channels, img_size = _infer_shape(paths)
    dataset = FeatureFrameDataset(paths, frame_stride=frame_stride, max_frames=max_frames)
    if len(dataset) == 0:
        raise ValueError("No frames available after applying frame_stride/max_frames.")
    print(f"[ML] feature_files={len(paths)} frame_samples={len(dataset)}")
    if len(dataset) < batch_size:
        print(
            f"[WARN] frame_samples={len(dataset)} is smaller than batch_size={batch_size}; "
            "training continues with smaller final batches."
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(
        in_channels=in_channels,
        img_size=img_size,
        latent_dim=latent_dim,
        base_channels=base_channels,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        if beta_warmup_epochs <= 0:
            beta = beta_max
        else:
            beta = beta_max * min(1.0, (epoch - 1) / beta_warmup_epochs)

        running = 0.0
        parts = {}
        for x in loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            x_hat_logits, mu, logvar = model(x)
            loss, parts = loss_fn(
                x,
                x_hat_logits,
                mu,
                logvar,
                beta=beta,
                w_contact=contact_weight,
                w_dist=dist_weight,
            )
            loss.backward()
            opt.step()
            running += loss.item()

        avg = running / max(1, len(loader))
        print(f"epoch {epoch:03d} beta={beta:.3f} loss={avg:.4f} parts={parts}")

    state = {
        "state_dict": model.state_dict(),
        "config": {
            "latent_dim": latent_dim,
            "base_channels": base_channels,
            "in_channels": in_channels,
            "img_size": img_size,
        },
    }
    torch.save(state, out_path)
    print(f"[OK] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a CVAE on featurized MD contact+distance maps.")
    parser.add_argument("--features", default=DEFAULT_FEATURE_GLOB, help="Glob or directory of .npy features")
    parser.add_argument("--out", default="cvae.pt", help="Output model path")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--beta-max", type=float, default=BETA_MAX)
    parser.add_argument("--beta-warmup-epochs", type=int, default=BETA_WARMUP_EPOCHS)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--dist-weight", type=float, default=1.0)
    parser.add_argument("--dataset", help="Optional dataset CSV used to filter features by split")
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--device", help="cuda or cpu")
    args = parser.parse_args()

    train(
        args.features,
        args.out,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        beta_max=args.beta_max,
        beta_warmup_epochs=args.beta_warmup_epochs,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        contact_weight=args.contact_weight,
        dist_weight=args.dist_weight,
        dataset_path=args.dataset,
        split=args.split,
        device=args.device,
    )


if __name__ == "__main__":
    main()

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


def _split_paths_for_validation(paths, val_fraction, val_seed):
    paths = list(paths)
    if val_fraction <= 0 or len(paths) < 2:
        return paths, []

    n_val = int(round(len(paths) * float(val_fraction)))
    n_val = max(1, n_val)
    n_val = min(len(paths) - 1, n_val)
    if n_val <= 0:
        return paths, []

    rng = np.random.default_rng(int(val_seed))
    order = rng.permutation(len(paths))
    val_idx = set(order[:n_val].tolist())

    train_paths = [p for i, p in enumerate(paths) if i not in val_idx]
    val_paths = [p for i, p in enumerate(paths) if i in val_idx]
    return train_paths, val_paths


def _run_epoch(
    model,
    loader,
    *,
    device,
    beta,
    contact_weight,
    dist_weight,
    optimizer=None,
):
    train_mode = optimizer is not None
    model.train(train_mode)

    running = 0.0
    batches = 0
    part_sums = {}

    for x in loader:
        x = x.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

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

        if train_mode:
            loss.backward()
            optimizer.step()

        running += loss.item()
        batches += 1
        for k, v in parts.items():
            part_sums[k] = part_sums.get(k, 0.0) + float(v)

    avg_loss = running / max(1, batches)
    avg_parts = {k: (v / max(1, batches)) for k, v in part_sums.items()}
    return avg_loss, avg_parts


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
    val_fraction=0.0,
    val_seed=42,
):
    out_path = Path(out_path)
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

    train_paths, val_paths = _split_paths_for_validation(paths, val_fraction, val_seed)
    if val_paths:
        print(
            f"[ML] validation split: train_feature_files={len(train_paths)} "
            f"val_feature_files={len(val_paths)} val_fraction={float(val_fraction):.3f}"
        )

    in_channels, img_size = _infer_shape(train_paths)
    dataset = FeatureFrameDataset(train_paths, frame_stride=frame_stride, max_frames=max_frames)
    if len(dataset) == 0:
        raise ValueError("No frames available after applying frame_stride/max_frames.")
    print(f"[ML] feature_files={len(train_paths)} frame_samples={len(dataset)}")
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

    val_loader = None
    if val_paths:
        val_dataset = FeatureFrameDataset(val_paths, frame_stride=frame_stride, max_frames=max_frames)
        if len(val_dataset) == 0:
            print("[WARN] validation split requested but produced zero validation frames; disabling validation.")
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
            print(f"[ML] val_frame_samples={len(val_dataset)}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(
        in_channels=in_channels,
        img_size=img_size,
        latent_dim=latent_dim,
        base_channels=base_channels,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        if beta_warmup_epochs <= 0:
            beta = beta_max
        else:
            beta = beta_max * min(1.0, (epoch - 1) / beta_warmup_epochs)

        train_loss, train_parts = _run_epoch(
            model,
            loader,
            device=device,
            beta=beta,
            contact_weight=contact_weight,
            dist_weight=dist_weight,
            optimizer=opt,
        )

        if val_loader is None:
            print(
                f"epoch {epoch:03d} beta={beta:.3f} "
                f"train_loss={train_loss:.4f} train_parts={train_parts}"
            )
            continue

        with torch.no_grad():
            val_loss, val_parts = _run_epoch(
                model,
                val_loader,
                device=device,
                beta=beta,
                contact_weight=contact_weight,
                dist_weight=dist_weight,
                optimizer=None,
            )
        print(
            f"epoch {epoch:03d} beta={beta:.3f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_parts={train_parts} val_parts={val_parts}"
        )

    state = {
        "state_dict": model.state_dict(),
        "config": {
            "latent_dim": latent_dim,
            "base_channels": base_channels,
            "in_channels": in_channels,
            "img_size": img_size,
            "val_fraction": float(val_fraction),
            "val_seed": int(val_seed),
            "train_feature_files": int(len(train_paths)),
            "val_feature_files": int(len(val_paths)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Optional validation fraction of selected feature files (complex-level).",
    )
    parser.add_argument("--val-seed", type=int, default=42)
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
        val_fraction=args.val_fraction,
        val_seed=args.val_seed,
    )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bridging.utils.affinity import experimental_delta_g_kcalmol, split_name
from bridging.utils.dataset_rows import row_pdb_id

from .config import (
    BATCH_SIZE,
    BETA_MAX,
    BETA_WARMUP_EPOCHS,
    DEFAULT_FEATURE_GLOB,
    EPOCHS,
    LATENT_DIM,
    LR,
    SET_HIDDEN,
)
from .dataset import ComplexTrajectoryDataset, FeatureFrameDataset, collect_feature_files, feature_pdb_id
from .model import CVAE, loss_fn


def _infer_shape(paths):
    arr = np.load(paths[0], mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"{paths[0]} expected shape (T,C,N,N), got {arr.shape}")
    _, c, n, _ = arr.shape
    return c, n


def _dataset_split_pdbs(dataset_path: str | Path, split: str) -> set[str]:
    df = pd.read_csv(dataset_path)
    split_mode = str(split).strip().lower()
    keep = []
    for row in df.to_dict("records"):
        pdb = row_pdb_id(row)
        if not pdb:
            continue
        if split_mode != "all" and split_name(row.get("split", "train")) != split_mode:
            continue
        keep.append(str(pdb).upper())
    return set(keep)


def _dataset_label_map(dataset_path: str | Path, split: str) -> dict[str, float]:
    df = pd.read_csv(dataset_path)
    split_mode = str(split).strip().lower()
    by_pdb = {}
    for row in df.to_dict("records"):
        pdb = row_pdb_id(row)
        if not pdb:
            continue
        if split_mode != "all" and split_name(row.get("split", "train")) != split_mode:
            continue
        y = experimental_delta_g_kcalmol(row)
        if y is None:
            continue
        by_pdb[str(pdb).upper()] = float(y)
    return by_pdb


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


def _run_epoch_frame(
    model,
    loader,
    *,
    device,
    beta,
    contact_weight,
    dist_weight,
    affinity_weight=0.0,
    affinity_head=None,
    optimizer=None,
):
    train_mode = optimizer is not None
    model.train(train_mode)
    if affinity_head is not None:
        affinity_head.train(train_mode)

    running = 0.0
    batches = 0
    part_sums = {}

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, has_y = batch
            y = y.to(device, non_blocking=True)
            has_y = has_y.to(device, non_blocking=True)
        else:
            x = batch
            y = None
            has_y = None
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

        if affinity_head is not None and float(affinity_weight) > 0 and y is not None and has_y is not None:
            mask = has_y.bool()
            if torch.any(mask):
                y_hat = affinity_head(mu).squeeze(-1)
                aff_loss = F.mse_loss(y_hat[mask], y[mask], reduction="mean")
                loss = loss + float(affinity_weight) * aff_loss
                parts["aff_mse"] = float(aff_loss.item())
            else:
                parts["aff_mse"] = float("nan")

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


def _temporal_pool(mu_seq: torch.Tensor) -> torch.Tensor:
    # mu_seq: (B, T, D)
    mean = mu_seq.mean(dim=1)
    std = mu_seq.std(dim=1, unbiased=False)
    if mu_seq.size(1) < 2:
        z = torch.zeros(mu_seq.size(0), 1, device=mu_seq.device, dtype=mu_seq.dtype)
        return torch.cat([mean, std, z, z], dim=1)
    delta = mu_seq[:, 1:, :] - mu_seq[:, :-1, :]
    step_norm = torch.linalg.vector_norm(delta, dim=2)
    step_mean = step_norm.mean(dim=1, keepdim=True)
    step_std = step_norm.std(dim=1, unbiased=False, keepdim=True)
    return torch.cat([mean, std, step_mean, step_std], dim=1)


def _run_epoch_trajectory(
    model,
    loader,
    *,
    device,
    beta,
    contact_weight,
    dist_weight,
    affinity_weight=0.0,
    affinity_head=None,
    optimizer=None,
):
    train_mode = optimizer is not None
    model.train(train_mode)
    if affinity_head is not None:
        affinity_head.train(train_mode)

    running = 0.0
    batches = 0
    part_sums = {}

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x_seq, y, has_y = batch
            y = y.to(device, non_blocking=True)
            has_y = has_y.to(device, non_blocking=True)
        else:
            x_seq = batch
            y = None
            has_y = None

        x_seq = x_seq.to(device, non_blocking=True)  # (B,T,C,N,N)
        bsz, tsteps = int(x_seq.size(0)), int(x_seq.size(1))
        x = x_seq.reshape(bsz * tsteps, *x_seq.shape[2:])

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

        if affinity_head is not None and float(affinity_weight) > 0 and y is not None and has_y is not None:
            mask = has_y.bool()
            if torch.any(mask):
                mu_seq = mu.reshape(bsz, tsteps, -1)
                pooled = _temporal_pool(mu_seq)
                y_hat = affinity_head(pooled).squeeze(-1)
                aff_loss = F.mse_loss(y_hat[mask], y[mask], reduction="mean")
                loss = loss + float(affinity_weight) * aff_loss
                parts["aff_mse"] = float(aff_loss.item())
            else:
                parts["aff_mse"] = float("nan")

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


def _prefix_parts(parts: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": float(v) for k, v in parts.items()}


def train(
    features,
    out_path,
    *,
    latent_dim=LATENT_DIM,
    set_hidden=SET_HIDDEN,
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
    affinity_weight=0.0,
    frames_per_complex=32,
):
    out_raw = str(out_path).strip()
    if not out_raw:
        raise ValueError("Output path is empty. Set --out to a file path, e.g. models/cvae.pt")
    out_path = Path(out_raw)
    if out_path.exists() and out_path.is_dir():
        raise ValueError(f"--out points to a directory, not a file: {out_path}")
    if out_path.name in {"", "."}:
        raise ValueError(f"Invalid output filename for --out: {out_path}")

    split_mode = str(split).strip().lower()
    if split_mode not in {"all", "train", "test"}:
        raise ValueError(f"Invalid split='{split}'. Expected one of: all, train, test.")

    paths = collect_feature_files(features)
    if not paths:
        raise FileNotFoundError(f"No features found for: {features}")

    if dataset_path:
        allowed = _dataset_split_pdbs(dataset_path, split_mode)
        before = len(paths)
        paths = [p for p in paths if (feature_pdb_id(p) in allowed)]
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

    label_map = {}
    if float(affinity_weight) > 0:
        if not dataset_path:
            print("[WARN] affinity_weight>0 but no --dataset provided; affinity supervision disabled.")
            affinity_weight = 0.0
        else:
            label_map = _dataset_label_map(dataset_path, split_mode)
            if not label_map:
                print("[WARN] no experimental labels found in dataset; affinity supervision disabled.")
                affinity_weight = 0.0
            else:
                print(f"[ML] semi_supervised labels={len(label_map)} affinity_weight={float(affinity_weight):.3f}")

    in_channels, img_size = _infer_shape(train_paths)

    dataset_frame = FeatureFrameDataset(
        train_paths,
        frame_stride=frame_stride,
        max_frames=max_frames,
        target_by_pdb=label_map if float(affinity_weight) > 0 else None,
    )
    if len(dataset_frame) == 0:
        raise ValueError("No frames available after applying frame_stride/max_frames.")
    print(f"[ML] feature_files={len(train_paths)} frame_samples={len(dataset_frame)}")

    dataset_traj = ComplexTrajectoryDataset(
        train_paths,
        frame_stride=frame_stride,
        max_frames=max_frames,
        frames_per_complex=frames_per_complex,
        target_by_pdb=label_map if float(affinity_weight) > 0 else None,
    )
    if len(dataset_traj) == 0:
        raise ValueError("No complexes available after applying frame_stride/max_frames.")
    print(
        f"[ML] feature_files={len(train_paths)} complex_samples={len(dataset_traj)} "
        f"frames_per_complex={int(frames_per_complex)}"
    )

    loader_frame = DataLoader(
        dataset_frame,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    loader_traj = DataLoader(
        dataset_traj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader_frame = None
    val_loader_traj = None
    if val_paths:
        val_dataset_frame = FeatureFrameDataset(
            val_paths,
            frame_stride=frame_stride,
            max_frames=max_frames,
            target_by_pdb=label_map if float(affinity_weight) > 0 else None,
        )
        if len(val_dataset_frame) > 0:
            val_loader_frame = DataLoader(
                val_dataset_frame,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
            print(f"[ML] val_frame_samples={len(val_dataset_frame)}")

        val_dataset_traj = ComplexTrajectoryDataset(
            val_paths,
            frame_stride=frame_stride,
            max_frames=max_frames,
            frames_per_complex=frames_per_complex,
            target_by_pdb=label_map if float(affinity_weight) > 0 else None,
        )
        if len(val_dataset_traj) > 0:
            val_loader_traj = DataLoader(
                val_dataset_traj,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
            print(f"[ML] val_complex_samples={len(val_dataset_traj)}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(
        in_channels=in_channels,
        img_size=img_size,
        latent_dim=latent_dim,
        set_hidden=set_hidden,
    ).to(device)

    affinity_head_frame = None
    affinity_head_traj = None
    params = list(model.parameters())
    if float(affinity_weight) > 0:
        affinity_head_frame = nn.Linear(latent_dim, 1).to(device)
        temporal_in = (2 * int(latent_dim)) + 2
        affinity_head_traj = nn.Sequential(
            nn.Linear(temporal_in, int(latent_dim)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(int(latent_dim), 1),
        ).to(device)
        params += list(affinity_head_frame.parameters())
        params += list(affinity_head_traj.parameters())

    opt = torch.optim.Adam(params, lr=lr)

    for epoch in range(1, epochs + 1):
        if beta_warmup_epochs <= 0:
            beta = beta_max
        else:
            beta = beta_max * min(1.0, (epoch - 1) / beta_warmup_epochs)

        train_loss_frame, train_parts_frame = _run_epoch_frame(
            model,
            loader_frame,
            device=device,
            beta=beta,
            contact_weight=contact_weight,
            dist_weight=dist_weight,
            affinity_weight=affinity_weight,
            affinity_head=affinity_head_frame,
            optimizer=opt,
        )
        train_loss_traj, train_parts_traj = _run_epoch_trajectory(
            model,
            loader_traj,
            device=device,
            beta=beta,
            contact_weight=contact_weight,
            dist_weight=dist_weight,
            affinity_weight=affinity_weight,
            affinity_head=affinity_head_traj,
            optimizer=opt,
        )

        train_loss = 0.5 * (train_loss_frame + train_loss_traj)
        train_parts = {}
        train_parts.update(_prefix_parts(train_parts_frame, "frame_"))
        train_parts.update(_prefix_parts(train_parts_traj, "traj_"))

        if val_loader_frame is None and val_loader_traj is None:
            print(
                f"epoch {epoch:03d} beta={beta:.3f} "
                f"train_loss={train_loss:.4f} train_parts={train_parts}"
            )
            continue

        with torch.no_grad():
            if val_loader_frame is not None:
                val_loss_frame, val_parts_frame = _run_epoch_frame(
                    model,
                    val_loader_frame,
                    device=device,
                    beta=beta,
                    contact_weight=contact_weight,
                    dist_weight=dist_weight,
                    affinity_weight=affinity_weight,
                    affinity_head=affinity_head_frame,
                    optimizer=None,
                )
            else:
                val_loss_frame, val_parts_frame = float("nan"), {}

            if val_loader_traj is not None:
                val_loss_traj, val_parts_traj = _run_epoch_trajectory(
                    model,
                    val_loader_traj,
                    device=device,
                    beta=beta,
                    contact_weight=contact_weight,
                    dist_weight=dist_weight,
                    affinity_weight=affinity_weight,
                    affinity_head=affinity_head_traj,
                    optimizer=None,
                )
            else:
                val_loss_traj, val_parts_traj = float("nan"), {}

        vals = [v for v in [val_loss_frame, val_loss_traj] if not np.isnan(v)]
        val_loss = float(np.mean(vals)) if vals else float("nan")
        val_parts = {}
        val_parts.update(_prefix_parts(val_parts_frame, "frame_"))
        val_parts.update(_prefix_parts(val_parts_traj, "traj_"))
        print(
            f"epoch {epoch:03d} beta={beta:.3f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_parts={train_parts} val_parts={val_parts}"
        )

    state = {
        "state_dict": model.state_dict(),
        "config": {
            "latent_dim": latent_dim,
            "set_hidden": int(set_hidden),
            "encoder_type": "set_edge",
            "decoder_type": "mlp",
            "training_mode": "hybrid",
            "in_channels": in_channels,
            "img_size": img_size,
            "frames_per_complex": int(frames_per_complex),
            "val_fraction": float(val_fraction),
            "val_seed": int(val_seed),
            "affinity_weight": float(affinity_weight),
            "train_feature_files": int(len(train_paths)),
            "val_feature_files": int(len(val_paths)),
        },
    }
    if affinity_head_frame is not None:
        state["affinity_head_frame_state_dict"] = affinity_head_frame.state_dict()
    if affinity_head_traj is not None:
        state["affinity_head_traj_state_dict"] = affinity_head_traj.state_dict()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    print(f"[OK] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train set-edge hybrid CVAE on featurized MD contact/distance maps.")
    parser.add_argument("--features", default=DEFAULT_FEATURE_GLOB, help="Glob or directory of .npy features")
    parser.add_argument("--out", default="cvae.pt", help="Output model path")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--set-hidden", type=int, default=SET_HIDDEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--beta-max", type=float, default=BETA_MAX)
    parser.add_argument("--beta-warmup-epochs", type=int, default=BETA_WARMUP_EPOCHS)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frames-per-complex", type=int, default=32)
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
    parser.add_argument(
        "--affinity-weight",
        type=float,
        default=0.0,
        help="Weight for latent->affinity MSE term. 0 keeps pure reconstruction+KL training.",
    )
    args = parser.parse_args()

    train(
        args.features,
        args.out,
        latent_dim=args.latent_dim,
        set_hidden=args.set_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        beta_max=args.beta_max,
        beta_warmup_epochs=args.beta_warmup_epochs,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        frames_per_complex=args.frames_per_complex,
        num_workers=args.num_workers,
        contact_weight=args.contact_weight,
        dist_weight=args.dist_weight,
        dataset_path=args.dataset,
        split=args.split,
        device=args.device,
        val_fraction=args.val_fraction,
        val_seed=args.val_seed,
        affinity_weight=args.affinity_weight,
    )


if __name__ == "__main__":
    main()

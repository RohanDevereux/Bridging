from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .dataset import FeatureStandardizer, PreparedGraphDataset, build_feature_spec
from .model import MaskedGraphVAE


def _fmt_seconds(seconds: float) -> str:
    s = max(0, int(round(float(seconds))))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _beta_for_epoch(epoch: int, max_epochs: int, beta_start: float, beta_end: float, anneal_fraction: float) -> float:
    warmup = max(1, int(round(max_epochs * anneal_fraction)))
    if epoch >= warmup:
        return float(beta_end)
    t = float(epoch) / float(warmup)
    return float(beta_start + t * (beta_end - beta_start))


def _epoch_pass(
    model: MaskedGraphVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    mask_ratio: float,
    beta: float,
    corr_weight: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"loss": 0.0, "recon": 0.0, "node_recon": 0.0, "edge_recon": 0.0, "kl": 0.0, "corr": 0.0, "n": 0}
    for batch in loader:
        batch = batch.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        loss, parts, _ = model.compute_loss(
            batch,
            mask_ratio=mask_ratio,
            beta=beta,
            corr_weight=corr_weight,
        )
        if train_mode:
            loss.backward()
            optimizer.step()
        bs = int(batch.y.shape[0])
        totals["n"] += bs
        for k in ("loss", "recon", "node_recon", "edge_recon", "kl", "corr"):
            totals[k] += parts[k] * bs
    if totals["n"] < 1:
        return {k: 0.0 for k in ("loss", "recon", "node_recon", "edge_recon", "kl", "corr")}
    return {k: totals[k] / totals["n"] for k in ("loss", "recon", "node_recon", "edge_recon", "kl", "corr")}


def _collect_latents(
    model: MaskedGraphVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    split: str,
) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu = model.encode_mu(batch).detach().cpu().numpy()
            y = batch.y.detach().cpu().numpy().reshape(-1)
            cid = getattr(batch, "complex_id", None)
            if isinstance(cid, str):
                cids = [cid]
            else:
                cids = list(cid) if cid is not None else [f"graph_{i}" for i in range(mu.shape[0])]
            for i in range(mu.shape[0]):
                row = {"complex_id": cids[i], "split": split, "dG": float(y[i])}
                for k in range(mu.shape[1]):
                    row[f"mu_{k}"] = float(mu[i, k])
                rows.append(row)
    return rows


def train_masked_graph_vae(
    *,
    records_path: Path,
    out_dir: Path,
    mode: str,
    device: str,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    mask_ratio: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    beta_start: float,
    beta_end: float,
    beta_anneal_fraction: float,
    corr_weight: float,
    seed: int,
    num_workers: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(seed)

    records = torch.load(records_path, map_location="cpu")
    spec = build_feature_spec(records, mode=mode)
    scaler = FeatureStandardizer.fit(records, spec)

    ds_train = PreparedGraphDataset(records, split="train", spec=spec, scaler=scaler)
    ds_val = PreparedGraphDataset(records, split="val", spec=spec, scaler=scaler)
    ds_test = PreparedGraphDataset(records, split="test", spec=spec, scaler=scaler)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_eval_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dev = torch.device(device)
    print(
        f"[GVAE][{mode}] start device={dev} train_graphs={len(ds_train)} "
        f"val_graphs={len(ds_val)} test_graphs={len(ds_test)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={batch_size} max_epochs={max_epochs}"
    )
    model = MaskedGraphVAE(
        node_in_dim=len(spec.node_input_names),
        edge_in_dim=len(spec.edge_input_names),
        node_target_idx=spec.node_target_idx,
        edge_target_idx=spec.edge_target_idx,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    run_t0 = time.perf_counter()
    epoch_times: list[float] = []

    for epoch in range(1, max_epochs + 1):
        epoch_t0 = time.perf_counter()
        beta = _beta_for_epoch(epoch, max_epochs, beta_start, beta_end, beta_anneal_fraction)
        train_metrics = _epoch_pass(
            model,
            train_loader,
            device=dev,
            optimizer=optimizer,
            mask_ratio=mask_ratio,
            beta=beta,
            corr_weight=corr_weight,
        )
        val_metrics = _epoch_pass(
            model,
            val_loader,
            device=dev,
            optimizer=None,
            mask_ratio=mask_ratio,
            beta=beta,
            corr_weight=corr_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "beta": beta,
                "train_loss": train_metrics["loss"],
                "train_recon": train_metrics["recon"],
                "val_loss": val_metrics["loss"],
                "val_recon": val_metrics["recon"],
            }
        )
        epoch_s = time.perf_counter() - epoch_t0
        epoch_times.append(epoch_s)
        elapsed_s = time.perf_counter() - run_t0
        mean_epoch_s = float(np.mean(epoch_times))
        remaining_epochs = max_epochs - epoch
        eta_s = remaining_epochs * mean_epoch_s
        print(
            f"[GVAE][{mode}] epoch={epoch:03d}/{max_epochs} beta={beta:.3f} "
            f"train_recon={train_metrics['recon']:.5f} val_recon={val_metrics['recon']:.5f} "
            f"best_val_recon={best_val if best_epoch > 0 else val_metrics['recon']:.5f} "
            f"epoch_s={epoch_s:.1f} elapsed={_fmt_seconds(elapsed_s)} eta={_fmt_seconds(eta_s)}"
        )

        if val_metrics["recon"] < best_val:
            best_val = val_metrics["recon"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[GVAE][{mode}] early stopping at epoch={epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    model.load_state_dict(best_state)

    latents_rows = []
    latents_rows.extend(_collect_latents(model, train_eval_loader, device=dev, split="train"))
    latents_rows.extend(_collect_latents(model, val_loader, device=dev, split="val"))
    latents_rows.extend(_collect_latents(model, test_loader, device=dev, split="test"))
    latents_df = pd.DataFrame(latents_rows)
    latents_path = out_dir / f"latents_{mode}.csv"
    latents_df.to_csv(latents_path, index=False)

    history_df = pd.DataFrame(history)
    history_path = out_dir / f"train_history_{mode}.csv"
    history_df.to_csv(history_path, index=False)

    ckpt_path = out_dir / f"masked_graph_vae_{mode}.pt"
    torch.save(
        {
            "mode": mode,
            "model_state_dict": best_state,
            "feature_spec": {
                "node_input_names": spec.node_input_names,
                "edge_input_names": spec.edge_input_names,
                "node_target_names": spec.node_target_names,
                "edge_target_names": spec.edge_target_names,
                "node_target_idx": spec.node_target_idx,
                "edge_target_idx": spec.edge_target_idx,
            },
            "scaler": scaler.to_dict(),
            "config": {
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "mask_ratio": mask_ratio,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "beta_anneal_fraction": beta_anneal_fraction,
                "corr_weight": corr_weight,
                "seed": seed,
            },
            "records_path": str(records_path),
            "best_epoch": best_epoch,
            "best_val_recon": best_val,
        },
        ckpt_path,
    )

    summary = {
        "mode": mode,
        "records_path": str(records_path),
        "model_path": str(ckpt_path),
        "latents_csv": str(latents_path),
        "history_csv": str(history_path),
        "best_epoch": int(best_epoch),
        "best_val_recon": float(best_val),
        "n_train_graphs": len(ds_train),
        "n_val_graphs": len(ds_val),
        "n_test_graphs": len(ds_test),
    }
    summary_path = out_dir / f"train_summary_{mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked Graph-VAE (mode S or SD) and export latents.")
    parser.add_argument("--records", required=True, help="Prepared graph_records.pt path.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["S", "SD"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--beta-anneal-fraction", type=float, default=0.30)
    parser.add_argument("--corr-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = train_masked_graph_vae(
        records_path=Path(args.records),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        device=args.device,
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        mask_ratio=float(args.mask_ratio),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
        beta_anneal_fraction=float(args.beta_anneal_fraction),
        corr_weight=float(args.corr_weight),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
    )
    print(f"[GVAE] mode={summary['mode']} best_val_recon={summary['best_val_recon']:.6f} best_epoch={summary['best_epoch']}")
    print(f"[GVAE] latents_csv={summary['latents_csv']}")


if __name__ == "__main__":
    main()

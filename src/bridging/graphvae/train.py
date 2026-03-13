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

from .dataset import FeatureStandardizer, PreparedGraphDataset, SUPPORTED_TARGET_POLICIES, build_feature_spec
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


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 1:
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "mean_error": float("nan"),
        }
    err = y_pred - y_true
    sse = float(np.sum(err * err))
    rmse = float(np.sqrt(sse / y_true.size))
    mae = float(np.mean(np.abs(err)))
    if y_true.size < 2:
        r2 = float("nan")
    else:
        centered = y_true - float(np.mean(y_true))
        sst = float(np.sum(centered * centered))
        r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {
        "n": int(y_true.size),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": _pearson(y_true, y_pred),
        "mean_error": float(np.mean(err)),
    }


def _epoch_pass(
    model: MaskedGraphVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    mask_ratio: float,
    beta: float,
    corr_weight: float,
    affinity_weight: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "node_recon": 0.0,
        "edge_recon": 0.0,
        "kl": 0.0,
        "corr": 0.0,
        "affinity_loss": 0.0,
        "n": 0,
        "skipped_batches": 0,
    }
    affinity_true_chunks: list[np.ndarray] = []
    affinity_pred_chunks: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            loss, parts, _mu, affinity_pred = model.compute_loss(
                batch,
                mask_ratio=mask_ratio,
                beta=beta,
                corr_weight=corr_weight,
                affinity_weight=affinity_weight,
            )
        if not torch.isfinite(loss):
            totals["skipped_batches"] += 1
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            continue
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        bs = int(batch.y.shape[0])
        totals["n"] += bs
        for k in ("loss", "recon", "node_recon", "edge_recon", "kl", "corr", "affinity_loss"):
            totals[k] += float(parts[k]) * bs
        if affinity_pred is not None and hasattr(batch, "y") and batch.y is not None:
            affinity_true_chunks.append(batch.y.detach().cpu().numpy().reshape(-1))
            affinity_pred_chunks.append(affinity_pred.detach().cpu().numpy().reshape(-1))

    if totals["n"] < 1:
        return {
            "loss": float("nan"),
            "recon": float("nan"),
            "node_recon": float("nan"),
            "edge_recon": float("nan"),
            "kl": float("nan"),
            "corr": float("nan"),
            "affinity_loss": float("nan"),
            "affinity_rmse": float("nan"),
            "affinity_mae": float("nan"),
            "affinity_r2": float("nan"),
            "affinity_pearson_r": float("nan"),
            "affinity_mean_error": float("nan"),
            "skipped_batches": float(totals["skipped_batches"]),
        }

    out = {
        k: totals[k] / totals["n"]
        for k in ("loss", "recon", "node_recon", "edge_recon", "kl", "corr", "affinity_loss")
    }
    if affinity_true_chunks:
        affinity_true = np.concatenate(affinity_true_chunks, axis=0)
        affinity_pred = np.concatenate(affinity_pred_chunks, axis=0)
        affinity_metrics = _regression_metrics(affinity_true, affinity_pred)
        out.update(
            {
                "affinity_rmse": affinity_metrics["rmse"],
                "affinity_mae": affinity_metrics["mae"],
                "affinity_r2": affinity_metrics["r2"],
                "affinity_pearson_r": affinity_metrics["pearson_r"],
                "affinity_mean_error": affinity_metrics["mean_error"],
            }
        )
    else:
        out.update(
            {
                "affinity_rmse": float("nan"),
                "affinity_mae": float("nan"),
                "affinity_r2": float("nan"),
                "affinity_pearson_r": float("nan"),
                "affinity_mean_error": float("nan"),
            }
        )
    out["skipped_batches"] = float(totals["skipped_batches"])
    return out


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


def _collect_affinity_predictions(
    model: MaskedGraphVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    split: str,
) -> tuple[list[dict], dict[str, float]]:
    model.eval()
    rows = []
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model.predict_affinity(batch).detach().cpu().numpy().reshape(-1)
            y = batch.y.detach().cpu().numpy().reshape(-1)
            cid = getattr(batch, "complex_id", None)
            if isinstance(cid, str):
                cids = [cid]
            else:
                cids = list(cid) if cid is not None else [f"graph_{i}" for i in range(pred.shape[0])]
            for i in range(pred.shape[0]):
                rows.append(
                    {
                        "complex_id": cids[i],
                        "split": split,
                        "dG": float(y[i]),
                        "dG_pred_head": float(pred[i]),
                        "error": float(pred[i] - y[i]),
                    }
                )
            y_true_chunks.append(y)
            y_pred_chunks.append(pred)
    if not y_true_chunks:
        return rows, _regression_metrics(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64))
    metrics = _regression_metrics(
        np.concatenate(y_true_chunks, axis=0),
        np.concatenate(y_pred_chunks, axis=0),
    )
    return rows, metrics


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
    checkpoint_every: int = 1,
    affinity_weight: float = 0.0,
    target_policy: str = "shared_static",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(1, int(checkpoint_every))
    _seed_all(seed)

    records = torch.load(records_path, map_location="cpu")
    train_targets = np.asarray([float(r["dG"]) for r in records if str(r.get("split")) == "train"], dtype=np.float64)
    if train_targets.size < 1:
        raise ValueError("No train targets found in prepared records.")
    affinity_target_mean = float(np.mean(train_targets))
    affinity_target_std = float(np.std(train_targets))
    if not np.isfinite(affinity_target_std) or affinity_target_std < 1e-6:
        affinity_target_std = 1.0

    spec = build_feature_spec(records, mode=mode, target_policy=target_policy)
    scaler = FeatureStandardizer.fit(records, spec)

    ds_train = PreparedGraphDataset(records, split="train", spec=spec, scaler=scaler)
    ds_val = PreparedGraphDataset(records, split="val", spec=spec, scaler=scaler)
    ds_test = PreparedGraphDataset(records, split="test", spec=spec, scaler=scaler)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_eval_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dev = torch.device(device)
    supervision_mode = "semi_supervised" if float(affinity_weight) > 0.0 else "unsupervised"
    print(
        f"[GVAE][{mode}][{supervision_mode}] start device={dev} train_graphs={len(ds_train)} "
        f"val_graphs={len(ds_val)} test_graphs={len(ds_test)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={batch_size} max_epochs={max_epochs} latent_dim={latent_dim} "
        f"affinity_weight={float(affinity_weight):.3f} target_policy={target_policy}"
    )
    model = MaskedGraphVAE(
        node_in_dim=len(spec.node_input_names),
        edge_in_dim=len(spec.edge_input_names),
        node_target_idx=spec.node_target_idx,
        edge_target_idx=spec.edge_target_idx,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        affinity_target_mean=affinity_target_mean,
        affinity_target_std=affinity_target_std,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_objective = float("inf")
    best_val_recon = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    run_t0 = time.perf_counter()
    epoch_times: list[float] = []
    history_path = out_dir / f"train_history_{mode}.csv"
    best_ckpt_path = out_dir / f"masked_graph_vae_{mode}_best.pt"
    last_ckpt_path = out_dir / f"masked_graph_vae_{mode}_last.pt"
    n_epochs_with_skipped_batches = 0
    unstable_reason: str | None = None

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
            affinity_weight=affinity_weight,
        )
        val_metrics = _epoch_pass(
            model,
            val_loader,
            device=dev,
            optimizer=None,
            mask_ratio=mask_ratio,
            beta=beta,
            corr_weight=corr_weight,
            affinity_weight=affinity_weight,
        )
        val_objective = float(val_metrics["recon"])
        if float(affinity_weight) > 0.0 and np.isfinite(val_metrics["affinity_loss"]):
            val_objective += float(affinity_weight) * float(val_metrics["affinity_loss"])
        epoch_skipped = int(train_metrics["skipped_batches"]) + int(val_metrics["skipped_batches"])
        if epoch_skipped > 0:
            n_epochs_with_skipped_batches += 1
            if unstable_reason is None:
                unstable_reason = "nonfinite_batches_detected"
        history.append(
            {
                "epoch": epoch,
                "beta": beta,
                "supervision_mode": supervision_mode,
                "affinity_weight": float(affinity_weight),
                "target_policy": target_policy,
                "train_loss": train_metrics["loss"],
                "train_recon": train_metrics["recon"],
                "train_node_recon": train_metrics["node_recon"],
                "train_edge_recon": train_metrics["edge_recon"],
                "train_kl": train_metrics["kl"],
                "train_corr": train_metrics["corr"],
                "train_affinity_loss": train_metrics["affinity_loss"],
                "train_affinity_rmse": train_metrics["affinity_rmse"],
                "train_affinity_mae": train_metrics["affinity_mae"],
                "train_affinity_r2": train_metrics["affinity_r2"],
                "train_affinity_pearson_r": train_metrics["affinity_pearson_r"],
                "val_loss": val_metrics["loss"],
                "val_recon": val_metrics["recon"],
                "val_node_recon": val_metrics["node_recon"],
                "val_edge_recon": val_metrics["edge_recon"],
                "val_kl": val_metrics["kl"],
                "val_corr": val_metrics["corr"],
                "val_affinity_loss": val_metrics["affinity_loss"],
                "val_affinity_rmse": val_metrics["affinity_rmse"],
                "val_affinity_mae": val_metrics["affinity_mae"],
                "val_affinity_r2": val_metrics["affinity_r2"],
                "val_affinity_pearson_r": val_metrics["affinity_pearson_r"],
                "val_objective": val_objective,
                "train_skipped_batches": int(train_metrics["skipped_batches"]),
                "val_skipped_batches": int(val_metrics["skipped_batches"]),
            }
        )
        epoch_s = time.perf_counter() - epoch_t0
        epoch_times.append(epoch_s)
        elapsed_s = time.perf_counter() - run_t0
        mean_epoch_s = float(np.mean(epoch_times))
        remaining_epochs = max_epochs - epoch
        eta_s = remaining_epochs * mean_epoch_s
        line = (
            f"[GVAE][{mode}][{supervision_mode}] epoch={epoch:03d}/{max_epochs} beta={beta:.3f} "
            f"train_recon={train_metrics['recon']:.5f} val_recon={val_metrics['recon']:.5f} "
            f"best_val_obj={best_val_objective if best_epoch > 0 else val_objective:.5f} "
            f"train_skip={int(train_metrics['skipped_batches'])} val_skip={int(val_metrics['skipped_batches'])} "
            f"epoch_s={epoch_s:.1f} elapsed={_fmt_seconds(elapsed_s)} eta={_fmt_seconds(eta_s)}"
        )
        if float(affinity_weight) > 0.0:
            line += (
                f" train_aff_rmse={train_metrics['affinity_rmse']:.4f} "
                f"val_aff_rmse={val_metrics['affinity_rmse']:.4f}"
            )
        print(line)

        if int(train_metrics["skipped_batches"]) >= len(train_loader) or int(val_metrics["skipped_batches"]) >= len(val_loader):
            print(f"[GVAE][{mode}][{supervision_mode}] stopping at epoch={epoch} due to all-batch non-finite collapse")
            break

        if np.isfinite(val_objective) and val_objective < best_val_objective:
            best_val_objective = val_objective
            best_val_recon = float(val_metrics["recon"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
            torch.save(
                {
                    "mode": mode,
                    "kind": "best",
                    "epoch": int(epoch),
                    "supervision_mode": supervision_mode,
                    "affinity_weight": float(affinity_weight),
                    "target_policy": target_policy,
                    "model_state_dict": best_state,
                    "feature_spec": {
                        "target_policy": spec.target_policy,
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
                        "affinity_weight": float(affinity_weight),
                        "target_policy": target_policy,
                        "affinity_target_mean": affinity_target_mean,
                        "affinity_target_std": affinity_target_std,
                    },
                    "records_path": str(records_path),
                    "best_epoch": int(best_epoch),
                    "best_val_recon": float(best_val_recon),
                    "best_val_objective": float(best_val_objective),
                },
                best_ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[GVAE][{mode}][{supervision_mode}] early stopping at epoch={epoch}")
                break

        if epoch % checkpoint_every == 0 or epoch == max_epochs:
            pd.DataFrame(history).to_csv(history_path, index=False)
            torch.save(
                {
                    "mode": mode,
                    "kind": "last",
                    "epoch": int(epoch),
                    "supervision_mode": supervision_mode,
                    "affinity_weight": float(affinity_weight),
                    "target_policy": target_policy,
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_epoch": int(best_epoch),
                    "best_val_recon": float(best_val_recon),
                    "best_val_objective": float(best_val_objective),
                },
                last_ckpt_path,
            )

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a finite best model state. "
            "This usually indicates numerical instability or extreme feature outliers."
        )
    model.load_state_dict(best_state)

    latents_rows = []
    latents_rows.extend(_collect_latents(model, train_eval_loader, device=dev, split="train"))
    latents_rows.extend(_collect_latents(model, val_loader, device=dev, split="val"))
    latents_rows.extend(_collect_latents(model, test_loader, device=dev, split="test"))
    latents_df = pd.DataFrame(latents_rows)
    latents_path = out_dir / f"latents_{mode}.csv"
    latents_df.to_csv(latents_path, index=False)

    affinity_predictions_path: str | None = None
    affinity_split_metrics: dict[str, dict[str, float]] = {}
    if float(affinity_weight) > 0.0:
        pred_rows = []
        split_rows, split_metrics = _collect_affinity_predictions(model, train_eval_loader, device=dev, split="train")
        pred_rows.extend(split_rows)
        affinity_split_metrics["train"] = split_metrics
        split_rows, split_metrics = _collect_affinity_predictions(model, val_loader, device=dev, split="val")
        pred_rows.extend(split_rows)
        affinity_split_metrics["val"] = split_metrics
        split_rows, split_metrics = _collect_affinity_predictions(model, test_loader, device=dev, split="test")
        pred_rows.extend(split_rows)
        affinity_split_metrics["test"] = split_metrics
        pred_df = pd.DataFrame(pred_rows)
        pred_path = out_dir / f"affinity_head_predictions_{mode}.csv"
        pred_df.to_csv(pred_path, index=False)
        affinity_predictions_path = str(pred_path)
        affinity_summary_path = out_dir / f"affinity_head_summary_{mode}.json"
        affinity_summary_path.write_text(json.dumps({"split_metrics": affinity_split_metrics}, indent=2), encoding="utf-8")

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    ckpt_path = out_dir / f"masked_graph_vae_{mode}.pt"
    torch.save(
        {
            "mode": mode,
            "supervision_mode": supervision_mode,
            "affinity_weight": float(affinity_weight),
            "target_policy": target_policy,
            "model_state_dict": best_state,
            "feature_spec": {
                "target_policy": spec.target_policy,
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
                "affinity_weight": float(affinity_weight),
                "target_policy": target_policy,
                "affinity_target_mean": affinity_target_mean,
                "affinity_target_std": affinity_target_std,
            },
            "records_path": str(records_path),
            "best_epoch": best_epoch,
            "best_val_recon": best_val_recon,
            "best_val_objective": best_val_objective,
            "best_checkpoint_path": str(best_ckpt_path),
            "last_checkpoint_path": str(last_ckpt_path),
        },
        ckpt_path,
    )

    summary = {
        "mode": mode,
        "supervision_mode": supervision_mode,
        "affinity_weight": float(affinity_weight),
        "target_policy": target_policy,
        "records_path": str(records_path),
        "model_path": str(ckpt_path),
        "latents_csv": str(latents_path),
        "history_csv": str(history_path),
        "best_epoch": int(best_epoch),
        "best_val_recon": float(best_val_recon),
        "best_val_objective": float(best_val_objective),
        "n_train_graphs": len(ds_train),
        "n_val_graphs": len(ds_val),
        "n_test_graphs": len(ds_test),
        "is_stable": bool(n_epochs_with_skipped_batches == 0),
        "n_epochs_with_skipped_batches": int(n_epochs_with_skipped_batches),
        "unstable_reason": unstable_reason,
        "affinity_head_predictions_csv": affinity_predictions_path,
        "affinity_head_split_metrics": affinity_split_metrics,
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
    parser.add_argument("--affinity-weight", type=float, default=0.0)
    parser.add_argument("--target-policy", choices=list(SUPPORTED_TARGET_POLICIES), default="shared_static")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
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
        affinity_weight=float(args.affinity_weight),
        target_policy=str(args.target_policy),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        checkpoint_every=int(args.checkpoint_every),
    )
    print(
        f"[GVAE] mode={summary['mode']} supervision={summary['supervision_mode']} "
        f"best_val_recon={summary['best_val_recon']:.6f} best_epoch={summary['best_epoch']}"
    )
    print(f"[GVAE] latents_csv={summary['latents_csv']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from .dataset import FeatureStandardizer, PreparedGraphDataset, build_feature_spec


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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")
    return {
        "n": int(y_true.shape[0]),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": pearson,
    }


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))


class GraphRegressor(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [GINEConv(_mlp(hidden_dim, hidden_dim, hidden_dim), edge_dim=hidden_dim) for _ in range(num_layers)]
        )
        self.head = _mlp(2 * hidden_dim, hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        h = F.relu(self.node_proj(data.x))
        e = F.relu(self.edge_proj(data.edge_attr))
        for conv in self.convs:
            h = F.relu(conv(h, data.edge_index, e))
        pooled = torch.cat([global_mean_pool(h, data.batch), global_max_pool(h, data.batch)], dim=1)
        return self.head(pooled).reshape(-1)


def _run_epoch(model, loader, *, device, optimizer=None) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.reshape(-1)
        loss = F.mse_loss(pred, y)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _predict(model, loader, *, device, split: str) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).detach().cpu().numpy()
            y = batch.y.detach().cpu().numpy().reshape(-1)
            cid = getattr(batch, "complex_id", None)
            cids = [cid] if isinstance(cid, str) else list(cid)
            for i in range(len(pred)):
                rows.append({"complex_id": cids[i], "split": split, "dG": float(y[i]), "dG_pred": float(pred[i])})
    return pd.DataFrame(rows)


def run_supervised_baseline(
    *,
    records_path: Path,
    out_dir: Path,
    mode: str,
    device: str = "cpu",
    hidden_dim: int = 128,
    num_layers: int = 3,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    batch_size: int = 16,
    max_epochs: int = 200,
    patience: int = 25,
    seed: int = 2026,
    num_workers: int = 0,
    checkpoint_every: int = 1,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(1, int(checkpoint_every))
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
        f"[BASE][{mode}] start device={dev} train_graphs={len(ds_train)} "
        f"val_graphs={len(ds_val)} test_graphs={len(ds_test)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={batch_size} max_epochs={max_epochs}"
    )
    model = GraphRegressor(
        node_dim=len(spec.node_input_names),
        edge_dim=len(spec.edge_input_names),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    bad = 0
    best_epoch = -1
    run_t0 = time.perf_counter()
    epoch_times: list[float] = []
    history: list[dict] = []
    history_path = out_dir / f"supervised_baseline_history_{mode}.csv"
    best_ckpt_path = out_dir / f"supervised_baseline_{mode}_best.pt"
    last_ckpt_path = out_dir / f"supervised_baseline_{mode}_last.pt"
    for epoch in range(1, max_epochs + 1):
        epoch_t0 = time.perf_counter()
        train_mse = _run_epoch(model, train_loader, device=dev, optimizer=optimizer)
        val_mse = _run_epoch(model, val_loader, device=dev, optimizer=None)
        epoch_s = time.perf_counter() - epoch_t0
        epoch_times.append(epoch_s)
        elapsed_s = time.perf_counter() - run_t0
        mean_epoch_s = float(np.mean(epoch_times))
        remaining_epochs = max_epochs - epoch
        eta_s = remaining_epochs * mean_epoch_s
        print(
            f"[BASE][{mode}] epoch={epoch:03d}/{max_epochs} "
            f"train_mse={train_mse:.5f} val_mse={val_mse:.5f} "
            f"best_val_mse={best_val if best_epoch > 0 else val_mse:.5f} "
            f"epoch_s={epoch_s:.1f} elapsed={_fmt_seconds(elapsed_s)} eta={_fmt_seconds(eta_s)}"
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "best_val_mse": float(best_val if best_epoch > 0 else val_mse),
                "epoch_s": float(epoch_s),
                "elapsed_s": float(elapsed_s),
                "eta_s": float(eta_s),
            }
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "mode": mode,
                    "kind": "best",
                    "epoch": int(epoch),
                    "model_state_dict": best_state,
                    "best_val_mse": float(best_val),
                },
                best_ckpt_path,
            )
        else:
            bad += 1
            if bad >= patience:
                break
        if epoch % checkpoint_every == 0 or epoch == max_epochs:
            pd.DataFrame(history).to_csv(history_path, index=False)
            torch.save(
                {
                    "mode": mode,
                    "kind": "last",
                    "epoch": int(epoch),
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_epoch": int(best_epoch),
                    "best_val_mse": float(best_val),
                },
                last_ckpt_path,
            )
    if best_state is None:
        raise RuntimeError("No best baseline model state.")
    model.load_state_dict(best_state)

    pred = pd.concat(
        [
            _predict(model, train_eval_loader, device=dev, split="train"),
            _predict(model, val_loader, device=dev, split="val"),
            _predict(model, test_loader, device=dev, split="test"),
        ],
        ignore_index=True,
    )
    pred_path = out_dir / f"supervised_baseline_pred_{mode}.csv"
    pred.to_csv(pred_path, index=False)

    metrics = {}
    for split in ("train", "val", "test"):
        d = pred[pred["split"] == split]
        metrics[split] = _metrics(d["dG"].to_numpy(dtype=np.float64), d["dG_pred"].to_numpy(dtype=np.float64))

    summary = {
        "mode": mode,
        "records_path": str(records_path),
        "predictions_csv": str(pred_path),
        "metrics": metrics,
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "history_csv": str(history_path),
        "best_checkpoint_path": str(best_ckpt_path),
        "last_checkpoint_path": str(last_ckpt_path),
    }
    summary_path = out_dir / f"supervised_baseline_summary_{mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised graph baseline on prepared records.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["S", "SD"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_supervised_baseline(
        records_path=Path(args.records),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        device=args.device,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        checkpoint_every=int(args.checkpoint_every),
    )
    t = summary["metrics"]["test"]
    print(f"[BASE] mode={summary['mode']} test_rmse={t['rmse']:.4f} test_r2={t['r2']:.4f}")


if __name__ == "__main__":
    main()

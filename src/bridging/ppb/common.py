from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    mean_error = float(np.mean(err))
    pearson_r = _safe_pearson(y_true, y_pred)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))

    return {
        "n": int(y_true.size),
        "mae": mae,
        "rmse": rmse,
        "mean_error": mean_error,
        "pearson_r": pearson_r,
        "r2": r2,
    }


def recursive_to(obj, device: str):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    if isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    if isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}
    return obj


def aggregate_complex(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["complex", "dG_true", "dG_pred", "n_rows"])
    return (
        df.groupby("complex", as_index=False)
        .agg(
            dG_true=("dG_true", "mean"),
            dG_pred=("dG_pred", "mean"),
            n_rows=("dG_pred", "size"),
        )
        .sort_values("complex")
        .reset_index(drop=True)
    )


def evaluate_loader(model, loader, device: str) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = recursive_to(batch, device)
            _, out = model(batch)
            preds = out["dG_pred"].detach().cpu().view(-1).numpy()
            truth = out["dG_true"].detach().cpu().view(-1).numpy()
            complexes = list(batch["complex"])
            for complex_id, y, yhat in zip(complexes, truth.tolist(), preds.tolist()):
                rows.append(
                    {
                        "complex": str(complex_id).upper(),
                        "dG_true": float(y),
                        "dG_pred": float(yhat),
                    }
                )
    return pd.DataFrame(rows)


def build_model_cfg(EasyDict, *, node_feat_dim, pair_feat_dim, num_layers, max_num_atoms):
    return EasyDict(
        {
            "encoder": EasyDict(
                {
                    "node_feat_dim": int(node_feat_dim),
                    "pair_feat_dim": int(pair_feat_dim),
                    "num_layers": int(num_layers),
                }
            ),
            "max_num_atoms": int(max_num_atoms),
            "num_classes": 1,
        }
    )


def common_transforms(args):
    train_tf_cfg = [
        {"type": "select_atom", "resolution": "full"},
        {"type": "add_atom_noise", "noise_std": float(args.atom_noise_std)},
        {"type": "add_chi_angle_noise", "noise_std": float(args.chi_noise_std)},
        {
            "type": "selected_region_fixed_size_patch",
            "select_attr": "itf_flag",
            "patch_size": int(args.patch_size),
        },
    ]
    val_tf_cfg = [
        {"type": "select_atom", "resolution": "full"},
        {
            "type": "selected_region_fixed_size_patch",
            "select_attr": "itf_flag",
            "patch_size": int(args.patch_size),
        },
    ]
    return train_tf_cfg, val_tf_cfg


def mode_summary_and_write(
    *,
    mode_name: str,
    out_dir: Path,
    pred_rows_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    args,
    csv_path: Path,
) -> dict:
    pred_rows_path = out_dir / f"{mode_name}_predictions_rows.csv"
    pred_rows_df.to_csv(pred_rows_path, index=False)

    pred_complex_df = (
        pred_rows_df.groupby(["fold", "split", "complex"], as_index=False)
        .agg(dG_true=("dG_true", "mean"), dG_pred=("dG_pred", "mean"), n_rows=("dG_pred", "size"))
        .sort_values(["fold", "split", "complex"])
        .reset_index(drop=True)
    )
    pred_complex_path = out_dir / f"{mode_name}_predictions_complex.csv"
    pred_complex_df.to_csv(pred_complex_path, index=False)

    fold_metrics_path = out_dir / f"{mode_name}_metrics_by_fold.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    summary = {
        "mode": mode_name,
        "csv_path": str(csv_path.resolve()),
        "num_cvfolds": int(args.num_cvfolds),
        "only_fold": None if getattr(args, "only_fold", None) is None else int(args.only_fold),
        "epochs": int(args.epochs),
        "max_iters": int(getattr(args, "max_iters", 0)),
        "val_freq": int(getattr(args, "val_freq", 0)),
        "batch_size": int(args.batch_size),
        "predictions_rows_csv": str(pred_rows_path.resolve()),
        "predictions_complex_csv": str(pred_complex_path.resolve()),
        "metrics_by_fold_csv": str(fold_metrics_path.resolve()),
    }
    for split_name in ("train", "test"):
        split_fold = fold_metrics_df[fold_metrics_df["split"] == split_name]
        split_complex = pred_complex_df[pred_complex_df["split"] == split_name]
        summary[f"{split_name}_metrics_mean"] = {
            k: float(np.nanmean(split_fold[k].to_numpy()))
            for k in ["mae", "rmse", "mean_error", "pearson_r", "r2"]
        }
        summary[f"{split_name}_metrics_std"] = {
            k: float(np.nanstd(split_fold[k].to_numpy()))
            for k in ["mae", "rmse", "mean_error", "pearson_r", "r2"]
        }
        summary[f"{split_name}_metrics_pooled_complex"] = regression_metrics(
            split_complex["dG_true"].to_numpy(),
            split_complex["dG_pred"].to_numpy(),
        )

    summary_path = out_dir / f"{mode_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PPB][{mode_name}] summary={summary_path}")
    return summary

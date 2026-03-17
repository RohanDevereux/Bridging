from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import regression_metrics


def _fit_ols_with_intercept(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X1 = np.column_stack([np.ones((X.shape[0],), dtype=np.float64), X])
    w, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return w


def _predict_ols_with_intercept(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X1 = np.column_stack([np.ones((X.shape[0],), dtype=np.float64), X])
    return X1 @ w


def run_ensemble_mode(
    *,
    out_dir: Path,
    baseline_pred_complex_csv: Path,
    frame_aug_pred_complex_csv: Path,
    trajectory_pred_complex_csv: Path,
) -> dict:
    base = pd.read_csv(baseline_pred_complex_csv).rename(columns={"dG_pred": "pred_baseline"})
    frame = pd.read_csv(frame_aug_pred_complex_csv).rename(columns={"dG_pred": "pred_frame_aug"})
    traj = pd.read_csv(trajectory_pred_complex_csv).rename(columns={"dG_pred": "pred_trajectory"})

    key = ["fold", "split", "complex"]
    merged = base[key + ["dG_true", "pred_baseline"]].merge(
        frame[key + ["dG_true", "pred_frame_aug"]],
        on=key,
        how="inner",
        suffixes=("_base", "_frame"),
    )
    merged = merged.merge(
        traj[key + ["dG_true", "pred_trajectory"]],
        on=key,
        how="inner",
    )
    merged["dG_true"] = merged[["dG_true_base", "dG_true_frame", "dG_true"]].mean(axis=1)
    merged = merged.drop(columns=["dG_true_base", "dG_true_frame"])
    merged = merged.sort_values(key).reset_index(drop=True)

    pred_rows = []
    fold_metrics = []
    weights_rows = []

    for fold in sorted(merged["fold"].unique().tolist()):
        train_df = merged[(merged["fold"] == fold) & (merged["split"] == "train")].copy()
        test_df = merged[(merged["fold"] == fold) & (merged["split"] == "test")].copy()
        cols = ["pred_baseline", "pred_frame_aug", "pred_trajectory"]

        w = _fit_ols_with_intercept(train_df[cols].to_numpy(), train_df["dG_true"].to_numpy())
        weights_rows.append(
            {
                "fold": int(fold),
                "intercept": float(w[0]),
                "w_baseline": float(w[1]),
                "w_frame_aug": float(w[2]),
                "w_trajectory": float(w[3]),
            }
        )

        train_df["dG_pred_ensemble"] = _predict_ols_with_intercept(train_df[cols].to_numpy(), w)
        test_df["dG_pred_ensemble"] = _predict_ols_with_intercept(test_df[cols].to_numpy(), w)

        for split_name, split_df in [("train", train_df), ("test", test_df)]:
            pred = split_df.rename(columns={"dG_pred_ensemble": "dG_pred"})[
                ["fold", "split", "complex", "dG_true", "dG_pred"]
            ]
            pred_rows.append(pred)
            metrics = regression_metrics(split_df["dG_true"].to_numpy(), split_df["dG_pred_ensemble"].to_numpy())
            metrics.update({"mode": "ensemble", "fold": int(fold), "split": split_name})
            fold_metrics.append(metrics)
            print(
                f"[PPB][ensemble] fold={fold} split={split_name} "
                f"n={metrics['n']} rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
                f"r={metrics['pearson_r']:.4f} r2={metrics['r2']:.4f} me={metrics['mean_error']:.4f}"
            )

    pred_rows_df = pd.concat(pred_rows, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics).sort_values(["split", "fold"]).reset_index(drop=True)
    weights_df = pd.DataFrame(weights_rows).sort_values("fold").reset_index(drop=True)

    pred_rows_csv = out_dir / "ensemble_predictions_complex.csv"
    pred_rows_df.to_csv(pred_rows_csv, index=False)
    metrics_csv = out_dir / "ensemble_metrics_by_fold.csv"
    fold_metrics_df.to_csv(metrics_csv, index=False)
    weights_csv = out_dir / "ensemble_weights_by_fold.csv"
    weights_df.to_csv(weights_csv, index=False)

    summary = {
        "mode": "ensemble",
        "predictions_complex_csv": str(pred_rows_csv.resolve()),
        "metrics_by_fold_csv": str(metrics_csv.resolve()),
        "weights_by_fold_csv": str(weights_csv.resolve()),
    }
    for split_name in ("train", "test"):
        split_fold = fold_metrics_df[fold_metrics_df["split"] == split_name]
        split_pred = pred_rows_df[pred_rows_df["split"] == split_name]
        summary[f"{split_name}_metrics_mean"] = {
            k: float(np.nanmean(split_fold[k].to_numpy()))
            for k in ["mae", "rmse", "mean_error", "pearson_r", "r2"]
        }
        summary[f"{split_name}_metrics_std"] = {
            k: float(np.nanstd(split_fold[k].to_numpy()))
            for k in ["mae", "rmse", "mean_error", "pearson_r", "r2"]
        }
        summary[f"{split_name}_metrics_pooled_complex"] = regression_metrics(
            split_pred["dG_true"].to_numpy(),
            split_pred["dG_pred"].to_numpy(),
        )

    summary_path = out_dir / "ensemble_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PPB][ensemble] summary={summary_path}")
    return summary

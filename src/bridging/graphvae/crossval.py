from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .regress import run_linear_probe
from .train import train_masked_graph_vae


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _fmt_seconds(seconds: float) -> str:
    s = max(0, int(round(float(seconds))))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _kfold_partitions(ids: list[str], *, n_splits: int, seed: int) -> list[list[str]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >=2")
    if len(ids) < n_splits:
        raise ValueError(f"n_splits={n_splits} requires at least {n_splits} samples, got {len(ids)}")
    shuffled = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for i, cid in enumerate(shuffled):
        folds[i % n_splits].append(cid)
    return folds


def _split_train_val_test(
    *,
    trainval_ids: list[str],
    test_ids: list[str],
    val_fraction_of_trainval: float,
    seed: int,
) -> dict[str, str]:
    if not 0.0 < float(val_fraction_of_trainval) < 1.0:
        raise ValueError("val_fraction_of_trainval must be in (0,1)")
    ids = list(trainval_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_val = int(round(float(val_fraction_of_trainval) * n))
    n_val = min(max(n_val, 1), max(n - 1, 1))
    val_ids = set(ids[:n_val])
    train_ids = [cid for cid in ids if cid not in val_ids]
    if not train_ids:
        moved = next(iter(val_ids))
        val_ids.remove(moved)
        train_ids = [moved]
    split: dict[str, str] = {}
    for cid in train_ids:
        split[cid] = "train"
    for cid in val_ids:
        split[cid] = "val"
    for cid in test_ids:
        split[cid] = "test"
    return split


def run_vae_crossval(
    *,
    records_path: Path,
    out_dir: Path,
    n_splits: int,
    n_repeats: int,
    val_fraction_of_trainval: float,
    seed: int,
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
    num_workers: int,
    train_checkpoint_every: int,
    alpha_grid: list[float],
    bootstrap: int,
    ridge_cv_folds: int,
    ridge_cv_repeats: int,
    ridge_cv_inner_folds: int,
) -> dict:
    if n_splits < 2 or n_repeats < 1:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    records = torch.load(records_path, map_location="cpu")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Invalid/empty records file for VAE CV: {records_path}")
    complex_ids = sorted({str(r["complex_id"]) for r in records})
    if len(complex_ids) < n_splits:
        raise RuntimeError(
            f"Cannot run VAE CV with n_splits={n_splits}; only {len(complex_ids)} unique complexes available."
        )

    fold_rows: list[dict] = []
    t0 = time.perf_counter()
    total_fits = n_splits * n_repeats * 2  # S and SD
    fit_idx = 0

    for rep in range(1, n_repeats + 1):
        folds = _kfold_partitions(complex_ids, n_splits=n_splits, seed=seed + rep)
        for fold_i in range(1, n_splits + 1):
            test_ids = folds[fold_i - 1]
            trainval_ids = [cid for j, f in enumerate(folds, start=1) if j != fold_i for cid in f]
            split_map = _split_train_val_test(
                trainval_ids=trainval_ids,
                test_ids=test_ids,
                val_fraction_of_trainval=val_fraction_of_trainval,
                seed=seed + rep * 1000 + fold_i,
            )

            fold_records = []
            for rec in records:
                cid = str(rec["complex_id"])
                if cid not in split_map:
                    continue
                rec_fold = dict(rec)
                rec_fold["split"] = split_map[cid]
                fold_records.append(rec_fold)

            fold_dir = out_dir / f"repeat_{rep:02d}" / f"fold_{fold_i:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_records_path = fold_dir / "graph_records_fold.pt"
            torch.save(fold_records, fold_records_path)

            for mode in ("S", "SD"):
                fit_idx += 1
                mode_seed = seed + rep * 10_000 + fold_i * 100 + (0 if mode == "S" else 1)
                mode_dir = fold_dir / f"mode_{mode}"
                elapsed = time.perf_counter() - t0
                print(
                    f"[CV] fit {fit_idx}/{total_fits} mode={mode} repeat={rep}/{n_repeats} "
                    f"fold={fold_i}/{n_splits} elapsed={_fmt_seconds(elapsed)}"
                )
                train_summary = train_masked_graph_vae(
                    records_path=fold_records_path,
                    out_dir=mode_dir,
                    mode=mode,
                    device=device,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    mask_ratio=mask_ratio,
                    lr=lr,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    beta_start=beta_start,
                    beta_end=beta_end,
                    beta_anneal_fraction=beta_anneal_fraction,
                    corr_weight=corr_weight,
                    seed=mode_seed,
                    num_workers=num_workers,
                    checkpoint_every=train_checkpoint_every,
                )
                reg_summary = run_linear_probe(
                    latents_csv=Path(train_summary["latents_csv"]),
                    out_dir=mode_dir,
                    alpha_grid=alpha_grid,
                    bootstrap=bootstrap,
                    ridge_cv_folds=ridge_cv_folds,
                    ridge_cv_repeats=ridge_cv_repeats,
                    ridge_cv_inner_folds=ridge_cv_inner_folds,
                    seed=mode_seed,
                )
                test = reg_summary["split_metrics"]["test"]
                fold_rows.append(
                    {
                        "repeat": int(rep),
                        "fold": int(fold_i),
                        "mode": mode,
                        "n_train_graphs": int(train_summary["n_train_graphs"]),
                        "n_val_graphs": int(train_summary["n_val_graphs"]),
                        "n_test_graphs": int(train_summary["n_test_graphs"]),
                        "best_epoch": int(train_summary["best_epoch"]),
                        "best_val_recon": float(train_summary["best_val_recon"]),
                        "ridge_alpha": float(reg_summary["alpha"]),
                        "test_rmse": float(test["rmse"]),
                        "test_mae": float(test["mae"]),
                        "test_r2": float(test["r2"]),
                        "test_pearson_r": float(test["pearson_r"]),
                        "test_spearman_r": float(test["spearman_r"]),
                        "fold_dir": str(fold_dir),
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    fold_csv = out_dir / "vae_cv_fold_metrics.csv"
    fold_df.to_csv(fold_csv, index=False)

    metrics = ["test_rmse", "test_mae", "test_r2", "test_pearson_r", "test_spearman_r", "best_val_recon"]
    mode_summary: dict[str, dict] = {}
    for mode in ("S", "SD"):
        sub = fold_df[fold_df["mode"] == mode]
        mode_summary[mode] = {
            "n_fits": int(len(sub)),
            **{m: _summary_stats(sub[m].to_numpy(dtype=np.float64)) for m in metrics},
        }

    delta_summary = {}
    if {"S", "SD"}.issubset(set(fold_df["mode"].unique().tolist())):
        piv = fold_df.pivot_table(index=["repeat", "fold"], columns="mode", values=["test_rmse", "test_r2", "test_pearson_r"])
        if not piv.empty:
            rmse_delta = piv[("test_rmse", "SD")] - piv[("test_rmse", "S")]
            r2_delta = piv[("test_r2", "SD")] - piv[("test_r2", "S")]
            pear_delta = piv[("test_pearson_r", "SD")] - piv[("test_pearson_r", "S")]
            delta_summary = {
                "delta_rmse_SD_minus_S": _summary_stats(rmse_delta.to_numpy(dtype=np.float64)),
                "delta_r2_SD_minus_S": _summary_stats(r2_delta.to_numpy(dtype=np.float64)),
                "delta_pearson_SD_minus_S": _summary_stats(pear_delta.to_numpy(dtype=np.float64)),
            }

    summary = {
        "records_path": str(records_path),
        "out_dir": str(out_dir),
        "n_complexes": int(len(complex_ids)),
        "n_splits": int(n_splits),
        "n_repeats": int(n_repeats),
        "n_total_mode_fits": int(total_fits),
        "val_fraction_of_trainval": float(val_fraction_of_trainval),
        "fold_metrics_csv": str(fold_csv),
        "mode_summary": mode_summary,
        "delta_summary": delta_summary,
        "note": "Each fit retrains VAE from scratch and evaluates ridge on outer held-out fold.",
    }
    summary_path = out_dir / "vae_cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


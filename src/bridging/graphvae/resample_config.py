from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .crossval import _kfold_partitions, _split_train_val_test
from .regress import run_linear_probe
from .supervised_baseline import run_supervised_baseline
from .train import train_masked_graph_vae


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _fmt_seconds(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_alpha_grid(raw: str) -> list[float]:
    out = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("alpha_grid must contain at least one value.")
    return out


def _resplit_records(records: list[dict], split_map: dict[str, str]) -> list[dict]:
    out: list[dict] = []
    for rec in records:
        cid = str(rec["complex_id"])
        if cid not in split_map:
            continue
        rec_fold = dict(rec)
        rec_fold["split"] = split_map[cid]
        out.append(rec_fold)
    return out


def run_resampled_config(
    *,
    records_path: Path,
    out_dir: Path,
    model_family: str,
    repeats: int,
    folds: int,
    val_fraction_of_trainval: float,
    seed: int,
    dataset_csv: Path | None,
    mmgbsa_csv: Path | None,
    mode: str,
    supervision_mode: str,
    affinity_weight: float,
    target_policy: str,
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
    checkpoint_every: int,
    alpha_grid: list[float],
    bootstrap: int,
    ridge_cv_folds: int,
    ridge_cv_repeats: int,
    ridge_cv_inner_folds: int,
) -> dict:
    records = torch.load(records_path, map_location="cpu")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Invalid/empty records file: {records_path}")

    complex_ids = sorted({str(r["complex_id"]) for r in records})
    if len(complex_ids) < int(folds):
        raise RuntimeError(
            f"Cannot run repeated CV with folds={folds}; only {len(complex_ids)} unique complexes available."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    total_fits = int(repeats) * int(folds)
    t0 = time.perf_counter()
    fit_idx = 0

    effective_affinity_weight = (
        float(affinity_weight) if str(supervision_mode) == "semi_supervised" and str(model_family) == "vae_ridge" else 0.0
    )

    for rep in range(1, int(repeats) + 1):
        partitions = _kfold_partitions(complex_ids, n_splits=int(folds), seed=int(seed) + rep)
        for fold_i in range(1, int(folds) + 1):
            fit_idx += 1
            test_ids = partitions[fold_i - 1]
            trainval_ids = [cid for j, part in enumerate(partitions, start=1) if j != fold_i for cid in part]
            split_map = _split_train_val_test(
                trainval_ids=trainval_ids,
                test_ids=test_ids,
                val_fraction_of_trainval=float(val_fraction_of_trainval),
                seed=int(seed) + rep * 1000 + fold_i,
            )
            fold_records = _resplit_records(records, split_map)
            fold_dir = out_dir / f"repeat_{rep:02d}" / f"fold_{fold_i:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_records_path = fold_dir / "graph_records_fold.pt"
            torch.save(fold_records, fold_records_path)

            elapsed = time.perf_counter() - t0
            print(
                f"[RESAMPLE] fit {fit_idx}/{total_fits} family={model_family} repeat={rep}/{repeats} "
                f"fold={fold_i}/{folds} elapsed={_fmt_seconds(elapsed)}"
            )

            fit_seed = int(seed) + rep * 10_000 + fold_i * 100
            if model_family == "vae_ridge":
                train_summary = train_masked_graph_vae(
                    records_path=fold_records_path,
                    out_dir=fold_dir,
                    mode=mode,
                    device=device,
                    latent_dim=int(latent_dim),
                    hidden_dim=int(hidden_dim),
                    num_layers=int(num_layers),
                    mask_ratio=float(mask_ratio),
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                    batch_size=int(batch_size),
                    max_epochs=int(max_epochs),
                    patience=int(patience),
                    beta_start=float(beta_start),
                    beta_end=float(beta_end),
                    beta_anneal_fraction=float(beta_anneal_fraction),
                    corr_weight=float(corr_weight),
                    affinity_weight=float(effective_affinity_weight),
                    target_policy=str(target_policy),
                    seed=int(fit_seed),
                    num_workers=int(num_workers),
                    checkpoint_every=int(checkpoint_every),
                )
                reg_summary = run_linear_probe(
                    latents_csv=Path(train_summary["latents_csv"]),
                    out_dir=fold_dir,
                    dataset_csv=dataset_csv,
                    mmgbsa_csv=mmgbsa_csv,
                    alpha_grid=alpha_grid,
                    bootstrap=int(bootstrap),
                    ridge_cv_folds=int(ridge_cv_folds),
                    ridge_cv_repeats=int(ridge_cv_repeats),
                    ridge_cv_inner_folds=int(ridge_cv_inner_folds),
                    seed=int(fit_seed),
                )
                row = {
                    "repeat": int(rep),
                    "fold": int(fold_i),
                    "model_family": str(model_family),
                    "mode": str(mode),
                    "supervision_mode": str(supervision_mode),
                    "affinity_weight": float(effective_affinity_weight),
                    "target_policy": str(target_policy),
                    "latent_dim": int(latent_dim),
                    "hidden_dim": int(hidden_dim),
                    "num_layers": int(num_layers),
                    "n_train_graphs": int(train_summary["n_train_graphs"]),
                    "n_val_graphs": int(train_summary["n_val_graphs"]),
                    "n_test_graphs": int(train_summary["n_test_graphs"]),
                    "best_epoch": int(train_summary["best_epoch"]),
                    "best_val_recon": float(train_summary["best_val_recon"]),
                    "best_val_objective": float(train_summary["best_val_objective"]),
                    "ridge_alpha": float(reg_summary["alpha"]),
                    "train_rmse": float(reg_summary["split_metrics"]["train"]["rmse"]),
                    "val_rmse": float(reg_summary["split_metrics"]["val"]["rmse"]),
                    "test_rmse": float(reg_summary["split_metrics"]["test"]["rmse"]),
                    "train_r2": float(reg_summary["split_metrics"]["train"]["r2"]),
                    "val_r2": float(reg_summary["split_metrics"]["val"]["r2"]),
                    "test_r2": float(reg_summary["split_metrics"]["test"]["r2"]),
                    "train_pearson_r": float(reg_summary["split_metrics"]["train"]["pearson_r"]),
                    "val_pearson_r": float(reg_summary["split_metrics"]["val"]["pearson_r"]),
                    "test_pearson_r": float(reg_summary["split_metrics"]["test"]["pearson_r"]),
                    "test_train_rmse_gap": float(reg_summary["split_metrics"]["test"]["rmse"])
                    - float(reg_summary["split_metrics"]["train"]["rmse"]),
                    "train_val_rmse_gap": float(reg_summary["split_metrics"]["val"]["rmse"])
                    - float(reg_summary["split_metrics"]["train"]["rmse"]),
                    "head_train_rmse": float((train_summary.get("affinity_head_split_metrics") or {}).get("train", {}).get("rmse", float("nan"))),
                    "head_val_rmse": float((train_summary.get("affinity_head_split_metrics") or {}).get("val", {}).get("rmse", float("nan"))),
                    "head_test_rmse": float((train_summary.get("affinity_head_split_metrics") or {}).get("test", {}).get("rmse", float("nan"))),
                    "head_train_r2": float((train_summary.get("affinity_head_split_metrics") or {}).get("train", {}).get("r2", float("nan"))),
                    "head_val_r2": float((train_summary.get("affinity_head_split_metrics") or {}).get("val", {}).get("r2", float("nan"))),
                    "head_test_r2": float((train_summary.get("affinity_head_split_metrics") or {}).get("test", {}).get("r2", float("nan"))),
                    "train_summary_json": str(fold_dir / f"train_summary_{mode}.json"),
                    "ridge_summary_json": str(fold_dir / "latent_ridge_summary.json"),
                    "fold_records_path": str(fold_records_path),
                }
            elif model_family == "supervised_baseline":
                base_summary = run_supervised_baseline(
                    records_path=fold_records_path,
                    out_dir=fold_dir,
                    mode=mode,
                    device=device,
                    hidden_dim=int(hidden_dim),
                    num_layers=int(num_layers),
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                    batch_size=int(batch_size),
                    max_epochs=int(max_epochs),
                    patience=int(patience),
                    seed=int(fit_seed),
                    num_workers=int(num_workers),
                    checkpoint_every=int(checkpoint_every),
                )
                row = {
                    "repeat": int(rep),
                    "fold": int(fold_i),
                    "model_family": str(model_family),
                    "mode": str(mode),
                    "supervision_mode": "supervised",
                    "affinity_weight": float("nan"),
                    "target_policy": "",
                    "latent_dim": float("nan"),
                    "hidden_dim": int(hidden_dim),
                    "num_layers": int(num_layers),
                    "n_train_graphs": int(sum(1 for r in fold_records if r["split"] == "train")),
                    "n_val_graphs": int(sum(1 for r in fold_records if r["split"] == "val")),
                    "n_test_graphs": int(sum(1 for r in fold_records if r["split"] == "test")),
                    "best_epoch": int(base_summary["best_epoch"]),
                    "best_val_recon": float("nan"),
                    "best_val_objective": float(base_summary["best_val_mse"]),
                    "ridge_alpha": float("nan"),
                    "train_rmse": float(base_summary["metrics"]["train"]["rmse"]),
                    "val_rmse": float(base_summary["metrics"]["val"]["rmse"]),
                    "test_rmse": float(base_summary["metrics"]["test"]["rmse"]),
                    "train_r2": float(base_summary["metrics"]["train"]["r2"]),
                    "val_r2": float(base_summary["metrics"]["val"]["r2"]),
                    "test_r2": float(base_summary["metrics"]["test"]["r2"]),
                    "train_pearson_r": float(base_summary["metrics"]["train"]["pearson_r"]),
                    "val_pearson_r": float(base_summary["metrics"]["val"]["pearson_r"]),
                    "test_pearson_r": float(base_summary["metrics"]["test"]["pearson_r"]),
                    "test_train_rmse_gap": float(base_summary["metrics"]["test"]["rmse"])
                    - float(base_summary["metrics"]["train"]["rmse"]),
                    "train_val_rmse_gap": float(base_summary["metrics"]["val"]["rmse"])
                    - float(base_summary["metrics"]["train"]["rmse"]),
                    "head_train_rmse": float("nan"),
                    "head_val_rmse": float("nan"),
                    "head_test_rmse": float("nan"),
                    "head_train_r2": float("nan"),
                    "head_val_r2": float("nan"),
                    "head_test_r2": float("nan"),
                    "train_summary_json": str(fold_dir / f"supervised_baseline_summary_{mode}.json"),
                    "ridge_summary_json": "",
                    "fold_records_path": str(fold_records_path),
                }
            else:
                raise ValueError(f"Unsupported model_family={model_family}")

            rows.append(row)
            print(
                f"[RESAMPLE] done family={row['model_family']} mode={row['mode']} "
                f"repeat={rep} fold={fold_i} test_rmse={row['test_rmse']:.4f} test_r2={row['test_r2']:.4f}"
            )

    fold_df = pd.DataFrame(rows)
    fold_csv = out_dir / "resample_fold_metrics.csv"
    fold_df.to_csv(fold_csv, index=False)

    summary = {
        "records_path": str(records_path),
        "out_dir": str(out_dir),
        "model_family": str(model_family),
        "mode": str(mode),
        "supervision_mode": str(supervision_mode if model_family == "vae_ridge" else "supervised"),
        "affinity_weight": float(effective_affinity_weight) if model_family == "vae_ridge" else None,
        "target_policy": str(target_policy) if model_family == "vae_ridge" else None,
        "latent_dim": (int(latent_dim) if model_family == "vae_ridge" else None),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "n_complexes": int(len(complex_ids)),
        "n_repeats": int(repeats),
        "n_splits": int(folds),
        "n_total_fits": int(len(rows)),
        "val_fraction_of_trainval": float(val_fraction_of_trainval),
        "fold_metrics_csv": str(fold_csv),
        "metric_summary": {
            metric: _summary_stats(fold_df[metric].to_list())
            for metric in [
                "train_rmse",
                "val_rmse",
                "test_rmse",
                "train_r2",
                "val_r2",
                "test_r2",
                "train_pearson_r",
                "val_pearson_r",
                "test_pearson_r",
                "test_train_rmse_gap",
                "train_val_rmse_gap",
            ]
        },
        "note": "Each fit retrains from scratch on a new outer split. Test metrics are outer-fold held-out metrics.",
    }
    summary_path = out_dir / "resample_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated outer-fold evaluation for one GraphVAE or baseline config.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-family", choices=["vae_ridge", "supervised_baseline"], default="vae_ridge")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction-of-trainval", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset")
    parser.add_argument("--mmgbsa")
    parser.add_argument("--mode", choices=["S", "SD"], required=True)
    parser.add_argument("--supervision-mode", choices=["unsupervised", "semi_supervised"], default="unsupervised")
    parser.add_argument("--affinity-weight", type=float, default=1.0)
    parser.add_argument("--target-policy", default="shared_static")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--beta-anneal-fraction", type=float, default=0.30)
    parser.add_argument("--corr-weight", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--alpha-grid", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--ridge-cv-folds", type=int, default=0)
    parser.add_argument("--ridge-cv-repeats", type=int, default=0)
    parser.add_argument("--ridge-cv-inner-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_resampled_config(
        records_path=Path(args.records),
        out_dir=Path(args.out_dir),
        model_family=str(args.model_family),
        repeats=int(args.repeats),
        folds=int(args.folds),
        val_fraction_of_trainval=float(args.val_fraction_of_trainval),
        seed=int(args.seed),
        dataset_csv=(Path(args.dataset) if args.dataset else None),
        mmgbsa_csv=(Path(args.mmgbsa) if args.mmgbsa else None),
        mode=str(args.mode),
        supervision_mode=str(args.supervision_mode),
        affinity_weight=float(args.affinity_weight),
        target_policy=str(args.target_policy),
        device=str(args.device),
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
        num_workers=int(args.num_workers),
        checkpoint_every=int(args.checkpoint_every),
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
        bootstrap=int(args.bootstrap),
        ridge_cv_folds=int(args.ridge_cv_folds),
        ridge_cv_repeats=int(args.ridge_cv_repeats),
        ridge_cv_inner_folds=int(args.ridge_cv_inner_folds),
    )
    test_rmse = summary["metric_summary"]["test_rmse"]["mean"]
    test_r2 = summary["metric_summary"]["test_r2"]["mean"]
    print(
        f"[RESAMPLE] family={summary['model_family']} mode={summary['mode']} "
        f"mean_test_rmse={test_rmse:.4f} mean_test_r2={test_r2:.4f}"
    )
    print(f"[RESAMPLE] summary={Path(args.out_dir) / 'resample_summary.json'}")


if __name__ == "__main__":
    main()

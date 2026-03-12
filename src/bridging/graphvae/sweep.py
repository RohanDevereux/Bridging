from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .regress import run_linear_probe
from .train import train_masked_graph_vae


def _fmt_seconds(seconds: float) -> str:
    s = max(0, int(round(float(seconds))))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _parse_int_list(raw: str) -> list[int]:
    out = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _parse_supervision_modes(raw: str) -> list[str]:
    allowed = {"unsupervised", "semi_supervised"}
    out = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"Unsupported supervision mode: {token}")
        out.append(token)
    if not out:
        raise ValueError("Expected at least one supervision mode.")
    return out


def _safe_metric(summary: dict, split: str, metric: str) -> float:
    return float((((summary.get("split_metrics") or {}).get(split) or {}).get(metric, float("nan"))))


def _safe_head_metric(train_summary: dict, split: str, metric: str) -> float:
    return float((((train_summary.get("affinity_head_split_metrics") or {}).get(split) or {}).get(metric, float("nan"))))


def _safe_cv_metric(summary: dict, metric: str) -> float:
    pooled = (summary.get("repeated_kfold") or {}).get("pooled_heldout_metrics") or {}
    return float(pooled.get(metric, float("nan")))


def _parse_feature_modes(raw: str) -> list[str]:
    allowed = {"S", "SD"}
    out = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"Unsupported feature mode: {token}")
        out.append(token)
    if not out:
        raise ValueError("Expected at least one feature mode.")
    return out


def _config_name(*, mode: str, latent_dim: int, supervision_mode: str) -> str:
    short = "semi" if supervision_mode == "semi_supervised" else "unsup"
    return f"{mode}_z{int(latent_dim):02d}_{short}"


def _markdown_report(*, rows: list[dict], records_path: Path, out_dir: Path, modes: list[str], affinity_weight: float) -> str:
    if not rows:
        return "# GraphVAE Sweep\n\nNo runs completed.\n"
    df = pd.DataFrame(rows).sort_values(["test_rmse", "val_rmse"], na_position="last").reset_index(drop=True)
    best = df.iloc[0]
    lines = [
        "# GraphVAE Sweep",
        "",
        f"- Records: `{records_path}`",
        f"- Feature modes: `{','.join(modes)}`",
        f"- Configs: `{len(df)}`",
        f"- Semi-supervised affinity weight: `{affinity_weight:.3f}`",
        "",
        "## Best By Test RMSE",
        "",
        f"- Experiment: `{best['experiment']}`",
        f"- Feature mode: `{best['mode']}`",
        f"- Latent dim: `{int(best['latent_dim'])}`",
        f"- Supervision: `{best['supervision_mode']}`",
        f"- Test RMSE: `{best['test_rmse']:.4f}`",
        f"- Test R2: `{best['test_r2']:.4f}`",
        f"- Test Pearson r: `{best['test_pearson_r']:.4f}`",
        "",
        "## Summary",
        "",
        "| Experiment | Mode | z | Supervision | Train RMSE | Val RMSE | Test RMSE | Test R2 | Test r | CV RMSE |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.to_dict("records"):
        lines.append(
            "| "
            + f"{row['experiment']} | "
            + f"{row['mode']} | "
            + f"{int(row['latent_dim'])} | "
            + f"{row['supervision_mode']} | "
            + f"{row['train_rmse']:.4f} | "
            + f"{row['val_rmse']:.4f} | "
            + f"{row['test_rmse']:.4f} | "
            + f"{row['test_r2']:.4f} | "
            + f"{row['test_pearson_r']:.4f} | "
            + f"{row['cv_pooled_rmse']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_saved_graph_sweep(
    *,
    records_path: Path,
    out_dir: Path,
    modes: list[str],
    latent_dims: list[int],
    supervision_modes: list[str],
    affinity_weight: float,
    device: str,
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
    checkpoint_every: int,
    dataset_csv: Path | None,
    mmgbsa_csv: Path | None,
    alpha_grid: list[float],
    bootstrap: int,
    ridge_cv_folds: int,
    ridge_cv_repeats: int,
    ridge_cv_inner_folds: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    run_t0 = time.perf_counter()
    total_configs = len(modes) * len(latent_dims) * len(supervision_modes)
    config_idx = 0

    for mode in modes:
        for latent_dim in latent_dims:
            for supervision_mode in supervision_modes:
                config_idx += 1
                config_affinity_weight = float(affinity_weight) if supervision_mode == "semi_supervised" else 0.0
                experiment = _config_name(mode=mode, latent_dim=latent_dim, supervision_mode=supervision_mode)
                experiment_dir = out_dir / experiment
                print(
                    f"[SWEEP] config {config_idx}/{total_configs} experiment={experiment} "
                    f"mode={mode} latent_dim={latent_dim} supervision={supervision_mode} "
                    f"affinity_weight={config_affinity_weight:.3f}"
                )
                train_summary = train_masked_graph_vae(
                    records_path=records_path,
                    out_dir=experiment_dir,
                    mode=mode,
                    device=device,
                    latent_dim=int(latent_dim),
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
                    affinity_weight=config_affinity_weight,
                    seed=seed,
                    num_workers=num_workers,
                    checkpoint_every=checkpoint_every,
                )
                reg_summary = run_linear_probe(
                    latents_csv=Path(train_summary["latents_csv"]),
                    out_dir=experiment_dir,
                    dataset_csv=dataset_csv,
                    mmgbsa_csv=mmgbsa_csv,
                    alpha_grid=alpha_grid,
                    bootstrap=bootstrap,
                    ridge_cv_folds=ridge_cv_folds,
                    ridge_cv_repeats=ridge_cv_repeats,
                    ridge_cv_inner_folds=ridge_cv_inner_folds,
                    seed=seed,
                )
                row = {
                    "experiment": experiment,
                    "experiment_dir": str(experiment_dir),
                    "mode": mode,
                    "latent_dim": int(latent_dim),
                    "supervision_mode": supervision_mode,
                    "affinity_weight": float(config_affinity_weight),
                    "best_epoch": int(train_summary["best_epoch"]),
                    "best_val_recon": float(train_summary["best_val_recon"]),
                    "best_val_objective": float(train_summary["best_val_objective"]),
                    "ridge_alpha": float(reg_summary["alpha"]),
                    "train_rmse": _safe_metric(reg_summary, "train", "rmse"),
                    "train_mae": _safe_metric(reg_summary, "train", "mae"),
                    "train_r2": _safe_metric(reg_summary, "train", "r2"),
                    "train_pearson_r": _safe_metric(reg_summary, "train", "pearson_r"),
                    "val_rmse": _safe_metric(reg_summary, "val", "rmse"),
                    "val_mae": _safe_metric(reg_summary, "val", "mae"),
                    "val_r2": _safe_metric(reg_summary, "val", "r2"),
                    "val_pearson_r": _safe_metric(reg_summary, "val", "pearson_r"),
                    "test_rmse": _safe_metric(reg_summary, "test", "rmse"),
                    "test_mae": _safe_metric(reg_summary, "test", "mae"),
                    "test_r2": _safe_metric(reg_summary, "test", "r2"),
                    "test_pearson_r": _safe_metric(reg_summary, "test", "pearson_r"),
                    "cv_pooled_rmse": _safe_cv_metric(reg_summary, "rmse"),
                    "cv_pooled_r2": _safe_cv_metric(reg_summary, "r2"),
                    "cv_pooled_pearson_r": _safe_cv_metric(reg_summary, "pearson_r"),
                    "train_val_rmse_gap": _safe_metric(reg_summary, "val", "rmse") - _safe_metric(reg_summary, "train", "rmse"),
                    "test_train_rmse_gap": _safe_metric(reg_summary, "test", "rmse") - _safe_metric(reg_summary, "train", "rmse"),
                    "head_train_rmse": _safe_head_metric(train_summary, "train", "rmse"),
                    "head_val_rmse": _safe_head_metric(train_summary, "val", "rmse"),
                    "head_test_rmse": _safe_head_metric(train_summary, "test", "rmse"),
                    "head_train_r2": _safe_head_metric(train_summary, "train", "r2"),
                    "head_val_r2": _safe_head_metric(train_summary, "val", "r2"),
                    "head_test_r2": _safe_head_metric(train_summary, "test", "r2"),
                    "train_summary_json": str(experiment_dir / f"train_summary_{mode}.json"),
                    "ridge_summary_json": str(experiment_dir / "latent_ridge_summary.json"),
                }
                rows.append(row)
                elapsed = time.perf_counter() - run_t0
                print(
                    f"[SWEEP] done experiment={experiment} test_rmse={row['test_rmse']:.4f} "
                    f"test_r2={row['test_r2']:.4f} elapsed={_fmt_seconds(elapsed)}"
                )

    results_df = pd.DataFrame(rows).sort_values(["test_rmse", "val_rmse"], na_position="last").reset_index(drop=True)
    summary_csv = out_dir / "sweep_summary.csv"
    summary_json = out_dir / "sweep_summary.json"
    report_md = out_dir / "sweep_report.md"
    results_df.to_csv(summary_csv, index=False)

    summary = {
        "records_path": str(records_path),
        "out_dir": str(out_dir),
        "modes": list(modes),
        "latent_dims": [int(x) for x in latent_dims],
        "supervision_modes": list(supervision_modes),
        "affinity_weight": float(affinity_weight),
        "n_configs": int(len(rows)),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "best_by_test_rmse": ({} if results_df.empty else results_df.iloc[0].to_dict()),
        "results": rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_md.write_text(
        _markdown_report(
            rows=rows,
            records_path=records_path,
            out_dir=out_dir,
            modes=modes,
            affinity_weight=affinity_weight,
        ),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent-dimension and supervision sweep on saved GraphVAE records.")
    parser.add_argument("--records", required=True, help="Prepared graph_records.pt path.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--modes", default="S,SD")
    parser.add_argument("--mode", choices=["S", "SD"], help="Backward-compatible alias for a single feature mode.")
    parser.add_argument("--latent-dims", default="8,16,32,64")
    parser.add_argument("--supervision-modes", default="unsupervised,semi_supervised")
    parser.add_argument("--affinity-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--dataset")
    parser.add_argument("--mmgbsa")
    parser.add_argument("--alpha-grid", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--ridge-cv-folds", type=int, default=5)
    parser.add_argument("--ridge-cv-repeats", type=int, default=3)
    parser.add_argument("--ridge-cv-inner-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    alphas = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]
    summary = run_saved_graph_sweep(
        records_path=Path(args.records),
        out_dir=Path(args.out_dir),
        modes=([str(args.mode)] if args.mode else _parse_feature_modes(args.modes)),
        latent_dims=_parse_int_list(args.latent_dims),
        supervision_modes=_parse_supervision_modes(args.supervision_modes),
        affinity_weight=float(args.affinity_weight),
        device=str(args.device),
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
        checkpoint_every=int(args.checkpoint_every),
        dataset_csv=(Path(args.dataset) if args.dataset else None),
        mmgbsa_csv=(Path(args.mmgbsa) if args.mmgbsa else None),
        alpha_grid=alphas,
        bootstrap=int(args.bootstrap),
        ridge_cv_folds=int(args.ridge_cv_folds),
        ridge_cv_repeats=int(args.ridge_cv_repeats),
        ridge_cv_inner_folds=int(args.ridge_cv_inner_folds),
    )
    best = summary.get("best_by_test_rmse") or {}
    if best:
        print(
            f"[SWEEP] best experiment={best['experiment']} test_rmse={float(best['test_rmse']):.4f} "
            f"test_r2={float(best['test_r2']):.4f}"
        )
    print(f"[SWEEP] summary={Path(args.out_dir) / 'sweep_summary.json'}")
    print(f"[SWEEP] report={Path(args.out_dir) / 'sweep_report.md'}")


if __name__ == "__main__":
    main()

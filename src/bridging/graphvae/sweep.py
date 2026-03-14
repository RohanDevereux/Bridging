from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import SUPPORTED_TARGET_POLICIES
from .record_views import (
    DEFAULT_INTERFACE_POLICY,
    SUBGROUP_ORDER,
    SUPPORTED_INTERFACE_POLICIES,
    load_complex_metadata,
    resolve_graph_view_variants,
    subgroup_map_from_metadata,
)
from .regress import run_linear_probe
from .supervised_baseline import run_supervised_baseline
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


def _parse_name_list(raw: str, *, allowed: set[str], label: str) -> list[str]:
    out = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"Unsupported {label}: {token}")
        out.append(token)
    if not out:
        raise ValueError(f"Expected at least one {label}.")
    return out


def _parse_optional_name_list(raw: str | None, *, allowed: set[str], label: str) -> list[str]:
    if raw is None:
        return []
    return _parse_name_list(raw, allowed=allowed, label=label)


def _safe_metric(summary: dict, split: str, metric: str) -> float:
    return float((((summary.get("split_metrics") or {}).get(split) or {}).get(metric, float("nan"))))


def _safe_head_metric(train_summary: dict, split: str, metric: str) -> float:
    return float((((train_summary.get("affinity_head_split_metrics") or {}).get(split) or {}).get(metric, float("nan"))))


def _safe_cv_metric(summary: dict, metric: str) -> float:
    pooled = (summary.get("repeated_kfold") or {}).get("pooled_heldout_metrics") or {}
    return float(pooled.get(metric, float("nan")))


def _safe_subgroup_metric(summary: dict, subgroup: str, split: str, metric: str) -> float:
    return float(
        ((((summary.get("subgroups") or {}).get("overall_model_split_metrics") or {}).get(subgroup) or {}).get(split) or {}).get(
            metric,
            float("nan"),
        )
    )


def _safe_subgroup_model_metric(summary: dict, subgroup: str, split: str, metric: str) -> float:
    return float(
        (((((summary.get("subgroups") or {}).get("subgroup_specific_models") or {}).get(subgroup) or {}).get("split_metrics") or {}).get(split) or {}).get(
            metric,
            float("nan"),
        )
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 1:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
        }
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    if y_true.size < 2:
        r2 = float("nan")
        pearson_r = float("nan")
    else:
        centered = y_true - float(np.mean(y_true))
        sst = float(np.sum(centered * centered))
        r2 = float(1.0 - np.sum(err * err) / sst) if sst > 0 else float("nan")
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
    }


def _baseline_subgroup_metrics(predictions_csv: Path, subgroup_by_complex: dict[str, str]) -> dict[str, dict[str, float]]:
    pred_df = pd.read_csv(predictions_csv)
    pred_df["subgroup"] = pred_df["complex_id"].map(subgroup_by_complex).fillna("other")
    test_df = pred_df[pred_df["split"] == "test"].copy()
    out: dict[str, dict[str, float]] = {}
    for subgroup in SUBGROUP_ORDER:
        group_df = test_df[test_df["subgroup"] == subgroup]
        out[subgroup] = _regression_metrics(
            group_df["dG"].to_numpy(dtype=np.float64),
            group_df["dG_pred"].to_numpy(dtype=np.float64),
        )
    return out


def _policy_tag(interface_policy: str | None) -> str:
    if not interface_policy:
        return "default"
    return str(interface_policy).replace("-", "_")


def _config_name(
    *,
    model_family: str,
    graph_view: str,
    interface_policy: str | None,
    mode: str,
    latent_dim: int | None,
    supervision_mode: str,
    target_policy: str | None,
    baseline_hidden_dim: int | None = None,
    baseline_num_layers: int | None = None,
) -> str:
    prefix = f"{graph_view}_{_policy_tag(interface_policy)}"
    if model_family == "supervised_baseline":
        return (
            f"{prefix}_{mode}_base_"
            f"hd{int(baseline_hidden_dim):03d}_"
            f"l{int(baseline_num_layers)}"
        )
    short = "semi" if supervision_mode == "semi_supervised" else "unsup"
    policy = str(target_policy).replace("-", "_")
    return f"{prefix}_{mode}_z{int(latent_dim):02d}_{short}_{policy}"


def _markdown_report(
    *,
    rows: list[dict],
    records_path: Path,
    graph_views: list[str],
    modes: list[str],
    affinity_weight: float,
    target_policies: list[str],
    interface_policies: list[str],
    match_interface_subset: bool,
    include_supervised_baseline: bool,
) -> str:
    if not rows:
        return "# GraphVAE Sweep\n\nNo runs completed.\n"
    df = pd.DataFrame(rows).sort_values(["is_stable", "test_rmse", "val_rmse"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
    best = df.iloc[0]
    lines = [
        "# GraphVAE Sweep",
        "",
        f"- Records: `{records_path}`",
        f"- Graph views: `{','.join(graph_views)}`",
        f"- Feature modes: `{','.join(modes)}`",
        f"- Configs: `{len(df)}`",
        f"- Semi-supervised affinity weight: `{affinity_weight:.3f}`",
        f"- Target policies: `{','.join(target_policies)}`",
        f"- Interface policies: `{','.join(interface_policies)}`",
        f"- Match full to interface-retained subset: `{bool(match_interface_subset)}`",
        f"- Include supervised baseline: `{bool(include_supervised_baseline)}`",
        "",
        "## Best By Test RMSE",
        "",
        f"- Experiment: `{best['experiment']}`",
        f"- Model family: `{best.get('model_family', 'vae_ridge')}`",
        f"- Graph view: `{best['graph_view']}`",
        f"- Interface policy: `{best.get('interface_policy', '')}`",
        f"- Feature mode: `{best['mode']}`",
        f"- Latent dim: `{best['latent_dim']}`",
        f"- Supervision: `{best['supervision_mode']}`",
        f"- Stable: `{bool(best['is_stable'])}`",
        f"- Test RMSE: `{best['test_rmse']:.4f}`",
        f"- Test R2: `{best['test_r2']:.4f}`",
        f"- Test Pearson r: `{best['test_pearson_r']:.4f}`",
        "",
        "## Summary",
        "",
        "| Experiment | Family | View | Interface Policy | Mode | z | Supervision | Target Policy | Stable | Train RMSE | Val RMSE | Test RMSE | Test R2 | Test r |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.to_dict("records"):
        lines.append(
            "| "
            + f"{row['experiment']} | "
            + f"{row.get('model_family', 'vae_ridge')} | "
            + f"{row['graph_view']} | "
            + f"{row.get('interface_policy', '')} | "
            + f"{row['mode']} | "
            + f"{row.get('latent_dim', '')} | "
            + f"{row['supervision_mode']} | "
            + f"{row.get('target_policy', '')} | "
            + f"{bool(row['is_stable'])} | "
            + f"{row['train_rmse']:.4f} | "
            + f"{row['val_rmse']:.4f} | "
            + f"{row['test_rmse']:.4f} | "
            + f"{row['test_r2']:.4f} | "
            + f"{row['test_pearson_r']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_saved_graph_sweep(
    *,
    records_path: Path,
    out_dir: Path,
    graph_views: list[str],
    interface_policies: list[str],
    modes: list[str],
    latent_dims: list[int],
    supervision_modes: list[str],
    target_policies: list[str],
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
    pdb_cache_root: Path | None,
    md_root: Path | None,
    prebuilt_view_root: Path | None,
    mmgbsa_csv: Path | None,
    alpha_grid: list[float],
    bootstrap: int,
    ridge_cv_folds: int,
    ridge_cv_repeats: int,
    ridge_cv_inner_folds: int,
    match_interface_subset: bool,
    include_supervised_baseline: bool,
    baseline_hidden_dims: list[int],
    baseline_num_layers_list: list[int],
    baseline_lr: float,
    baseline_weight_decay: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(view != "full" for view in graph_views) and dataset_csv is None:
        raise ValueError("--dataset is required when using graph views other than full.")

    subgroup_by_complex: dict[str, str] = {}
    if dataset_csv is not None:
        subgroup_by_complex = subgroup_map_from_metadata(load_complex_metadata(Path(dataset_csv)))

    view_root = prebuilt_view_root if prebuilt_view_root is not None else (out_dir / "record_views")
    if prebuilt_view_root is not None:
        print(f"[SWEEP] using prebuilt view root={view_root}")
    view_variants, view_reports = resolve_graph_view_variants(
        records_path=records_path,
        dataset_csv=dataset_csv,
        graph_views=graph_views,
        interface_policies=interface_policies,
        view_root=view_root,
        pdb_cache_root=pdb_cache_root,
        md_root=md_root,
        match_interface_subset=match_interface_subset,
        reuse_existing=(prebuilt_view_root is not None),
        progress_every=25,
        log_prefix="[VIEW]",
    )

    rows: list[dict] = []
    run_t0 = time.perf_counter()
    total_view_datasets = sum(len([view for view in graph_views if view in variant["paths"]]) for variant in view_variants)
    total_configs = total_view_datasets * len(modes) * len(latent_dims) * len(supervision_modes) * len(target_policies)
    if include_supervised_baseline:
        total_configs += total_view_datasets * len(modes) * len(baseline_hidden_dims) * len(baseline_num_layers_list)
    config_idx = 0

    for variant in view_variants:
        interface_policy = variant["interface_policy"]
        for graph_view in graph_views:
            if graph_view not in variant["paths"]:
                continue
            view_records = variant["paths"][graph_view]
            for mode in modes:
                for latent_dim in latent_dims:
                    for supervision_mode in supervision_modes:
                        for target_policy in target_policies:
                            config_idx += 1
                            config_affinity_weight = float(affinity_weight) if supervision_mode == "semi_supervised" else 0.0
                            experiment = _config_name(
                                model_family="vae_ridge",
                                graph_view=graph_view,
                                interface_policy=interface_policy,
                                mode=mode,
                                latent_dim=latent_dim,
                                supervision_mode=supervision_mode,
                                target_policy=target_policy,
                            )
                            experiment_dir = out_dir / experiment
                            print(
                                f"[SWEEP] config {config_idx}/{total_configs} experiment={experiment} "
                                f"family=vae_ridge graph_view={graph_view} interface_policy={interface_policy} "
                                f"mode={mode} latent_dim={latent_dim} supervision={supervision_mode} "
                                f"affinity_weight={config_affinity_weight:.3f} target_policy={target_policy}"
                            )
                            train_summary = train_masked_graph_vae(
                                records_path=view_records,
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
                                target_policy=target_policy,
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
                                "model_family": "vae_ridge",
                                "experiment": experiment,
                                "experiment_dir": str(experiment_dir),
                                "graph_view": graph_view,
                                "interface_policy": interface_policy,
                                "mode": mode,
                                "latent_dim": int(latent_dim),
                                "baseline_hidden_dim": float("nan"),
                                "baseline_num_layers": float("nan"),
                                "supervision_mode": supervision_mode,
                                "affinity_weight": float(config_affinity_weight),
                                "target_policy": target_policy,
                                "best_epoch": int(train_summary["best_epoch"]),
                                "best_val_recon": float(train_summary["best_val_recon"]),
                                "best_val_objective": float(train_summary["best_val_objective"]),
                                "is_stable": bool(train_summary.get("is_stable", True)),
                                "n_epochs_with_skipped_batches": int(train_summary.get("n_epochs_with_skipped_batches", 0)),
                                "unstable_reason": train_summary.get("unstable_reason"),
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
                            for subgroup in SUBGROUP_ORDER:
                                row[f"{subgroup}_global_test_rmse"] = _safe_subgroup_metric(reg_summary, subgroup, "test", "rmse")
                                row[f"{subgroup}_global_test_r2"] = _safe_subgroup_metric(reg_summary, subgroup, "test", "r2")
                                row[f"{subgroup}_subgroup_model_test_rmse"] = _safe_subgroup_model_metric(
                                    reg_summary,
                                    subgroup,
                                    "test",
                                    "rmse",
                                )
                                row[f"{subgroup}_subgroup_model_test_r2"] = _safe_subgroup_model_metric(
                                    reg_summary,
                                    subgroup,
                                    "test",
                                    "r2",
                                )
                            rows.append(row)
                            elapsed = time.perf_counter() - run_t0
                            print(
                                f"[SWEEP] done experiment={experiment} stable={row['is_stable']} "
                                f"test_rmse={row['test_rmse']:.4f} test_r2={row['test_r2']:.4f} "
                                f"elapsed={_fmt_seconds(elapsed)}"
                            )

    if include_supervised_baseline:
        for variant in view_variants:
            interface_policy = variant["interface_policy"]
            for graph_view in graph_views:
                if graph_view not in variant["paths"]:
                    continue
                view_records = variant["paths"][graph_view]
                for mode in modes:
                    for base_hidden_dim in baseline_hidden_dims:
                        for base_num_layers in baseline_num_layers_list:
                            config_idx += 1
                            experiment = _config_name(
                                model_family="supervised_baseline",
                                graph_view=graph_view,
                                interface_policy=interface_policy,
                                mode=mode,
                                latent_dim=None,
                                supervision_mode="supervised",
                                target_policy=None,
                                baseline_hidden_dim=base_hidden_dim,
                                baseline_num_layers=base_num_layers,
                            )
                            experiment_dir = out_dir / experiment
                            print(
                                f"[SWEEP] config {config_idx}/{total_configs} experiment={experiment} "
                                f"family=supervised_baseline graph_view={graph_view} interface_policy={interface_policy} "
                                f"mode={mode} hidden_dim={base_hidden_dim} num_layers={base_num_layers}"
                            )
                            base_summary = run_supervised_baseline(
                                records_path=view_records,
                                out_dir=experiment_dir,
                                mode=mode,
                                device=device,
                                hidden_dim=int(base_hidden_dim),
                                num_layers=int(base_num_layers),
                                lr=float(baseline_lr),
                                weight_decay=float(baseline_weight_decay),
                                batch_size=batch_size,
                                max_epochs=max_epochs,
                                patience=patience,
                                seed=seed,
                                num_workers=num_workers,
                                checkpoint_every=checkpoint_every,
                            )
                            subgroup_metrics = (
                                _baseline_subgroup_metrics(Path(base_summary["predictions_csv"]), subgroup_by_complex)
                                if subgroup_by_complex
                                else {}
                            )
                            row = {
                                "model_family": "supervised_baseline",
                                "experiment": experiment,
                                "experiment_dir": str(experiment_dir),
                                "graph_view": graph_view,
                                "interface_policy": interface_policy,
                                "mode": mode,
                                "latent_dim": float("nan"),
                                "baseline_hidden_dim": int(base_hidden_dim),
                                "baseline_num_layers": int(base_num_layers),
                                "supervision_mode": "supervised",
                                "affinity_weight": float("nan"),
                                "target_policy": "",
                                "best_epoch": int(base_summary["best_epoch"]),
                                "best_val_recon": float("nan"),
                                "best_val_objective": float(base_summary["best_val_mse"]),
                                "is_stable": True,
                                "n_epochs_with_skipped_batches": 0,
                                "unstable_reason": "",
                                "ridge_alpha": float("nan"),
                                "train_rmse": float(base_summary["metrics"]["train"]["rmse"]),
                                "train_mae": float(base_summary["metrics"]["train"]["mae"]),
                                "train_r2": float(base_summary["metrics"]["train"]["r2"]),
                                "train_pearson_r": float(base_summary["metrics"]["train"]["pearson_r"]),
                                "val_rmse": float(base_summary["metrics"]["val"]["rmse"]),
                                "val_mae": float(base_summary["metrics"]["val"]["mae"]),
                                "val_r2": float(base_summary["metrics"]["val"]["r2"]),
                                "val_pearson_r": float(base_summary["metrics"]["val"]["pearson_r"]),
                                "test_rmse": float(base_summary["metrics"]["test"]["rmse"]),
                                "test_mae": float(base_summary["metrics"]["test"]["mae"]),
                                "test_r2": float(base_summary["metrics"]["test"]["r2"]),
                                "test_pearson_r": float(base_summary["metrics"]["test"]["pearson_r"]),
                                "cv_pooled_rmse": float("nan"),
                                "cv_pooled_r2": float("nan"),
                                "cv_pooled_pearson_r": float("nan"),
                                "train_val_rmse_gap": float(base_summary["metrics"]["val"]["rmse"]) - float(base_summary["metrics"]["train"]["rmse"]),
                                "test_train_rmse_gap": float(base_summary["metrics"]["test"]["rmse"]) - float(base_summary["metrics"]["train"]["rmse"]),
                                "head_train_rmse": float("nan"),
                                "head_val_rmse": float("nan"),
                                "head_test_rmse": float("nan"),
                                "head_train_r2": float("nan"),
                                "head_val_r2": float("nan"),
                                "head_test_r2": float("nan"),
                                "train_summary_json": str(experiment_dir / f"supervised_baseline_summary_{mode}.json"),
                                "ridge_summary_json": "",
                            }
                            for subgroup in SUBGROUP_ORDER:
                                metrics = subgroup_metrics.get(subgroup, {})
                                row[f"{subgroup}_global_test_rmse"] = float(metrics.get("rmse", float("nan")))
                                row[f"{subgroup}_global_test_r2"] = float(metrics.get("r2", float("nan")))
                                row[f"{subgroup}_subgroup_model_test_rmse"] = float("nan")
                                row[f"{subgroup}_subgroup_model_test_r2"] = float("nan")
                            rows.append(row)
                            elapsed = time.perf_counter() - run_t0
                            print(
                                f"[SWEEP] done experiment={experiment} stable=True "
                                f"test_rmse={row['test_rmse']:.4f} test_r2={row['test_r2']:.4f} "
                                f"elapsed={_fmt_seconds(elapsed)}"
                            )

    results_df = pd.DataFrame(rows).sort_values(["is_stable", "test_rmse", "val_rmse"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
    summary_csv = out_dir / "sweep_summary.csv"
    summary_json = out_dir / "sweep_summary.json"
    report_md = out_dir / "sweep_report.md"
    results_df.to_csv(summary_csv, index=False)

    summary = {
        "records_path": str(records_path),
        "out_dir": str(out_dir),
        "graph_views": list(graph_views),
        "interface_policies": list(interface_policies),
        "view_root": str(view_root),
        "used_prebuilt_views": bool(prebuilt_view_root is not None),
        "modes": list(modes),
        "latent_dims": [int(x) for x in latent_dims],
        "supervision_modes": list(supervision_modes),
        "target_policies": list(target_policies),
        "affinity_weight": float(affinity_weight),
        "match_interface_subset": bool(match_interface_subset),
        "include_supervised_baseline": bool(include_supervised_baseline),
        "baseline_hidden_dims": [int(x) for x in baseline_hidden_dims],
        "baseline_num_layers_list": [int(x) for x in baseline_num_layers_list],
        "baseline_lr": float(baseline_lr),
        "baseline_weight_decay": float(baseline_weight_decay),
        "pdb_cache_root": (None if pdb_cache_root is None else str(pdb_cache_root)),
        "md_root": (None if md_root is None else str(md_root)),
        "n_configs": int(len(rows)),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "view_reports": view_reports,
        "best_by_test_rmse": ({} if results_df.empty else results_df.iloc[0].to_dict()),
        "results": rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_md.write_text(
        _markdown_report(
            rows=rows,
            records_path=records_path,
            graph_views=graph_views,
            modes=modes,
            affinity_weight=affinity_weight,
            target_policies=target_policies,
            interface_policies=(interface_policies or [DEFAULT_INTERFACE_POLICY]),
            match_interface_subset=match_interface_subset,
            include_supervised_baseline=include_supervised_baseline,
        ),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent-dimension and supervision sweep on saved GraphVAE records.")
    parser.add_argument("--records", required=True, help="Prepared graph_records.pt path.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--graph-views", default="full")
    parser.add_argument("--interface-policies", default=DEFAULT_INTERFACE_POLICY)
    parser.add_argument("--modes", default="S,SD")
    parser.add_argument("--mode", choices=["S", "SD"], help="Backward-compatible alias for a single feature mode.")
    parser.add_argument("--latent-dims", default="8,16,32,64")
    parser.add_argument("--supervision-modes", default="unsupervised,semi_supervised")
    parser.add_argument("--affinity-weight", type=float, default=1.0)
    parser.add_argument("--target-policies", default="shared_static")
    parser.add_argument("--target-policy", choices=list(SUPPORTED_TARGET_POLICIES), help="Backward-compatible alias for a single target policy.")
    parser.add_argument(
        "--match-interface-subset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When sweeping full and interface together, filter full-view records to the interface-retained complex IDs.",
    )
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
    parser.add_argument("--pdb-cache-root")
    parser.add_argument("--md-root")
    parser.add_argument("--prebuilt-view-root")
    parser.add_argument("--mmgbsa")
    parser.add_argument("--alpha-grid", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--ridge-cv-folds", type=int, default=5)
    parser.add_argument("--ridge-cv-repeats", type=int, default=3)
    parser.add_argument("--ridge-cv-inner-folds", type=int, default=5)
    parser.add_argument(
        "--include-supervised-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the direct supervised GNN baseline grid in the same sweep.",
    )
    parser.add_argument("--baseline-hidden-dims", default="128,256")
    parser.add_argument("--baseline-num-layers-list", default="3,4")
    parser.add_argument("--baseline-lr", type=float, default=3e-4)
    parser.add_argument("--baseline-weight-decay", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    alphas = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]
    graph_views = _parse_name_list(args.graph_views, allowed={"full", "interface"}, label="graph view")
    interface_policies = (
        _parse_name_list(args.interface_policies, allowed=set(SUPPORTED_INTERFACE_POLICIES), label="interface policy")
        if "interface" in graph_views
        else []
    )
    target_policies = (
        [str(args.target_policy)]
        if args.target_policy
        else _parse_name_list(args.target_policies, allowed=set(SUPPORTED_TARGET_POLICIES), label="target policy")
    )
    summary = run_saved_graph_sweep(
        records_path=Path(args.records),
        out_dir=Path(args.out_dir),
        graph_views=graph_views,
        interface_policies=interface_policies,
        modes=([str(args.mode)] if args.mode else _parse_name_list(args.modes, allowed={"S", "SD"}, label="feature mode")),
        latent_dims=_parse_int_list(args.latent_dims),
        supervision_modes=_parse_name_list(
            args.supervision_modes,
            allowed={"unsupervised", "semi_supervised"},
            label="supervision mode",
        ),
        target_policies=target_policies,
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
        pdb_cache_root=(Path(args.pdb_cache_root) if args.pdb_cache_root else None),
        md_root=(Path(args.md_root) if args.md_root else None),
        prebuilt_view_root=(Path(args.prebuilt_view_root) if args.prebuilt_view_root else None),
        mmgbsa_csv=(Path(args.mmgbsa) if args.mmgbsa else None),
        alpha_grid=alphas,
        bootstrap=int(args.bootstrap),
        ridge_cv_folds=int(args.ridge_cv_folds),
        ridge_cv_repeats=int(args.ridge_cv_repeats),
        ridge_cv_inner_folds=int(args.ridge_cv_inner_folds),
        match_interface_subset=bool(args.match_interface_subset),
        include_supervised_baseline=bool(args.include_supervised_baseline),
        baseline_hidden_dims=_parse_int_list(args.baseline_hidden_dims),
        baseline_num_layers_list=_parse_int_list(args.baseline_num_layers_list),
        baseline_lr=float(args.baseline_lr),
        baseline_weight_decay=float(args.baseline_weight_decay),
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

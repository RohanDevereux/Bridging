from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DEFAULT_CVAE_CHECKPOINT, DEFAULT_DATASET, DEFAULT_OUT_DIR
from .dataset import build_records, collect_feature_index, load_prodigy_map
from .latents import build_latent_cache, load_cvae_checkpoint
from .model import fit_regressor, regression_metrics


def _split_mask(df: pd.DataFrame, split_name: str):
    split = df["split"].astype(str).str.lower()
    if split_name == "train":
        return split != "test"
    return split == "test"


def _count_by_split(df: pd.DataFrame, mask: pd.Series):
    train_n = int((mask & _split_mask(df, "train")).sum())
    test_n = int((mask & _split_mask(df, "test")).sum())
    return train_n, test_n


def _warn_availability(df: pd.DataFrame, mask: pd.Series, label: str):
    total_train, total_test = _count_by_split(df, pd.Series(True, index=df.index))
    avail_train, avail_test = _count_by_split(df, mask)
    if avail_train == 0:
        raise RuntimeError(
            f"No train rows available for {label}. "
            f"Available counts train={avail_train}/{total_train} test={avail_test}/{total_test}."
        )
    if avail_test == 0:
        print(
            f"[WARN] {label}: train={avail_train}/{total_train}, test={avail_test}/{total_test}. "
            "No test rows currently available; training continues."
        )
    else:
        print(f"[INFO] {label}: train={avail_train}/{total_train}, test={avail_test}/{total_test}")


def _matrix_from_rows(df: pd.DataFrame, pooled_by_row: dict[int, np.ndarray], mask: pd.Series):
    rows = df[mask].copy()
    keep = rows["row_index"].map(lambda x: int(x) in pooled_by_row)
    rows = rows[keep]
    if rows.empty:
        return rows, np.zeros((0, 0), dtype=np.float32)
    X = np.stack([pooled_by_row[int(i)] for i in rows["row_index"].tolist()], axis=0).astype(np.float32)
    return rows, X


def _metric_frame(df: pd.DataFrame, estimate_col: str):
    mask = df["experimental_delta_g"].notna() & df[estimate_col].notna()
    y = df.loc[mask, "experimental_delta_g"].to_numpy(dtype=np.float32)
    p = df.loc[mask, estimate_col].to_numpy(dtype=np.float32)
    return regression_metrics(y, p)


def _summary_tables(df: pd.DataFrame):
    out = {}
    for split_name in ["train", "test"]:
        part = df[_split_mask(df, split_name)]
        out[split_name] = {
            "prodigy": _metric_frame(part, "prodigy_estimate"),
            "direct_latent": _metric_frame(part, "estimate_direct_from_latent"),
            "corrected_prodigy": _metric_frame(part, "estimate_corrected_from_prodigy"),
        }
    return out


def _print_summary(summary: dict):
    for split_name in ["train", "test"]:
        stats = summary[split_name]
        print(f"[SUMMARY] split={split_name}")
        for key in ["prodigy", "direct_latent", "corrected_prodigy"]:
            s = stats[key]
            mae = s["mae"]
            rmse = s["rmse"]
            r2 = s["r2"]
            p = s["pearson"]
            print(
                f"  - {key:18s} n={s['n']:3d} "
                f"mae={mae:.3f} rmse={rmse:.3f} r2={r2:.3f} pearson={p:.3f}"
                if s["n"] > 0
                else f"  - {key:18s} n=0"
            )


def _save_head_state(path: Path, direct, correction):
    payload = {}
    if direct is not None:
        payload["direct"] = {
            "mean": direct.mean,
            "std": direct.std,
            "state_dict": direct.model.state_dict(),
            "val_loss": direct.val_loss,
            "train_count": direct.train_count,
        }
    if correction is not None:
        payload["correction"] = {
            "mean": correction.mean,
            "std": correction.std,
            "state_dict": correction.model.state_dict(),
            "val_loss": correction.val_loss,
            "train_count": correction.train_count,
        }
    torch.save(payload, path)


def run(
    *,
    dataset_path: str | Path = DEFAULT_DATASET,
    cvae_checkpoint: str | Path = DEFAULT_CVAE_CHECKPOINT,
    features: list[str] | None = None,
    prodigy_path: str | Path | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    latents_dir: str | Path | None = None,
    latent_batch_size: int = 256,
    overwrite_latents: bool = False,
    weight_decay: float = 1e-2,
    lr: float = 1e-2,
    steps: int = 2000,
    patience: int = 200,
    seed: int = 0,
    device: str | None = None,
):
    dataset_path = Path(dataset_path)
    cvae_checkpoint = Path(cvae_checkpoint)
    if not cvae_checkpoint.exists():
        raise FileNotFoundError(f"CVAE checkpoint not found: {cvae_checkpoint}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = Path(latents_dir) if latents_dir else out_dir / f"{dataset_path.stem}_latents"

    feature_index, duplicates = collect_feature_index(features, dataset_stem=dataset_path.stem)
    if duplicates:
        print(f"[WARN] duplicate feature files for {len(duplicates)} PDB IDs; selected best-matching path")

    prodigy_map = load_prodigy_map(dataset_path, prodigy_path=prodigy_path)
    df = build_records(dataset_path, feature_index, prodigy_map)
    if df.empty:
        raise RuntimeError("No dataset rows found.")

    print(
        f"[REG] dataset={dataset_path} rows={len(df)} "
        f"feature_available={int(df['feature_available'].sum())} "
        f"prodigy_available={int(df['prodigy_available'].sum())}"
    )

    model, model_device, model_cfg = load_cvae_checkpoint(cvae_checkpoint, device=device)
    print(f"[REG] loaded CVAE checkpoint={cvae_checkpoint} latent_dim={model_cfg['latent_dim']} device={model_device}")

    feature_rows = df[df["feature_available"]].to_dict("records")
    pooled_by_row = build_latent_cache(
        model,
        model_device,
        feature_rows,
        latents_dir=latents_dir,
        batch_size=latent_batch_size,
        overwrite=overwrite_latents,
    )
    df["latent_available"] = df["row_index"].map(lambda i: int(i) in pooled_by_row)

    direct_mask = df["latent_available"] & df["experimental_available"]
    _warn_availability(df, direct_mask, "direct-head training rows")
    correction_mask = direct_mask & df["prodigy_available"]
    if int(correction_mask.sum()) == 0:
        print("[WARN] correction-head training rows: none available; correction model will be skipped")
    else:
        _warn_availability(df, correction_mask, "correction-head training rows")

    train_direct_rows, X_train_direct = _matrix_from_rows(df, pooled_by_row, direct_mask & _split_mask(df, "train"))
    y_train_direct = train_direct_rows["experimental_delta_g"].to_numpy(dtype=np.float32)
    direct_model = fit_regressor(
        X_train_direct,
        y_train_direct,
        weight_decay=weight_decay,
        lr=lr,
        steps=steps,
        patience=patience,
        seed=seed,
        device=device,
    )

    correction_model = None
    if int(correction_mask.sum()) > 0:
        train_corr_rows, X_train_corr = _matrix_from_rows(
            df,
            pooled_by_row,
            correction_mask & _split_mask(df, "train"),
        )
        y_train_corr = (
            train_corr_rows["experimental_delta_g"].to_numpy(dtype=np.float32)
            - train_corr_rows["prodigy_estimate"].to_numpy(dtype=np.float32)
        )
        if len(train_corr_rows) > 0:
            correction_model = fit_regressor(
                X_train_corr,
                y_train_corr,
                weight_decay=weight_decay,
                lr=lr,
                steps=steps,
                patience=patience,
                seed=seed,
                device=device,
            )

    df["estimate_direct_from_latent"] = np.nan
    df["estimate_predicted_correction"] = np.nan
    df["estimate_corrected_from_prodigy"] = np.nan

    all_lat_rows, X_all_lat = _matrix_from_rows(df, pooled_by_row, df["latent_available"])
    if len(all_lat_rows) > 0:
        direct_pred = direct_model.predict(X_all_lat)
        df.loc[all_lat_rows.index, "estimate_direct_from_latent"] = direct_pred

    if correction_model is not None and len(all_lat_rows) > 0:
        corr_pred = correction_model.predict(X_all_lat)
        df.loc[all_lat_rows.index, "estimate_predicted_correction"] = corr_pred
        has_prod = df.loc[all_lat_rows.index, "prodigy_estimate"].notna()
        idx = all_lat_rows.index[has_prod]
        df.loc[idx, "estimate_corrected_from_prodigy"] = (
            df.loc[idx, "prodigy_estimate"] + df.loc[idx, "estimate_predicted_correction"]
        )

    summary = _summary_tables(df)
    _print_summary(summary)

    predictions_csv = out_dir / f"{dataset_path.stem}_regression_predictions.csv"
    summary_json = out_dir / f"{dataset_path.stem}_regression_summary.json"
    models_pt = out_dir / f"{dataset_path.stem}_regression_heads.pt"

    # Requested outputs:
    # - experimental_delta_g
    # - prodigy_estimate
    # - estimate_direct_from_latent
    # - estimate_corrected_from_prodigy
    df.to_csv(predictions_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_head_state(models_pt, direct_model, correction_model)

    print(f"[REG] wrote predictions: {predictions_csv}")
    print(f"[REG] wrote summary:     {summary_json}")
    print(f"[REG] wrote heads:       {models_pt}")
    return {
        "predictions_csv": predictions_csv,
        "summary_json": summary_json,
        "models_pt": models_pt,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train pooled-latent regression heads (direct and PRODIGY-correction)."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset CSV path")
    parser.add_argument("--cvae", default=str(DEFAULT_CVAE_CHECKPOINT), help="Trained CVAE checkpoint path")
    parser.add_argument(
        "--features",
        action="append",
        help="Feature directory/glob (repeatable). If omitted, scans generatedData roots.",
    )
    parser.add_argument("--prodigy", help="PRODIGY estimates CSV path")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--latents-dir", help="Latents cache directory")
    parser.add_argument("--latent-batch-size", type=int, default=256)
    parser.add_argument("--overwrite-latents", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="cuda or cpu")
    args = parser.parse_args()

    try:
        run(
            dataset_path=args.dataset,
            cvae_checkpoint=args.cvae,
            features=args.features,
            prodigy_path=args.prodigy,
            out_dir=args.out_dir,
            latents_dir=args.latents_dir,
            latent_batch_size=args.latent_batch_size,
            overwrite_latents=args.overwrite_latents,
            weight_decay=args.weight_decay,
            lr=args.lr,
            steps=args.steps,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
        )
    except Exception as exc:
        raise SystemExit(f"[REG] {exc}") from exc


if __name__ == "__main__":
    main()

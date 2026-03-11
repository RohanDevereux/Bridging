from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from bridging.regression.dataset import load_mmgbsa_map

from .ids import canonical_complex_id


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    a = pd.Series(y_true).rank(method="average")
    b = pd.Series(y_pred).rank(method="average")
    return float(a.corr(b))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "n": int(y_true.shape[0]),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if y_true.shape[0] >= 2 else float("nan"),
        "pearson_r": _pearson(y_true, y_pred),
        "spearman_r": _spearman(y_true, y_pred),
        "mean_error": float(np.mean(y_pred - y_true)),
    }


def _bootstrap_coef_ci(
    *,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, list[float]]:
    if n_bootstrap < 1:
        return {}
    rng = np.random.default_rng(seed)
    coefs = []
    intercepts = []
    n = X.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        model = Ridge(alpha=alpha)
        model.fit(X[idx], y[idx])
        coefs.append(model.coef_.copy())
        intercepts.append(float(model.intercept_))
    arr = np.asarray(coefs, dtype=np.float64)
    lower = np.percentile(arr, 2.5, axis=0).tolist()
    upper = np.percentile(arr, 97.5, axis=0).tolist()
    intercept_ci = [float(np.percentile(intercepts, 2.5)), float(np.percentile(intercepts, 97.5))]
    return {
        "coef_ci_95_lower": lower,
        "coef_ci_95_upper": upper,
        "intercept_ci_95": intercept_ci,
    }


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


def _fit_ridge(
    *,
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid: list[float],
    inner_folds: int,
):
    alphas = np.asarray(alpha_grid, dtype=np.float64)
    cv_inner = min(max(2, int(inner_folds)), int(X.shape[0]))
    if X.shape[0] >= 2 and cv_inner >= 2:
        model = RidgeCV(alphas=alphas, cv=cv_inner)
    else:
        model = Ridge(alpha=float(alphas[0]))
    model.fit(X, y)
    alpha = float(model.alpha_) if hasattr(model, "alpha_") else float(alphas[0])
    return model, alpha


def _load_mmgbsa_by_complex_id(
    *,
    dataset_csv: Path,
    mmgbsa_csv: Path,
) -> tuple[dict[str, float], dict]:
    mmgbsa_map = load_mmgbsa_map(dataset_csv, mmgbsa_path=mmgbsa_csv)
    df = pd.read_csv(dataset_csv)
    out: dict[str, float] = {}
    duplicate_complex_ids: set[str] = set()
    row_to_cid: dict[int, str] = {}
    for row_index, row in enumerate(df.to_dict("records")):
        cid = canonical_complex_id(row)
        if not cid:
            continue
        if cid in out:
            duplicate_complex_ids.add(cid)
        row_to_cid[row_index] = cid
    for row_index, value in mmgbsa_map.items():
        cid = row_to_cid.get(int(row_index))
        if not cid:
            continue
        out[cid] = float(value)
    meta = {
        "dataset_csv": str(dataset_csv),
        "mmgbsa_csv": str(mmgbsa_csv),
        "available_complexes": int(len(out)),
        "duplicate_complex_ids_in_dataset": int(len(duplicate_complex_ids)),
        "duplicate_complex_id_sample": sorted(duplicate_complex_ids)[:20],
    }
    return out, meta


def _compute_split_metrics_from_column(df: pd.DataFrame, value_col: str) -> dict:
    out = {}
    for split_name in ("train", "val", "test"):
        part = df[(df["split"] == split_name) & df["dG"].notna() & df[value_col].notna()].copy()
        if part.empty:
            out[split_name] = {"n": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "pearson_r": float("nan"), "spearman_r": float("nan"), "mean_error": float("nan")}
            continue
        out[split_name] = _metrics(
            part["dG"].to_numpy(dtype=np.float64),
            part[value_col].to_numpy(dtype=np.float64),
        )
    return out


def _repeated_kfold_probe(
    *,
    X: np.ndarray,
    y: np.ndarray,
    complex_ids: np.ndarray,
    alpha_grid: list[float],
    n_splits: int,
    n_repeats: int,
    inner_folds: int,
    seed: int,
    out_dir: Path,
    baseline_name: str | None = None,
    baseline: np.ndarray | None = None,
) -> dict:
    if n_splits < 2 or n_repeats < 1:
        return {}
    if X.shape[0] < n_splits:
        raise ValueError(
            f"Repeated K-fold requested with n_splits={n_splits}, but only {X.shape[0]} samples are available."
        )

    has_baseline = baseline is not None
    if has_baseline:
        baseline = np.asarray(baseline, dtype=np.float64)
        if baseline.shape[0] != X.shape[0]:
            raise ValueError("Baseline vector length does not match sample count.")

    fold_rows: list[dict] = []
    pred_rows: list[dict] = []

    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
            X_tr = X[tr_idx]
            y_tr = y[tr_idx]
            X_te = X[te_idx]
            y_te = y[te_idx]

            model, alpha = _fit_ridge(
                X=X_tr,
                y=y_tr,
                alpha_grid=alpha_grid,
                inner_folds=inner_folds,
            )
            y_pred = model.predict(X_te)
            direct_metrics = _metrics(y_te, y_pred)
            row = {
                "repeat": int(rep + 1),
                "fold": int(fold_id),
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "alpha": float(alpha),
                **direct_metrics,
            }

            baseline_pred = np.full_like(y_te, np.nan, dtype=np.float64)
            corrected_pred = np.full_like(y_te, np.nan, dtype=np.float64)
            correction_alpha = float("nan")
            correction_train_n = 0
            correction_test_n = 0

            if has_baseline:
                b_tr = baseline[tr_idx]
                b_te = baseline[te_idx]
                tr_mask = np.isfinite(b_tr)
                te_mask = np.isfinite(b_te)
                if tr_mask.any() and te_mask.any():
                    correction_train_n = int(tr_mask.sum())
                    correction_test_n = int(te_mask.sum())
                    corr_model, correction_alpha = _fit_ridge(
                        X=X_tr[tr_mask],
                        y=(y_tr[tr_mask] - b_tr[tr_mask]),
                        alpha_grid=alpha_grid,
                        inner_folds=inner_folds,
                    )
                    baseline_pred[te_mask] = b_te[te_mask]
                    corrected_pred[te_mask] = b_te[te_mask] + corr_model.predict(X_te[te_mask])
                    base_m = _metrics(y_te[te_mask], baseline_pred[te_mask])
                    corr_m = _metrics(y_te[te_mask], corrected_pred[te_mask])
                    row.update(
                        {
                            f"{baseline_name}_n_test": int(base_m["n"]),
                            f"{baseline_name}_rmse": float(base_m["rmse"]),
                            f"{baseline_name}_mae": float(base_m["mae"]),
                            f"{baseline_name}_r2": float(base_m["r2"]),
                            f"{baseline_name}_pearson_r": float(base_m["pearson_r"]),
                            f"{baseline_name}_spearman_r": float(base_m["spearman_r"]),
                            f"corrected_{baseline_name}_n_test": int(corr_m["n"]),
                            f"corrected_{baseline_name}_rmse": float(corr_m["rmse"]),
                            f"corrected_{baseline_name}_mae": float(corr_m["mae"]),
                            f"corrected_{baseline_name}_r2": float(corr_m["r2"]),
                            f"corrected_{baseline_name}_pearson_r": float(corr_m["pearson_r"]),
                            f"corrected_{baseline_name}_spearman_r": float(corr_m["spearman_r"]),
                            f"corrected_{baseline_name}_alpha": float(correction_alpha),
                            f"corrected_{baseline_name}_n_train": int(correction_train_n),
                        }
                    )

            fold_rows.append(row)
            for i, idx in enumerate(te_idx):
                pred_row = {
                    "repeat": int(rep + 1),
                    "fold": int(fold_id),
                    "complex_id": str(complex_ids[idx]),
                    "dG": float(y[idx]),
                    "dG_pred": float(y_pred[i]),
                    "error": float(y_pred[i] - y[idx]),
                    "alpha": float(alpha),
                }
                if has_baseline:
                    pred_row[baseline_name] = (
                        None if not np.isfinite(baseline_pred[i]) else float(baseline_pred[i])
                    )
                    pred_row[f"corrected_{baseline_name}"] = (
                        None if not np.isfinite(corrected_pred[i]) else float(corrected_pred[i])
                    )
                    pred_row[f"corrected_{baseline_name}_alpha"] = (
                        None if not np.isfinite(correction_alpha) else float(correction_alpha)
                    )
                pred_rows.append(pred_row)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    fold_csv = out_dir / "latent_ridge_cv_fold_metrics.csv"
    pred_csv = out_dir / "latent_ridge_cv_predictions.csv"
    fold_df.to_csv(fold_csv, index=False)
    pred_df.to_csv(pred_csv, index=False)

    metric_names = ["rmse", "mae", "r2", "pearson_r", "spearman_r", "mean_error"]
    summary = {name: _summary_stats(fold_df[name].to_numpy(dtype=np.float64)) for name in metric_names}
    alpha_stats = _summary_stats(fold_df["alpha"].to_numpy(dtype=np.float64))
    pooled = _metrics(
        pred_df["dG"].to_numpy(dtype=np.float64),
        pred_df["dG_pred"].to_numpy(dtype=np.float64),
    )

    out = {
        "n_splits": int(n_splits),
        "n_repeats": int(n_repeats),
        "n_fits": int(n_splits * n_repeats),
        "inner_folds": int(inner_folds),
        "fold_metrics_csv": str(fold_csv),
        "heldout_predictions_csv": str(pred_csv),
        "fold_metric_summary": summary,
        "alpha_summary": alpha_stats,
        "pooled_heldout_metrics": pooled,
        "note": "Each sample appears in held-out predictions once per repeat.",
    }

    if has_baseline:
        base_col_prefix = str(baseline_name)
        corr_col_prefix = f"corrected_{baseline_name}"
        base_fold_metrics = {}
        corr_fold_metrics = {}
        for short_name, col_suffix in (
            ("rmse", "rmse"),
            ("mae", "mae"),
            ("r2", "r2"),
            ("pearson_r", "pearson_r"),
            ("spearman_r", "spearman_r"),
        ):
            base_col = f"{base_col_prefix}_{col_suffix}"
            corr_col = f"{corr_col_prefix}_{col_suffix}"
            if base_col in fold_df.columns:
                base_fold_metrics[short_name] = _summary_stats(fold_df[base_col].to_numpy(dtype=np.float64))
            if corr_col in fold_df.columns:
                corr_fold_metrics[short_name] = _summary_stats(fold_df[corr_col].to_numpy(dtype=np.float64))

        base_pred = pred_df.dropna(subset=[baseline_name]).copy()
        corr_pred = pred_df.dropna(subset=[f"corrected_{baseline_name}"]).copy()
        out[baseline_name] = {
            "fold_metric_summary": base_fold_metrics,
            "pooled_heldout_metrics": (
                {}
                if base_pred.empty
                else _metrics(
                    base_pred["dG"].to_numpy(dtype=np.float64),
                    base_pred[baseline_name].to_numpy(dtype=np.float64),
                )
            ),
            "available_heldout_predictions": int(len(base_pred)),
        }
        out[f"corrected_{baseline_name}"] = {
            "fold_metric_summary": corr_fold_metrics,
            "pooled_heldout_metrics": (
                {}
                if corr_pred.empty
                else _metrics(
                    corr_pred["dG"].to_numpy(dtype=np.float64),
                    corr_pred[f"corrected_{baseline_name}"].to_numpy(dtype=np.float64),
                )
            ),
            "alpha_summary": (
                {}
                if f"{corr_col_prefix}_alpha" not in fold_df.columns
                else _summary_stats(fold_df[f"{corr_col_prefix}_alpha"].to_numpy(dtype=np.float64))
            ),
            "available_heldout_predictions": int(len(corr_pred)),
        }

    return out


def run_linear_probe(
    *,
    latents_csv: Path,
    out_dir: Path,
    alpha_grid: list[float],
    bootstrap: int,
    seed: int,
    ridge_cv_folds: int = 0,
    ridge_cv_repeats: int = 0,
    ridge_cv_inner_folds: int = 5,
    dataset_csv: Path | None = None,
    mmgbsa_csv: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(latents_csv)
    df = df.reset_index(drop=True)
    mu_cols = [c for c in df.columns if c.startswith("mu_")]
    if not mu_cols:
        raise ValueError("No latent columns found (mu_*)")

    if mmgbsa_csv is not None and dataset_csv is None:
        raise ValueError("--mmgbsa requires --dataset so complex_id values can be matched back to dataset rows.")

    baseline_meta = {}
    if dataset_csv is not None and mmgbsa_csv is not None:
        mmgbsa_map, baseline_meta = _load_mmgbsa_by_complex_id(dataset_csv=dataset_csv, mmgbsa_csv=mmgbsa_csv)
        df["mmgbsa_estimate"] = df["complex_id"].map(mmgbsa_map)
    else:
        df["mmgbsa_estimate"] = np.nan

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    X_train = train_df[mu_cols].to_numpy(dtype=np.float64)
    y_train = train_df["dG"].to_numpy(dtype=np.float64)

    model, alpha = _fit_ridge(
        X=X_train,
        y=y_train,
        alpha_grid=alpha_grid,
        inner_folds=ridge_cv_inner_folds,
    )

    pred_frames = []
    split_metrics = {}
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        X = split_df[mu_cols].to_numpy(dtype=np.float64)
        y = split_df["dG"].to_numpy(dtype=np.float64)
        y_pred = model.predict(X)
        m = _metrics(y, y_pred)
        split_metrics[split_name] = m
        pred_df = split_df[["complex_id", "split", "dG", "mmgbsa_estimate"]].copy()
        pred_df["row_order"] = split_df.index.to_numpy(dtype=np.int64)
        pred_df["dG_pred"] = y_pred
        pred_frames.append(pred_df)

    pred_out = pd.concat(pred_frames, ignore_index=True)

    coef_df = pd.DataFrame({"feature": mu_cols, "coef": model.coef_})
    coef_path = out_dir / "latent_ridge_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    boot = _bootstrap_coef_ci(
        X=X_train,
        y=y_train,
        alpha=alpha,
        n_bootstrap=int(bootstrap),
        seed=seed,
    )

    mmgbsa_summary = {}
    correction_coef_path = None
    correction_alpha = None
    correction_intercept = None
    correction_boot = {}
    if pred_out["mmgbsa_estimate"].notna().any():
        baseline_metrics = _compute_split_metrics_from_column(pred_out, "mmgbsa_estimate")
        train_corr_df = train_df[train_df["mmgbsa_estimate"].notna()].copy()
        if not train_corr_df.empty:
            X_train_corr = train_corr_df[mu_cols].to_numpy(dtype=np.float64)
            y_train_corr = (
                train_corr_df["dG"].to_numpy(dtype=np.float64)
                - train_corr_df["mmgbsa_estimate"].to_numpy(dtype=np.float64)
            )
            corr_model, correction_alpha = _fit_ridge(
                X=X_train_corr,
                y=y_train_corr,
                alpha_grid=alpha_grid,
                inner_folds=ridge_cv_inner_folds,
            )
            correction_intercept = float(corr_model.intercept_)
            correction_boot = _bootstrap_coef_ci(
                X=X_train_corr,
                y=y_train_corr,
                alpha=float(correction_alpha),
                n_bootstrap=int(bootstrap),
                seed=seed,
            )
            correction_coef_df = pd.DataFrame({"feature": mu_cols, "coef": corr_model.coef_})
            correction_coef_path = out_dir / "latent_ridge_mmgbsa_correction_coefficients.csv"
            correction_coef_df.to_csv(correction_coef_path, index=False)

            pred_out["dG_pred_correction_mmgbsa"] = np.nan
            pred_out["dG_pred_corrected_from_mmgbsa"] = np.nan
            has_base = pred_out["mmgbsa_estimate"].notna()
            base_row_order = pred_out.loc[has_base, "row_order"].to_numpy(dtype=np.int64)
            X_all_base = df.loc[base_row_order, mu_cols].to_numpy(dtype=np.float64)
            corr_pred = corr_model.predict(X_all_base)
            pred_out.loc[has_base, "dG_pred_correction_mmgbsa"] = corr_pred
            pred_out.loc[has_base, "dG_pred_corrected_from_mmgbsa"] = (
                pred_out.loc[has_base, "mmgbsa_estimate"].to_numpy(dtype=np.float64) + corr_pred
            )
            corrected_metrics = _compute_split_metrics_from_column(pred_out, "dG_pred_corrected_from_mmgbsa")
        else:
            pred_out["dG_pred_correction_mmgbsa"] = np.nan
            pred_out["dG_pred_corrected_from_mmgbsa"] = np.nan
            corrected_metrics = {}

        cv = _repeated_kfold_probe(
            X=df[mu_cols].to_numpy(dtype=np.float64),
            y=df["dG"].to_numpy(dtype=np.float64),
            complex_ids=df["complex_id"].to_numpy(),
            alpha_grid=alpha_grid,
            n_splits=int(ridge_cv_folds),
            n_repeats=int(ridge_cv_repeats),
            inner_folds=int(ridge_cv_inner_folds),
            seed=seed,
            out_dir=out_dir,
            baseline_name="mmgbsa",
            baseline=df["mmgbsa_estimate"].to_numpy(dtype=np.float64),
        )
        mmgbsa_summary = {
            **baseline_meta,
            "split_metrics_baseline": baseline_metrics,
            "split_metrics_corrected": corrected_metrics,
            "correction_alpha": (None if correction_alpha is None else float(correction_alpha)),
            "correction_intercept": correction_intercept,
            "correction_coefficients_csv": (None if correction_coef_path is None else str(correction_coef_path)),
            "correction_bootstrap": correction_boot,
            "repeated_kfold": {
                "baseline": (cv.get("mmgbsa") or {}),
                "corrected": (cv.get("corrected_mmgbsa") or {}),
            },
        }
    else:
        pred_out["dG_pred_correction_mmgbsa"] = np.nan
        pred_out["dG_pred_corrected_from_mmgbsa"] = np.nan
        cv = _repeated_kfold_probe(
            X=df[mu_cols].to_numpy(dtype=np.float64),
            y=df["dG"].to_numpy(dtype=np.float64),
            complex_ids=df["complex_id"].to_numpy(),
            alpha_grid=alpha_grid,
            n_splits=int(ridge_cv_folds),
            n_repeats=int(ridge_cv_repeats),
            inner_folds=int(ridge_cv_inner_folds),
            seed=seed,
            out_dir=out_dir,
        )

    pred_path = out_dir / "latent_ridge_predictions.csv"
    pred_out = pred_out.drop(columns=["row_order"], errors="ignore")
    pred_out.to_csv(pred_path, index=False)

    summary = {
        "latents_csv": str(latents_csv),
        "predictions_csv": str(pred_path),
        "coefficients_csv": str(coef_path),
        "alpha": float(alpha),
        "intercept": float(model.intercept_),
        "split_metrics": split_metrics,
        "bootstrap": boot,
        "repeated_kfold": {
            k: v
            for k, v in cv.items()
            if k not in {"mmgbsa", "corrected_mmgbsa"}
        },
        "mmgbsa": mmgbsa_summary,
    }
    summary_path = out_dir / "latent_ridge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit linear DeltaG regression on 8D VAE latents.")
    parser.add_argument("--latents-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", help="Dataset CSV used to build complex_id mapping for optional baselines.")
    parser.add_argument("--mmgbsa", help="Optional MMGBSA estimates CSV for baseline/correction evaluation.")
    parser.add_argument("--alpha-grid", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--ridge-cv-folds", type=int, default=0, help="Repeated random K-fold outer splits (0 disables).")
    parser.add_argument("--ridge-cv-repeats", type=int, default=0, help="Number of random repeats for outer K-fold.")
    parser.add_argument("--ridge-cv-inner-folds", type=int, default=5, help="Inner CV folds for alpha selection.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    alphas = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]
    summary = run_linear_probe(
        latents_csv=Path(args.latents_csv),
        out_dir=Path(args.out_dir),
        dataset_csv=(Path(args.dataset) if args.dataset else None),
        mmgbsa_csv=(Path(args.mmgbsa) if args.mmgbsa else None),
        alpha_grid=alphas,
        bootstrap=int(args.bootstrap),
        ridge_cv_folds=int(args.ridge_cv_folds),
        ridge_cv_repeats=int(args.ridge_cv_repeats),
        ridge_cv_inner_folds=int(args.ridge_cv_inner_folds),
        seed=int(args.seed),
    )
    test = summary["split_metrics"]["test"]
    print(
        f"[RIDGE] test rmse={test['rmse']:.4f} mae={test['mae']:.4f} "
        f"r2={test['r2']:.4f} r={test['pearson_r']:.4f}"
    )
    mmgbsa = summary.get("mmgbsa") or {}
    corr_test = (mmgbsa.get("split_metrics_corrected") or {}).get("test") or {}
    if corr_test.get("n", 0) > 0:
        print(
            f"[RIDGE][MMGBSA] corrected_test rmse={corr_test['rmse']:.4f} "
            f"mae={corr_test['mae']:.4f} r2={corr_test['r2']:.4f} "
            f"r={corr_test['pearson_r']:.4f}"
        )
    cv = summary.get("repeated_kfold") or {}
    if cv:
        pooled = cv["pooled_heldout_metrics"]
        print(
            f"[RIDGE][CV] fits={cv['n_fits']} "
            f"pooled_rmse={pooled['rmse']:.4f} pooled_r2={pooled['r2']:.4f} "
            f"pooled_r={pooled['pearson_r']:.4f}"
        )
    print(f"[RIDGE] summary={Path(args.out_dir) / 'latent_ridge_summary.json'}")


if __name__ == "__main__":
    main()

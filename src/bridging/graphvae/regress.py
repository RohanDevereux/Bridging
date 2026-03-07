from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
        "r2": float(r2_score(y_true, y_pred)),
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


def run_linear_probe(
    *,
    latents_csv: Path,
    out_dir: Path,
    alpha_grid: list[float],
    bootstrap: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(latents_csv)
    mu_cols = [c for c in df.columns if c.startswith("mu_")]
    if not mu_cols:
        raise ValueError("No latent columns found (mu_*)")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    X_train = train_df[mu_cols].to_numpy(dtype=np.float64)
    y_train = train_df["dG"].to_numpy(dtype=np.float64)

    model = RidgeCV(alphas=np.asarray(alpha_grid, dtype=np.float64))
    model.fit(X_train, y_train)
    alpha = float(model.alpha_)

    pred_frames = []
    split_metrics = {}
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        X = split_df[mu_cols].to_numpy(dtype=np.float64)
        y = split_df["dG"].to_numpy(dtype=np.float64)
        y_pred = model.predict(X)
        m = _metrics(y, y_pred)
        split_metrics[split_name] = m
        pred_df = split_df[["complex_id", "split", "dG"]].copy()
        pred_df["dG_pred"] = y_pred
        pred_frames.append(pred_df)

    pred_out = pd.concat(pred_frames, ignore_index=True)
    pred_path = out_dir / "latent_ridge_predictions.csv"
    pred_out.to_csv(pred_path, index=False)

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
    summary = {
        "latents_csv": str(latents_csv),
        "predictions_csv": str(pred_path),
        "coefficients_csv": str(coef_path),
        "alpha": alpha,
        "intercept": float(model.intercept_),
        "split_metrics": split_metrics,
        "bootstrap": boot,
    }
    summary_path = out_dir / "latent_ridge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit linear DeltaG regression on 8D VAE latents.")
    parser.add_argument("--latents-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha-grid", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    alphas = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]
    summary = run_linear_probe(
        latents_csv=Path(args.latents_csv),
        out_dir=Path(args.out_dir),
        alpha_grid=alphas,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    test = summary["split_metrics"]["test"]
    print(
        f"[RIDGE] test rmse={test['rmse']:.4f} mae={test['mae']:.4f} "
        f"r2={test['r2']:.4f} r={test['pearson_r']:.4f}"
    )
    print(f"[RIDGE] summary={Path(args.out_dir) / 'latent_ridge_summary.json'}")


if __name__ == "__main__":
    main()


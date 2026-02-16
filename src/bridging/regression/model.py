from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)


def _make_val_split(n: int, seed: int = 0, frac: float = 0.2):
    if n <= 1:
        idx = np.arange(n, dtype=int)
        return idx, idx
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * frac)))
    n_val = min(n - 1, n_val)
    val = idx[:n_val]
    train = idx[n_val:]
    return train, val


def _closed_form_fit(X: np.ndarray, y: np.ndarray, *, alpha: float = 0.0):
    if len(X) == 0:
        raise ValueError("No training rows were provided.")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    xtx = X1.T @ X1
    if alpha > 0:
        reg = np.eye(X1.shape[1], dtype=np.float64) * float(alpha)
        reg[-1, -1] = 0.0  # do not penalize bias
        xtx = xtx + reg
    xty = X1.T @ y
    w = np.linalg.pinv(xtx) @ xty
    coef = w[:-1].astype(np.float32)
    bias = float(w[-1])
    return coef, bias


def _closed_form_predict(X: np.ndarray, coef: np.ndarray, bias: float):
    X = np.asarray(X, dtype=np.float32)
    return (X @ coef + np.float32(bias)).astype(np.float32)


@dataclass
class LinearRegressor:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    bias: float
    val_loss: float
    train_count: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros((0,), dtype=np.float32)
        Xs = standardize_apply(X.astype(np.float32), self.mean, self.std)
        return _closed_form_predict(Xs, self.coef, self.bias)


def fit_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    weight_decay: float = 1e-2,
    seed: int = 0,
):
    tr_idx, va_idx = _make_val_split(len(X_train), seed=seed, frac=0.2)
    mean, std = standardize_fit(X_train.astype(np.float32))
    Xs = standardize_apply(X_train.astype(np.float32), mean, std)
    y = y_train.astype(np.float32)

    alpha = max(float(weight_decay), 0.0)
    coef, bias = _closed_form_fit(Xs[tr_idx], y[tr_idx], alpha=alpha)
    pred_val = _closed_form_predict(Xs[va_idx], coef, bias)
    val_loss = float(np.mean((pred_val - y[va_idx]) ** 2))
    return LinearRegressor(
        mean=mean,
        std=std,
        coef=coef,
        bias=bias,
        val_loss=val_loss,
        train_count=int(len(X_train)),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if len(y_true) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "pearson": float("nan")}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    if len(y_true) < 2:
        r2 = float("nan")
        pearson = float("nan")
    else:
        var = float(np.var(y_true))
        r2 = float(1.0 - np.mean(err * err) / var) if var > 0 else float("nan")
        std_true = float(np.std(y_true))
        std_pred = float(np.std(y_pred))
        if std_true <= 0 or std_pred <= 0:
            pearson = float("nan")
        else:
            pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson}

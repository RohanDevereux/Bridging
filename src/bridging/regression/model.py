from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class RidgeLikeLinear(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


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


def train_linear_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    weight_decay: float = 1e-2,
    lr: float = 1e-2,
    steps: int = 2000,
    patience: int = 200,
    seed: int = 0,
    device: str | None = None,
):
    if len(X_train) == 0:
        raise ValueError("No training rows were provided.")

    torch.manual_seed(int(seed))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RidgeLikeLinear(X_train.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_train, dtype=torch.float32, device=dev)
    if len(X_val) == 0:
        X_val = X_train
        y_val = y_train
    Xv = torch.tensor(X_val, dtype=torch.float32, device=dev)
    yv = torch.tensor(y_val, dtype=torch.float32, device=dev)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad = 0

    for _ in range(int(steps)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


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
    model: RidgeLikeLinear | None
    val_loss: float
    device: str
    train_count: int
    method: str
    coef: np.ndarray | None = None
    bias: float | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros((0,), dtype=np.float32)
        Xs = standardize_apply(X.astype(np.float32), self.mean, self.std)
        if self.method in {"ols", "ridge_closed"}:
            if self.coef is None or self.bias is None:
                raise RuntimeError(f"{self.method} regressor missing coefficients.")
            return _closed_form_predict(Xs, self.coef, self.bias)
        if self.model is None:
            raise RuntimeError("Torch regressor missing model state.")
        dev = torch.device(self.device)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
        with torch.no_grad():
            out = self.model(Xt).detach().cpu().numpy().astype(np.float32)
        return out


def fit_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    weight_decay: float = 1e-2,
    lr: float = 1e-2,
    steps: int = 2000,
    patience: int = 200,
    seed: int = 0,
    device: str | None = None,
    method: str = "ridge_closed",
):
    tr_idx, va_idx = _make_val_split(len(X_train), seed=seed, frac=0.2)
    mean, std = standardize_fit(X_train.astype(np.float32))
    Xs = standardize_apply(X_train.astype(np.float32), mean, std)
    y = y_train.astype(np.float32)

    if method == "ols":
        coef, bias = _closed_form_fit(Xs[tr_idx], y[tr_idx], alpha=0.0)
        pred_val = _closed_form_predict(Xs[va_idx], coef, bias)
        val_loss = float(np.mean((pred_val - y[va_idx]) ** 2))
        return LinearRegressor(
            mean=mean,
            std=std,
            model=None,
            val_loss=val_loss,
            device="cpu",
            train_count=int(len(X_train)),
            method=method,
            coef=coef,
            bias=bias,
        )

    if method == "ridge_closed":
        alpha = max(float(weight_decay), 0.0)
        coef, bias = _closed_form_fit(Xs[tr_idx], y[tr_idx], alpha=alpha)
        pred_val = _closed_form_predict(Xs[va_idx], coef, bias)
        val_loss = float(np.mean((pred_val - y[va_idx]) ** 2))
        return LinearRegressor(
            mean=mean,
            std=std,
            model=None,
            val_loss=val_loss,
            device="cpu",
            train_count=int(len(X_train)),
            method=method,
            coef=coef,
            bias=bias,
        )

    if method != "ridge_torch":
        raise ValueError("method must be one of: ols, ridge_closed, ridge_torch")

    model, val_loss = train_linear_torch(
        Xs[tr_idx],
        y[tr_idx],
        Xs[va_idx],
        y[va_idx],
        weight_decay=weight_decay,
        lr=lr,
        steps=steps,
        patience=patience,
        seed=seed,
        device=device,
    )
    dev = str(next(model.parameters()).device)
    return LinearRegressor(
        mean=mean,
        std=std,
        model=model,
        val_loss=float(val_loss),
        device=dev,
        train_count=int(len(X_train)),
        method=method,
        coef=None,
        bias=None,
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

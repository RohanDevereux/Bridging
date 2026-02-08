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


def train_linear(
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


@dataclass
class LinearRegressor:
    mean: np.ndarray
    std: np.ndarray
    model: RidgeLikeLinear
    val_loss: float
    device: str
    train_count: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros((0,), dtype=np.float32)
        Xs = standardize_apply(X.astype(np.float32), self.mean, self.std)
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
):
    tr_idx, va_idx = _make_val_split(len(X_train), seed=seed, frac=0.2)
    mean, std = standardize_fit(X_train.astype(np.float32))
    Xs = standardize_apply(X_train.astype(np.float32), mean, std)
    model, val_loss = train_linear(
        Xs[tr_idx],
        y_train[tr_idx].astype(np.float32),
        Xs[va_idx],
        y_train[va_idx].astype(np.float32),
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

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from .common import (
    build_model_cfg,
    common_transforms,
    mode_summary_and_write,
    regression_metrics,
    seed_all,
)


def _source_rank(entry: dict) -> int:
    src = str(entry.get("pdbcode", "")).split("_", 1)[0].upper().strip()
    if src.startswith("F") and src[1:].isdigit():
        return int(src[1:])
    return 10**9


def _complex_index(entries: list[dict]) -> dict[str, list[int]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for i, e in enumerate(entries):
        key = str(e["PP_ID"]).upper()
        grouped.setdefault(key, []).append((_source_rank(e), i))
    out: dict[str, list[int]] = {}
    for key, pairs in grouped.items():
        pairs = sorted(pairs, key=lambda t: (t[0], t[1]))
        out[key] = [idx for _, idx in pairs]
    return out


def _sample_indices_temporal(rng: np.random.Generator, indices: list[int], k: int) -> list[int]:
    if k <= 0 or len(indices) <= k:
        return list(indices)
    pick_pos = rng.choice(np.arange(len(indices), dtype=np.int64), size=k, replace=False)
    pick_pos = np.sort(pick_pos)
    return [indices[int(p)] for p in pick_pos.tolist()]


class TrajectoryTemporalModel(nn.Module):
    """Joint temporal + frame-independent trajectory head.

    Frame embeddings from DG_Network are consumed in two ways:
    1) biGRU over ordered frames (temporal signal)
    2) mean/max pooling over frames (order-independent signal)
    """

    def __init__(self, frame_model: nn.Module, frame_dim: int, hidden_dim: int):
        super().__init__()
        self.frame_model = frame_model
        self.temporal = nn.GRU(
            input_size=int(frame_dim),
            hidden_size=int(hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        fused_dim = (2 * int(hidden_dim)) + (2 * int(frame_dim))
        self.head = nn.Sequential(
            nn.Linear(fused_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def encode_frames(self, batch: dict) -> torch.Tensor:
        h = self.frame_model.encode(batch)  # (N, L, F)
        return h.max(dim=1)[0]  # (N, F)

    def _pool_stats(self, seq_batch: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, tmax, _ = seq_batch.shape
        idx = torch.arange(tmax, device=seq_batch.device).unsqueeze(0).expand(bsz, tmax)
        mask = idx < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).to(seq_batch.dtype)

        mean_pool = (seq_batch * mask_f).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).to(seq_batch.dtype)
        neg_inf = torch.finfo(seq_batch.dtype).min
        max_in = seq_batch.masked_fill(~mask.unsqueeze(-1), neg_inf)
        max_pool = max_in.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return mean_pool, max_pool

    def forward_sequences(self, seq_batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            seq_batch,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.temporal(packed)  # (2, B, H) for single-layer biGRU
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)

        mean_pool, max_pool = self._pool_stats(seq_batch, lengths)  # (B, F), (B, F)
        fused = torch.cat([h_last, mean_pool, max_pool], dim=1)
        return self.head(fused).squeeze(-1)  # (B,)


def _split_by_counts(x: torch.Tensor, counts: list[int]) -> list[torch.Tensor]:
    out = []
    ptr = 0
    for n in counts:
        out.append(x[ptr : ptr + n])
        ptr += n
    return out


def _next_complex_group(state: dict, batch_size_complex: int) -> list[str]:
    order = state["complex_order"]
    ptr = int(state["complex_ptr"])
    if ptr >= len(order):
        order = list(state["train_complexes"])
        state["py_rng"].shuffle(order)
        state["complex_order"] = order
        ptr = 0
    group = order[ptr : ptr + batch_size_complex]
    state["complex_ptr"] = ptr + batch_size_complex
    return group


def _evaluate_trajectory_model(
    model: TrajectoryTemporalModel,
    dataset,
    complex_to_indices: dict[str, list[int]],
    *,
    frames_per_complex_eval: int,
    collate_fn,
    recursive_to,
    device: str,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for c in sorted(complex_to_indices):
            idxs = complex_to_indices[c]
            if frames_per_complex_eval > 0 and len(idxs) > frames_per_complex_eval:
                pos = np.linspace(0, len(idxs) - 1, num=frames_per_complex_eval)
                pick = np.rint(pos).astype(np.int64)
                pick = np.clip(pick, 0, len(idxs) - 1)
                use = [idxs[int(i)] for i in pick.tolist()]
            else:
                use = list(idxs)

            items = [dataset[i] for i in use]
            batch = collate_fn(items)
            batch = recursive_to(batch, device)
            frame_emb = model.encode_frames(batch)  # (T, F)
            seq = frame_emb.unsqueeze(0)  # (1, T, F)
            lengths = torch.tensor([int(frame_emb.size(0))], device=frame_emb.device, dtype=torch.long)
            y_pred = model.forward_sequences(seq, lengths)[0]
            y_true = batch["dG"].reshape(-1)[0]
            rows.append(
                {
                    "complex": c,
                    "dG_true": float(y_true.item()),
                    "dG_pred": float(y_pred.item()),
                }
            )
    return pd.DataFrame(rows)


def run_trajectory_mode(
    *,
    mode_name: str,
    csv_path: Path,
    out_dir: Path,
    args,
    modules,
) -> dict:
    MixedDataset = modules["MixedDataset"]
    DG_Network = modules["DG_Network"]
    PaddingCollate_struc = modules["PaddingCollate_struc"]
    get_transform = modules["get_transform"]
    recursive_to = modules["recursive_to"]
    EasyDict = modules["EasyDict"]

    cache_dir = out_dir / "cache" / csv_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_tf_cfg, val_tf_cfg = common_transforms(args)
    model_cfg = build_model_cfg(
        EasyDict,
        node_feat_dim=args.node_feat_dim,
        pair_feat_dim=args.pair_feat_dim,
        num_layers=args.num_layers,
        max_num_atoms=args.max_num_atoms,
    )
    collate_fn = PaddingCollate_struc()

    all_pred_rows: list[pd.DataFrame] = []
    all_fold_metrics: list[dict] = []
    total_folds = int(args.num_cvfolds)
    if getattr(args, "only_fold", None) is None:
        fold_indices = list(range(total_folds))
    else:
        fold_indices = [int(args.only_fold)]
    num_active_folds = len(fold_indices)

    fold_states = []
    best_states = []
    best_val_rmse = []

    for fold in fold_indices:
        seed_all(int(args.seed) + fold)
        fold_seed = int(args.seed) + fold
        reset_cache = bool(args.reset_cache and fold == 0)

        train_ds = MixedDataset(
            csv_path=str(csv_path),
            cache_dir=str(cache_dir),
            cvfold_index=fold,
            num_cvfolds=int(args.num_cvfolds),
            split="train",
            split_seed=int(args.seed),
            blocklist=frozenset(),
            transform=get_transform(train_tf_cfg),
            strict=True,
            reset=reset_cache,
        )
        val_ds = MixedDataset(
            csv_path=str(csv_path),
            cache_dir=str(cache_dir),
            cvfold_index=fold,
            num_cvfolds=int(args.num_cvfolds),
            split="val",
            split_seed=int(args.seed),
            blocklist=frozenset(),
            transform=get_transform(val_tf_cfg),
            strict=True,
            reset=False,
        )
        if bool(args.eval_train_on_val_transform):
            train_eval_ds = MixedDataset(
                csv_path=str(csv_path),
                cache_dir=str(cache_dir),
                cvfold_index=fold,
                num_cvfolds=int(args.num_cvfolds),
                split="train",
                split_seed=int(args.seed),
                blocklist=frozenset(),
                transform=get_transform(val_tf_cfg),
                strict=True,
                reset=False,
            )
        else:
            train_eval_ds = train_ds

        train_complex_to_indices = _complex_index(train_ds.entries)
        val_complex_to_indices = _complex_index(val_ds.entries)
        train_eval_complex_to_indices = _complex_index(train_eval_ds.entries)
        train_complexes = sorted(train_complex_to_indices.keys())

        frame_model = DG_Network(model_cfg).to(args.device)
        model = TrajectoryTemporalModel(
            frame_model=frame_model,
            frame_dim=int(args.node_feat_dim),
            hidden_dim=int(args.temporal_hidden_dim),
        ).to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            betas=(0.9, 0.999),
            weight_decay=float(args.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(args.scheduler_factor),
            patience=int(args.scheduler_patience),
            min_lr=float(args.scheduler_min_lr),
        )

        print(
            f"[PPB][{mode_name}] fold={fold} train_complex={len(train_complex_to_indices)} "
            f"test_complex={len(val_complex_to_indices)} train_rows={len(train_ds)} test_rows={len(val_ds)}"
        )

        fold_states.append(
            {
                "fold": fold,
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "train_ds": train_ds,
                "val_ds": val_ds,
                "train_eval_ds": train_eval_ds,
                "train_complex_to_indices": train_complex_to_indices,
                "val_complex_to_indices": val_complex_to_indices,
                "train_eval_complex_to_indices": train_eval_complex_to_indices,
                "train_complexes": train_complexes,
                "complex_order": [],
                "complex_ptr": 0,
                "np_rng": np.random.default_rng(fold_seed),
                "py_rng": random.Random(fold_seed),
            }
        )
        best_states.append(copy.deepcopy(model.state_dict()))
        best_val_rmse.append(float("inf"))

    max_iters = int(args.max_iters)
    val_freq = max(1, int(args.val_freq))
    log_freq = max(1, int(args.train_log_freq))
    batch_size_complex = max(1, int(args.traj_batch_complexes))

    for it in range(1, max_iters + 1):
        active_idx = it % num_active_folds
        state = fold_states[active_idx]
        model = state["model"]
        optimizer = state["optimizer"]

        model.train()
        group = _next_complex_group(state, batch_size_complex)
        if not group:
            continue

        flat_idx = []
        counts = []
        for c in group:
            idxs = state["train_complex_to_indices"][c]
            pick = _sample_indices_temporal(state["np_rng"], idxs, int(args.traj_frames_train))
            flat_idx.extend(pick)
            counts.append(len(pick))
        if not flat_idx:
            continue

        items = [state["train_ds"][i] for i in flat_idx]
        batch = collate_fn(items)
        batch = recursive_to(batch, args.device)

        optimizer.zero_grad(set_to_none=True)
        frame_emb = model.encode_frames(batch)  # (N_frames, F)
        seq_list = _split_by_counts(frame_emb, counts)
        seq_padded = pad_sequence(seq_list, batch_first=True)  # (B, T, F)
        lengths = torch.tensor(counts, device=seq_padded.device, dtype=torch.long)
        pred_t = model.forward_sequences(seq_padded, lengths)  # (B,)

        dG_all = batch["dG"].reshape(-1)
        true_vals = []
        ptr = 0
        for n in counts:
            true_vals.append(dG_all[ptr])
            ptr += n
        true_t = torch.stack(true_vals, dim=0)
        loss = F.mse_loss(pred_t, true_t, reduction="mean")
        loss.backward()
        clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()

        if it % log_freq == 0:
            lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"[PPB][{mode_name}] iter={it:06d} fold={state['fold']} "
                f"train_loss={float(loss.item()):.4f} lr={lr:.6g}"
            )

        if it % val_freq == 0:
            for eval_idx, eval_state in enumerate(fold_states):
                val_rows = _evaluate_trajectory_model(
                    eval_state["model"],
                    eval_state["val_ds"],
                    eval_state["val_complex_to_indices"],
                    frames_per_complex_eval=int(args.traj_frames_eval),
                    collate_fn=collate_fn,
                    recursive_to=recursive_to,
                    device=args.device,
                )
                if val_rows.empty:
                    val_metrics = {
                        "rmse": float("inf"),
                        "pearson_r": float("nan"),
                        "r2": float("nan"),
                    }
                    val_mse = float("inf")
                else:
                    val_metrics = regression_metrics(
                        val_rows["dG_true"].to_numpy(),
                        val_rows["dG_pred"].to_numpy(),
                    )
                    val_err = val_rows["dG_pred"].to_numpy() - val_rows["dG_true"].to_numpy()
                    val_mse = float(np.mean(val_err * val_err))

                eval_state["scheduler"].step(val_mse)
                if val_metrics["rmse"] < best_val_rmse[eval_idx]:
                    best_val_rmse[eval_idx] = val_metrics["rmse"]
                    best_states[eval_idx] = copy.deepcopy(eval_state["model"].state_dict())

                print(
                    f"[PPB][{mode_name}] iter={it:06d} fold={eval_idx} "
                    f"val_rmse={val_metrics['rmse']:.4f} val_r={val_metrics['pearson_r']:.4f} "
                    f"val_r2={val_metrics['r2']:.4f} val_mse={val_mse:.6f}"
                )

    for idx, state in enumerate(fold_states):
        state["model"].load_state_dict(best_states[idx])

    for state in fold_states:
        fold = int(state["fold"])
        train_rows = _evaluate_trajectory_model(
            state["model"],
            state["train_eval_ds"],
            state["train_eval_complex_to_indices"],
            frames_per_complex_eval=int(args.traj_frames_eval),
            collate_fn=collate_fn,
            recursive_to=recursive_to,
            device=args.device,
        )
        train_rows["fold"] = fold
        train_rows["split"] = "train"

        test_rows = _evaluate_trajectory_model(
            state["model"],
            state["val_ds"],
            state["val_complex_to_indices"],
            frames_per_complex_eval=int(args.traj_frames_eval),
            collate_fn=collate_fn,
            recursive_to=recursive_to,
            device=args.device,
        )
        test_rows["fold"] = fold
        test_rows["split"] = "test"

        fold_pred = pd.concat([train_rows, test_rows], ignore_index=True)
        all_pred_rows.append(fold_pred)

        for split_name in ("train", "test"):
            split_df = fold_pred[fold_pred["split"] == split_name].copy()
            metrics = regression_metrics(split_df["dG_true"].to_numpy(), split_df["dG_pred"].to_numpy())
            metrics.update({"mode": mode_name, "fold": fold, "split": split_name})
            all_fold_metrics.append(metrics)
            print(
                f"[PPB][{mode_name}] fold={fold} split={split_name} "
                f"n={metrics['n']} rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
                f"r={metrics['pearson_r']:.4f} r2={metrics['r2']:.4f} me={metrics['mean_error']:.4f}"
            )

    pred_rows_df = pd.concat(all_pred_rows, ignore_index=True)
    fold_metrics_df = pd.DataFrame(all_fold_metrics).sort_values(["split", "fold"]).reset_index(drop=True)
    return mode_summary_and_write(
        mode_name=mode_name,
        out_dir=out_dir,
        pred_rows_df=pred_rows_df,
        fold_metrics_df=fold_metrics_df,
        args=args,
        csv_path=csv_path,
    )

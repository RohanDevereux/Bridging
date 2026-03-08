from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .common import (
    aggregate_complex,
    build_model_cfg,
    common_transforms,
    evaluate_loader,
    mode_summary_and_write,
    regression_metrics,
    seed_all,
)


def _next_batch(loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def run_frame_mode(
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

        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=int(args.num_workers),
        )
        train_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=int(args.num_workers),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=int(args.num_workers),
        )

        model = DG_Network(model_cfg).to(args.device)
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

        train_complex_n = len({str(e["PP_ID"]).upper() for e in train_ds.entries})
        test_complex_n = len({str(e["PP_ID"]).upper() for e in val_ds.entries})
        print(
            f"[PPB][{mode_name}] fold={fold} "
            f"train_rows={len(train_ds)} test_rows={len(val_ds)} "
            f"train_complex={train_complex_n} test_complex={test_complex_n}"
        )

        fold_states.append(
            {
                "fold": fold,
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "train_loader": train_loader,
                "train_iter": iter(train_loader),
                "train_eval_loader": train_eval_loader,
                "val_loader": val_loader,
            }
        )
        best_states.append(copy.deepcopy(model.state_dict()))
        best_val_rmse.append(float("inf"))

    max_iters = int(args.max_iters)
    val_freq = max(1, int(args.val_freq))
    log_freq = max(1, int(args.train_log_freq))

    for it in range(1, max_iters + 1):
        active_idx = (it - 1) % num_active_folds
        state = fold_states[active_idx]
        model = state["model"]
        optimizer = state["optimizer"]

        model.train()
        batch, state["train_iter"] = _next_batch(state["train_loader"], state["train_iter"])
        batch = recursive_to(batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss_dict, _ = model(batch)
        loss = loss_dict["regression"]
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
                val_rows = evaluate_loader(eval_state["model"], eval_state["val_loader"], args.device)
                val_complex = aggregate_complex(val_rows)
                val_metrics = regression_metrics(
                    val_complex["dG_true"].to_numpy(),
                    val_complex["dG_pred"].to_numpy(),
                )
                if val_rows.empty:
                    val_mse = float("inf")
                else:
                    val_err = val_rows["dG_pred"].to_numpy() - val_rows["dG_true"].to_numpy()
                    val_mse = float(np.mean(val_err * val_err))
                eval_state["scheduler"].step(val_mse)
                if val_metrics["rmse"] < best_val_rmse[eval_idx]:
                    best_val_rmse[eval_idx] = val_metrics["rmse"]
                    best_states[eval_idx] = copy.deepcopy(eval_state["model"].state_dict())

                print(
                    f"[PPB][{mode_name}] iter={it:06d} fold={eval_state['fold']} "
                    f"val_rmse={val_metrics['rmse']:.4f} val_r={val_metrics['pearson_r']:.4f} "
                    f"val_r2={val_metrics['r2']:.4f} val_mse={val_mse:.6f}"
                )

    for idx, state in enumerate(fold_states):
        state["model"].load_state_dict(best_states[idx])

    for state in fold_states:
        fold = int(state["fold"])
        pred_train = evaluate_loader(state["model"], state["train_eval_loader"], args.device)
        pred_train["fold"] = fold
        pred_train["split"] = "train"
        pred_test = evaluate_loader(state["model"], state["val_loader"], args.device)
        pred_test["fold"] = fold
        pred_test["split"] = "test"
        fold_pred = pd.concat([pred_train, pred_test], ignore_index=True)
        all_pred_rows.append(fold_pred)

        for split_name in ("train", "test"):
            split_df = aggregate_complex(fold_pred[fold_pred["split"] == split_name])
            metrics = regression_metrics(
                split_df["dG_true"].to_numpy(),
                split_df["dG_pred"].to_numpy(),
            )
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

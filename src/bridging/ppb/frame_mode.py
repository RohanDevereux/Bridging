from __future__ import annotations

import copy
from pathlib import Path

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

    for fold in range(int(args.num_cvfolds)):
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

        best_state = copy.deepcopy(model.state_dict())
        best_val_rmse = float("inf")

        train_complex_n = len({str(e["PP_ID"]).upper() for e in train_ds.entries})
        test_complex_n = len({str(e["PP_ID"]).upper() for e in val_ds.entries})
        print(
            f"[PPB][{mode_name}] fold={fold} "
            f"train_rows={len(train_ds)} test_rows={len(val_ds)} "
            f"train_complex={train_complex_n} test_complex={test_complex_n}"
        )

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            running = 0.0
            steps = 0
            for batch in train_loader:
                batch = recursive_to(batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss_dict, _ = model(batch)
                loss = loss_dict["regression"]
                loss.backward()
                clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                running += float(loss.item())
                steps += 1

            val_rows = evaluate_loader(model, val_loader, args.device)
            val_complex = aggregate_complex(val_rows)
            val_metrics = regression_metrics(
                val_complex["dG_true"].to_numpy(),
                val_complex["dG_pred"].to_numpy(),
            )
            train_loss = running / max(1, steps)
            print(
                f"[PPB][{mode_name}] fold={fold} epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} val_rmse={val_metrics['rmse']:.4f} "
                f"val_r={val_metrics['pearson_r']:.4f} val_r2={val_metrics['r2']:.4f}"
            )
            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)

        pred_train = evaluate_loader(model, train_eval_loader, args.device)
        pred_train["fold"] = fold
        pred_train["split"] = "train"
        pred_test = evaluate_loader(model, val_loader, args.device)
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

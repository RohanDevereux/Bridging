from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bridging.graphvae.prep.record_views import load_complex_metadata, normalize_subgroup_label, subgroup_map_from_metadata


HEADLINE_TAGS = (
    "full_base_S",
    "full_base_SD",
    "full_S_semi_shared_static_z32",
    "iface_base_S",
    "iface_S_semi_shared_static_z8",
)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "n_rows": 0.0,
            "n_complexes": 0.0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r": float("nan"),
            "r2": float("nan"),
        }
    y = df["dG"].to_numpy(dtype=np.float64)
    p = df["dG_pred"].to_numpy(dtype=np.float64)
    err = p - y
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    if len(df) >= 2:
        r = float(np.corrcoef(y, p)[0, 1])
        y_bar = float(np.mean(y))
        sst = float(np.sum((y - y_bar) ** 2))
        r2 = float(1.0 - np.sum(err * err) / sst) if sst > 0 else float("nan")
    else:
        r = float("nan")
        r2 = float("nan")
    return {
        "n_rows": float(len(df)),
        "n_complexes": float(df["complex_id"].nunique()),
        "rmse": rmse,
        "mae": mae,
        "r": r,
        "r2": r2,
    }


def _tag_from_path(parts: tuple[str, ...]) -> str:
    for token in parts:
        if token in {"full_base_S", "full_base_SD", "iface_base_S", "iface_base_SD"}:
            return token
        if token.startswith(("full_", "iface_")) and any(z in token for z in ("_z8", "_z16", "_z32")):
            return token
    raise ValueError(f"Could not determine config tag from path parts: {parts}")


def _repeat_from_path(parts: tuple[str, ...]) -> str:
    return next(token for token in parts if token.startswith("repeat_"))


def _fold_from_path(parts: tuple[str, ...]) -> str:
    return next(token for token in parts if token.startswith("fold_"))


def _family_for_tag(tag: str) -> str:
    if tag in {"full_base_S", "full_base_SD"}:
        return "full baseline"
    if tag in {"iface_base_S", "iface_base_SD"}:
        return "interface baseline"

    view = "full" if tag.startswith("full_") else "interface"
    supervision = "semi" if "_semi_" in tag else "unsup"
    mode = "_SD_" if "_SD_" in tag else "_S_"
    mode_label = "SD" if mode == "_SD_" else "S"
    return f"{view} vae {supervision} {mode_label}"


def _load_metadata_subgroups(dataset_csv: Path) -> dict[str, str]:
    metadata = load_complex_metadata(dataset_csv)
    subgroup_map = subgroup_map_from_metadata(metadata)
    return {cid: normalize_subgroup_label(label) for cid, label in subgroup_map.items()}


def build_combined_predictions(predictions_root: Path, dataset_csv: Path) -> pd.DataFrame:
    subgroup_by_complex = _load_metadata_subgroups(dataset_csv)
    frames: list[pd.DataFrame] = []

    for pred_csv in sorted(predictions_root.rglob("latent_ridge_predictions.csv")):
        parts = pred_csv.parts
        tag = _tag_from_path(parts)
        repeat = _repeat_from_path(parts)
        fold = _fold_from_path(parts)
        df = pd.read_csv(pred_csv)
        if "subgroup" in df.columns:
            df["subgroup"] = df["subgroup"].map(normalize_subgroup_label)
        else:
            df["subgroup"] = df["complex_id"].map(subgroup_by_complex)
        df["subgroup"] = df["subgroup"].fillna(df["complex_id"].map(subgroup_by_complex)).fillna("other")
        df["tag"] = tag
        df["repeat"] = repeat
        df["fold"] = fold
        df["model_family"] = "vae_ridge"
        df["family"] = _family_for_tag(tag)
        frames.append(df)

    for pred_csv in sorted(predictions_root.rglob("supervised_baseline_pred_*.csv")):
        parts = pred_csv.parts
        tag = _tag_from_path(parts)
        repeat = _repeat_from_path(parts)
        fold = _fold_from_path(parts)
        df = pd.read_csv(pred_csv)
        df["subgroup"] = df["complex_id"].map(subgroup_by_complex).fillna("other")
        df["tag"] = tag
        df["repeat"] = repeat
        df["fold"] = fold
        df["model_family"] = "supervised_baseline"
        df["family"] = _family_for_tag(tag)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No prediction CSVs found under {predictions_root}")

    out = pd.concat(frames, ignore_index=True)
    out["subgroup"] = out["subgroup"].map(normalize_subgroup_label)
    return out


def subgroup_metrics_by_tag(test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (tag, subgroup), group in test_df.groupby(["tag", "subgroup"], sort=True):
        pooled = _metrics(group)
        fold_stats = (
            group.groupby(["repeat", "fold"], sort=True)
            .apply(_metrics, include_groups=False)
            .apply(pd.Series)
            .reset_index(drop=True)
        )
        row = {
            "tag": tag,
            "model_family": str(group["model_family"].iloc[0]),
            "family": str(group["family"].iloc[0]),
            "subgroup": subgroup,
            "n_rows_pooled_test": int(pooled["n_rows"]),
            "n_complexes": int(group["complex_id"].nunique()),
            "pooled_test_rmse": pooled["rmse"],
            "pooled_test_mae": pooled["mae"],
            "pooled_test_r": pooled["r"],
            "pooled_test_r2": pooled["r2"],
            "mean_outer_test_rmse": float(fold_stats["rmse"].mean()),
            "std_outer_test_rmse": float(fold_stats["rmse"].std(ddof=1)),
            "mean_outer_test_mae": float(fold_stats["mae"].mean()),
            "std_outer_test_mae": float(fold_stats["mae"].std(ddof=1)),
            "mean_outer_test_r": float(fold_stats["r"].mean()),
            "std_outer_test_r": float(fold_stats["r"].std(ddof=1)),
            "mean_outer_test_r2": float(fold_stats["r2"].mean()),
            "std_outer_test_r2": float(fold_stats["r2"].std(ddof=1)),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["tag", "subgroup"]).reset_index(drop=True)


def subgroup_metrics_by_family(test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (family, subgroup), group in test_df.groupby(["family", "subgroup"], sort=True):
        pooled = _metrics(group)
        row = {
            "family": family,
            "subgroup": subgroup,
            "n_rows_pooled_test": int(pooled["n_rows"]),
            "n_complexes": int(group["complex_id"].nunique()),
            "pooled_test_rmse": pooled["rmse"],
            "pooled_test_mae": pooled["mae"],
            "pooled_test_r": pooled["r"],
            "pooled_test_r2": pooled["r2"],
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["family", "subgroup"]).reset_index(drop=True)


def subgroup_counts(combined_df: pd.DataFrame) -> pd.DataFrame:
    unique_df = combined_df[["complex_id", "subgroup"]].drop_duplicates().copy()
    out = (
        unique_df.groupby("subgroup", sort=True)["complex_id"]
        .nunique()
        .rename("n_complexes")
        .reset_index()
        .sort_values("subgroup")
        .reset_index(drop=True)
    )
    out.loc[len(out)] = {
        "subgroup": "total",
        "n_complexes": int(unique_df["complex_id"].nunique()),
    }
    return out


def best_by_subgroup(tag_df: pd.DataFrame) -> pd.DataFrame:
    best_rows: list[pd.Series] = []
    for subgroup, group in tag_df.groupby("subgroup", sort=True):
        best = group.sort_values(
            ["mean_outer_test_rmse", "pooled_test_rmse", "mean_outer_test_r"],
            ascending=[True, True, False],
        ).iloc[0]
        best_rows.append(best)
    return pd.DataFrame(best_rows).reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reproducible subgroup summaries from final resample predictions.")
    parser.add_argument("--predictions-root", required=True, help="Root directory containing extracted final resample prediction files.")
    parser.add_argument("--dataset", required=True, help="Processed dataset CSV used to derive subgroup labels.")
    parser.add_argument("--out-dir", required=True, help="Directory for combined predictions and subgroup summary CSVs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    predictions_root = Path(args.predictions_root)
    dataset_csv = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_df = build_combined_predictions(predictions_root=predictions_root, dataset_csv=dataset_csv)
    test_df = combined_df[combined_df["split"] == "test"].copy()

    combined_path = out_dir / "final_resample_predictions_combined.csv.gz"
    counts_path = out_dir / "final_resample_subset_subgroup_counts.csv"
    tag_path = out_dir / "final_resample_subgroup_metrics_by_tag.csv"
    family_path = out_dir / "final_resample_subgroup_metrics_by_family.csv"
    headline_path = out_dir / "final_resample_subgroup_metrics_headline_models.csv"
    best_path = out_dir / "final_resample_subgroup_best_models.csv"
    meta_path = out_dir / "final_resample_subgroup_metadata.json"

    combined_df.to_csv(combined_path, index=False, compression="gzip")
    counts_df = subgroup_counts(combined_df)
    counts_df.to_csv(counts_path, index=False)

    tag_df = subgroup_metrics_by_tag(test_df)
    tag_df.to_csv(tag_path, index=False)

    family_df = subgroup_metrics_by_family(test_df)
    family_df.to_csv(family_path, index=False)

    headline_df = tag_df[tag_df["tag"].isin(HEADLINE_TAGS)].copy().reset_index(drop=True)
    headline_df.to_csv(headline_path, index=False)

    best_df = best_by_subgroup(tag_df)
    best_df.to_csv(best_path, index=False)

    metadata = {
        "predictions_root": str(predictions_root),
        "dataset_csv": str(dataset_csv),
        "n_rows_combined": int(len(combined_df)),
        "n_rows_test": int(len(test_df)),
        "n_tags": int(combined_df["tag"].nunique()),
        "n_unique_complexes": int(combined_df["complex_id"].nunique()),
        "headline_tags": list(HEADLINE_TAGS),
        "outputs": {
            "combined_predictions_csv_gz": str(combined_path),
            "subgroup_counts_csv": str(counts_path),
            "subgroup_metrics_by_tag_csv": str(tag_path),
            "subgroup_metrics_by_family_csv": str(family_path),
            "subgroup_metrics_headline_models_csv": str(headline_path),
            "subgroup_best_models_csv": str(best_path),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] combined={combined_path}")
    print(f"[OK] counts={counts_path}")
    print(f"[OK] by_tag={tag_path}")
    print(f"[OK] by_family={family_path}")
    print(f"[OK] headline={headline_path}")
    print(f"[OK] best={best_path}")
    print(f"[OK] meta={meta_path}")


if __name__ == "__main__":
    main()

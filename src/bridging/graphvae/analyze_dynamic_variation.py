from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DYNAMIC_EDGE_FEATURES_WITH_DIST, DYNAMIC_NODE_FEATURES


def _as_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        return {"n": 0}
    q = np.quantile(arr, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(np.max(arr)),
    }


def _feature_column(arr: np.ndarray, names: list[str], feature_name: str) -> np.ndarray:
    idx = {name: i for i, name in enumerate(names)}
    return np.asarray(arr[:, idx[feature_name]], dtype=np.float64)


def analyze_dynamic_variation(*, records_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = torch.load(records_path, map_location="cpu")
    if not records:
        raise ValueError(f"No records in {records_path}")

    node_names = list(records[0]["node_feature_names"])
    edge_names = list(records[0]["edge_feature_names"])
    node_dyn = [n for n in DYNAMIC_NODE_FEATURES if n in node_names]
    edge_dyn = [n for n in DYNAMIC_EDGE_FEATURES_WITH_DIST if n in edge_names]
    if not node_dyn and not edge_dyn:
        raise ValueError("No dynamic features found in records.")

    per_complex_rows: list[dict] = []
    global_store: dict[str, list[np.ndarray]] = {f: [] for f in (node_dyn + edge_dyn)}

    for rec in records:
        cid = str(rec.get("complex_id", "UNKNOWN"))
        split = str(rec.get("split", "unknown"))
        pdb_id = str(rec.get("pdb_id", "UNKNOWN"))
        node_arr = _as_numpy(rec["node_features"])
        edge_arr = _as_numpy(rec["edge_features"])
        row = {
            "complex_id": cid,
            "split": split,
            "pdb_id": pdb_id,
            "n_nodes": int(node_arr.shape[0]),
            "n_edges": int(edge_arr.shape[0]),
        }

        for feat in node_dyn:
            vals = _feature_column(node_arr, node_names, feat)
            global_store[feat].append(vals)
            row[f"{feat}__mean"] = float(np.nanmean(vals))
            row[f"{feat}__std"] = float(np.nanstd(vals))
            row[f"{feat}__p05"] = float(np.nanquantile(vals, 0.05))
            row[f"{feat}__p95"] = float(np.nanquantile(vals, 0.95))

        for feat in edge_dyn:
            vals = _feature_column(edge_arr, edge_names, feat)
            global_store[feat].append(vals)
            row[f"{feat}__mean"] = float(np.nanmean(vals))
            row[f"{feat}__std"] = float(np.nanstd(vals))
            row[f"{feat}__p05"] = float(np.nanquantile(vals, 0.05))
            row[f"{feat}__p95"] = float(np.nanquantile(vals, 0.95))
            if feat == "dyn_contact_freq_8A":
                finite = vals[np.isfinite(vals)]
                if finite.size > 0:
                    row["dyn_contact_freq_8A__frac_near0"] = float(np.mean(finite <= 0.05))
                    row["dyn_contact_freq_8A__frac_mid"] = float(np.mean((finite > 0.05) & (finite < 0.95)))
                    row["dyn_contact_freq_8A__frac_near1"] = float(np.mean(finite >= 0.95))
                else:
                    row["dyn_contact_freq_8A__frac_near0"] = float("nan")
                    row["dyn_contact_freq_8A__frac_mid"] = float("nan")
                    row["dyn_contact_freq_8A__frac_near1"] = float("nan")

        per_complex_rows.append(row)

    per_complex_df = pd.DataFrame(per_complex_rows)
    per_complex_csv = out_dir / "dynamic_variation_per_complex.csv"
    per_complex_df.to_csv(per_complex_csv, index=False)

    global_stats: dict[str, dict] = {}
    for feat, chunks in global_store.items():
        if not chunks:
            continue
        all_vals = np.concatenate([c[np.isfinite(c)] for c in chunks if c.size > 0], axis=0)
        global_stats[feat] = _stats(all_vals)

    contact_global = {}
    if "dyn_contact_freq_8A" in global_store:
        all_contact = np.concatenate(
            [c[np.isfinite(c)] for c in global_store["dyn_contact_freq_8A"] if c.size > 0],
            axis=0,
        )
        if all_contact.size > 0:
            contact_global = {
                "frac_near0": float(np.mean(all_contact <= 0.05)),
                "frac_mid": float(np.mean((all_contact > 0.05) & (all_contact < 0.95))),
                "frac_near1": float(np.mean(all_contact >= 0.95)),
            }

    heuristic_flags: list[str] = []
    if contact_global:
        if contact_global["frac_mid"] < 0.10:
            heuristic_flags.append("contact_freq_is_mostly_binary")
    rmsf_stats = global_stats.get("dyn_rmsf_ca")
    if rmsf_stats and rmsf_stats.get("n", 0) > 0 and rmsf_stats["p50"] < 0.02:
        heuristic_flags.append("rmsf_is_very_low_for_most_nodes")
    water_std_stats = global_stats.get("dyn_water_count_5A_std")
    if water_std_stats and water_std_stats.get("n", 0) > 0 and water_std_stats["p50"] < 0.20:
        heuristic_flags.append("water_count_std_is_low_for_most_nodes")

    split_summary = (
        per_complex_df.groupby("split", dropna=False)
        .agg(
            n_complex=("complex_id", "count"),
            mean_nodes=("n_nodes", "mean"),
            mean_edges=("n_edges", "mean"),
            contact_mid_mean=("dyn_contact_freq_8A__frac_mid", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
        if "dyn_contact_freq_8A__frac_mid" in per_complex_df.columns
        else []
    )

    report = {
        "records_path": str(records_path),
        "n_complexes": int(len(per_complex_df)),
        "node_dynamic_features": node_dyn,
        "edge_dynamic_features": edge_dyn,
        "global_feature_stats": global_stats,
        "global_contact_binary_profile": contact_global,
        "split_summary": split_summary,
        "heuristic_flags": heuristic_flags,
        "heuristic_note": (
            "Heuristics are diagnostics only. They indicate whether 1 ns appears to produce "
            "non-trivial dynamic spread in current features; they are not a proof of predictive value."
        ),
        "per_complex_csv": str(per_complex_csv),
    }
    out_json = out_dir / "dynamic_variation_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize dynamic feature variation from prepared graph records."
    )
    parser.add_argument("--records", required=True, help="Path to graph_records.pt")
    parser.add_argument("--out-dir", required=True, help="Output directory for report and CSV")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = analyze_dynamic_variation(
        records_path=Path(args.records),
        out_dir=Path(args.out_dir),
    )
    print(f"[DYNVAR] complexes={report['n_complexes']}")
    if report["global_contact_binary_profile"]:
        p = report["global_contact_binary_profile"]
        print(
            "[DYNVAR] contact_freq profile "
            f"near0={p['frac_near0']:.3f} mid={p['frac_mid']:.3f} near1={p['frac_near1']:.3f}"
        )
    if report["heuristic_flags"]:
        print(f"[DYNVAR] heuristic_flags={','.join(report['heuristic_flags'])}")
    else:
        print("[DYNVAR] heuristic_flags=none")
    print(f"[DYNVAR] report={Path(args.out_dir) / 'dynamic_variation_report.json'}")
    print(f"[DYNVAR] per_complex={Path(args.out_dir) / 'dynamic_variation_per_complex.csv'}")


if __name__ == "__main__":
    main()


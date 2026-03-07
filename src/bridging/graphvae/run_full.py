from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bridging.MD.paths import PDB_CACHE_DIR

from .prepare import build_prepared_dataset
from .regress import run_linear_probe
from .supervised_baseline import run_supervised_baseline
from .train import train_masked_graph_vae


def run_full_pipeline(
    *,
    dataset: Path,
    md_root: Path,
    out_dir: Path,
    pdb_cache_root: Path,
    graph_source: str,
    build_deeprank: bool,
    deep_rank_hdf5: list[Path] | None,
    deeprank_prefix: Path | None,
    influence_radius: float,
    max_edge_length: float | None,
    train_fraction: float,
    val_fraction: float,
    split_seed: int,
    frames_per_complex: int,
    include_dynamic_dist_stats: bool,
    require_all_protein_nodes: bool,
    overwrite: bool,
    prepare_progress_every: int,
    reuse_prepared: bool,
    device: str,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    mask_ratio: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    beta_start: float,
    beta_end: float,
    beta_anneal_fraction: float,
    corr_weight: float,
    seed: int,
    num_workers: int,
    alpha_grid: list[float],
    bootstrap: int,
    run_supervised: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = out_dir / "prepared"
    records_path = prepared_dir / "graph_records.pt"
    pipeline_t0 = time.perf_counter()
    print(f"[PIPE] start out_dir={out_dir}")

    if reuse_prepared and records_path.exists():
        print(f"[PIPE] prepare reuse records={records_path}")
        prepare_report = json.loads((prepared_dir / "prepare_report.json").read_text(encoding="utf-8"))
    else:
        step_t0 = time.perf_counter()
        print("[PIPE] prepare start")
        prepare_report = build_prepared_dataset(
            dataset_csv=dataset,
            md_root=md_root,
            out_dir=prepared_dir,
            pdb_cache_root=pdb_cache_root,
            graph_source=graph_source,
            build_deeprank=build_deeprank,
            deeprank_hdf5=deep_rank_hdf5,
            deeprank_prefix=deeprank_prefix,
            influence_radius=influence_radius,
            max_edge_length=max_edge_length,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            split_seed=split_seed,
            frames_per_complex=frames_per_complex,
            include_dynamic_dist_stats=include_dynamic_dist_stats,
            require_all_protein_nodes=require_all_protein_nodes,
            overwrite=overwrite,
            progress_every=prepare_progress_every,
        )
        print(f"[PIPE] prepare done elapsed_s={time.perf_counter() - step_t0:.1f}")

    mode_s_dir = out_dir / "mode_S"
    mode_sd_dir = out_dir / "mode_SD"
    step_t0 = time.perf_counter()
    print("[PIPE] train mode=S start")
    train_s = train_masked_graph_vae(
        records_path=records_path,
        out_dir=mode_s_dir,
        mode="S",
        device=device,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mask_ratio=mask_ratio,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_anneal_fraction=beta_anneal_fraction,
        corr_weight=corr_weight,
        seed=seed,
        num_workers=num_workers,
    )
    print(f"[PIPE] train mode=S done elapsed_s={time.perf_counter() - step_t0:.1f}")
    step_t0 = time.perf_counter()
    print("[PIPE] train mode=SD start")
    train_sd = train_masked_graph_vae(
        records_path=records_path,
        out_dir=mode_sd_dir,
        mode="SD",
        device=device,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mask_ratio=mask_ratio,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_anneal_fraction=beta_anneal_fraction,
        corr_weight=corr_weight,
        seed=seed,
        num_workers=num_workers,
    )
    print(f"[PIPE] train mode=SD done elapsed_s={time.perf_counter() - step_t0:.1f}")

    step_t0 = time.perf_counter()
    print("[PIPE] ridge probes start")
    reg_s = run_linear_probe(
        latents_csv=Path(train_s["latents_csv"]),
        out_dir=mode_s_dir,
        alpha_grid=alpha_grid,
        bootstrap=bootstrap,
        seed=seed,
    )
    reg_sd = run_linear_probe(
        latents_csv=Path(train_sd["latents_csv"]),
        out_dir=mode_sd_dir,
        alpha_grid=alpha_grid,
        bootstrap=bootstrap,
        seed=seed,
    )
    print(f"[PIPE] ridge probes done elapsed_s={time.perf_counter() - step_t0:.1f}")
    sup = {}
    if run_supervised:
        step_t0 = time.perf_counter()
        print("[PIPE] supervised baselines start")
        sup_s = run_supervised_baseline(
            records_path=records_path,
            out_dir=mode_s_dir,
            mode="S",
            device=device,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            num_workers=num_workers,
        )
        sup_sd = run_supervised_baseline(
            records_path=records_path,
            out_dir=mode_sd_dir,
            mode="SD",
            device=device,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            num_workers=num_workers,
        )
        sup = {"S": sup_s, "SD": sup_sd}
        print(f"[PIPE] supervised baselines done elapsed_s={time.perf_counter() - step_t0:.1f}")

    test_s = reg_s["split_metrics"]["test"]
    test_sd = reg_sd["split_metrics"]["test"]
    comparison = {
        "prepare_report": prepare_report,
        "mode_S_train": train_s,
        "mode_SD_train": train_sd,
        "mode_S_regression": reg_s,
        "mode_SD_regression": reg_sd,
        "supervised_baselines": sup,
        "primary_outcome": {
            "metric": "test_rmse (lower is better), test_r2 (higher is better)",
            "S_test_rmse": test_s["rmse"],
            "SD_test_rmse": test_sd["rmse"],
            "delta_rmse_SD_minus_S": test_sd["rmse"] - test_s["rmse"],
            "S_test_r2": test_s["r2"],
            "SD_test_r2": test_sd["r2"],
            "delta_r2_SD_minus_S": test_sd["r2"] - test_s["r2"],
        },
    }
    compare_path = out_dir / "compare_S_vs_SD.json"
    compare_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"[PIPE] done total_elapsed={time.perf_counter() - pipeline_t0:.1f}s compare={compare_path}")
    return comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full S vs SD masked Graph-VAE + linear probe pipeline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--md-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pdb-cache-root", default=str(PDB_CACHE_DIR))
    parser.add_argument(
        "--graph-source",
        choices=["raw_pdb", "md_topology_protein"],
        default="md_topology_protein",
        help="Source PDB for DeepRank graph generation. md_topology_protein matches MD topology ID space.",
    )
    parser.add_argument("--build-deeprank", action="store_true")
    parser.add_argument("--deep-rank-hdf5", nargs="*")
    parser.add_argument("--deeprank-prefix")
    parser.add_argument(
        "--influence-radius",
        type=float,
        default=1_000_000.0,
        help="DeepRank influence radius in Angstrom. Large finite default approximates whole-complex coverage.",
    )
    parser.add_argument("--max-edge-length", type=float)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--frames-per-complex", type=int, default=120)
    parser.add_argument("--include-dynamic-dist-stats", action="store_true")
    parser.add_argument("--allow-partial-node-coverage", action="store_true")
    parser.add_argument("--prepare-progress-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--beta-anneal-fraction", type=float, default=0.30)
    parser.add_argument("--corr-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha-grid", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--run-supervised-baselines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    alphas = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]
    out = run_full_pipeline(
        dataset=Path(args.dataset),
        md_root=Path(args.md_root),
        out_dir=Path(args.out_dir),
        pdb_cache_root=Path(args.pdb_cache_root).expanduser(),
        graph_source=str(args.graph_source),
        build_deeprank=bool(args.build_deeprank),
        deep_rank_hdf5=[Path(x) for x in args.deep_rank_hdf5] if args.deep_rank_hdf5 else None,
        deeprank_prefix=Path(args.deeprank_prefix) if args.deeprank_prefix else None,
        influence_radius=float(args.influence_radius),
        max_edge_length=float(args.max_edge_length) if args.max_edge_length is not None else None,
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        frames_per_complex=int(args.frames_per_complex),
        include_dynamic_dist_stats=bool(args.include_dynamic_dist_stats),
        require_all_protein_nodes=not bool(args.allow_partial_node_coverage),
        prepare_progress_every=int(args.prepare_progress_every),
        overwrite=bool(args.overwrite),
        reuse_prepared=bool(args.reuse_prepared),
        device=args.device,
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        mask_ratio=float(args.mask_ratio),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
        beta_anneal_fraction=float(args.beta_anneal_fraction),
        corr_weight=float(args.corr_weight),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        alpha_grid=alphas,
        bootstrap=int(args.bootstrap),
        run_supervised=bool(args.run_supervised_baselines),
    )
    prim = out["primary_outcome"]
    print(
        f"[COMPARE] S_rmse={prim['S_test_rmse']:.4f} SD_rmse={prim['SD_test_rmse']:.4f} "
        f"delta={prim['delta_rmse_SD_minus_S']:.4f}"
    )
    print(f"[COMPARE] summary={Path(args.out_dir) / 'compare_S_vs_SD.json'}")


if __name__ == "__main__":
    main()

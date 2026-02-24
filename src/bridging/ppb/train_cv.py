from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import recursive_to
from .ensemble_mode import run_ensemble_mode
from .frame_mode import run_frame_mode
from .trajectory_mode import run_trajectory_mode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate PPB-style models with 5-fold complex CV. "
            "Runs baseline (static PDB), frame-augmented (independent MD frames), "
            "trajectory temporal model, and optional 3-model ensemble."
        )
    )
    parser.add_argument("--baseline-csv", help="Prepared PPB baseline CSV")
    parser.add_argument("--frame-aug-csv", help="Prepared PPB frame-augmented CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory for metrics/predictions")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--num-cvfolds", type=int, default=5)
    parser.add_argument(
        "--only-fold",
        type=int,
        default=None,
        help="Run only one fold index (0-based) while keeping split definition from --num-cvfolds.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Deprecated; use --max-iters/--val-freq.")
    parser.add_argument("--max-iters", type=int, default=100_000)
    parser.add_argument("--val-freq", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-factor", type=float, default=0.8)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--train-log-freq", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=100.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--node-feat-dim", type=int, default=128)
    parser.add_argument("--pair-feat-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-num-atoms", type=int, default=15)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--atom-noise-std", type=float, default=0.02)
    parser.add_argument("--chi-noise-std", type=float, default=0.02)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument(
        "--eval-train-on-val-transform",
        action="store_true",
        help=(
            "Evaluate train split using val transform (no noise). "
            "Default evaluates train split with train transform."
        ),
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-frame-aug", action="store_true")
    parser.add_argument("--skip-trajectory", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument(
        "--temporal-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for trajectory temporal GRU head.",
    )
    parser.add_argument(
        "--traj-frames-train",
        type=int,
        default=24,
        help="Frames sampled per complex per train step in trajectory model.",
    )
    parser.add_argument(
        "--traj-frames-eval",
        type=int,
        default=120,
        help="Frames used per complex at evaluation in trajectory model.",
    )
    parser.add_argument(
        "--traj-batch-complexes",
        type=int,
        default=4,
        help="Number of complexes per optimization step in trajectory model.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.baseline_csv and not args.frame_aug_csv:
        raise ValueError("Provide at least one of --baseline-csv or --frame-aug-csv")
    if args.only_fold is not None and not (0 <= int(args.only_fold) < int(args.num_cvfolds)):
        raise ValueError(f"--only-fold must be in [0, {int(args.num_cvfolds)-1}]")

    print("[PPB] using local vendored BaselineModel package")

    from easydict import EasyDict
    from BaselineModel.datasets.mixed_dataset import MixedDataset
    from BaselineModel.models.dg_model import DG_Network
    from BaselineModel.utils.data import PaddingCollate_struc
    from BaselineModel.utils.transforms import get_transform

    modules = {
        "EasyDict": EasyDict,
        "MixedDataset": MixedDataset,
        "DG_Network": DG_Network,
        "PaddingCollate_struc": PaddingCollate_struc,
        "recursive_to": recursive_to,
        "get_transform": get_transform,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    produced = {}

    if args.baseline_csv and not bool(args.skip_baseline):
        summary = run_frame_mode(
            mode_name="baseline",
            csv_path=Path(args.baseline_csv),
            out_dir=out_dir,
            args=args,
            modules=modules,
        )
        summaries.append(summary)
        produced["baseline"] = summary

    if args.frame_aug_csv and not bool(args.skip_frame_aug):
        summary = run_frame_mode(
            mode_name="frame_aug",
            csv_path=Path(args.frame_aug_csv),
            out_dir=out_dir,
            args=args,
            modules=modules,
        )
        summaries.append(summary)
        produced["frame_aug"] = summary

    if args.frame_aug_csv and not bool(args.skip_trajectory):
        summary = run_trajectory_mode(
            mode_name="trajectory",
            csv_path=Path(args.frame_aug_csv),
            out_dir=out_dir,
            args=args,
            modules=modules,
        )
        summaries.append(summary)
        produced["trajectory"] = summary

    if not bool(args.skip_ensemble):
        need = {"baseline", "frame_aug", "trajectory"}
        if need.issubset(set(produced.keys())):
            ens_summary = run_ensemble_mode(
                out_dir=out_dir,
                baseline_pred_complex_csv=Path(produced["baseline"]["predictions_complex_csv"]),
                frame_aug_pred_complex_csv=Path(produced["frame_aug"]["predictions_complex_csv"]),
                trajectory_pred_complex_csv=Path(produced["trajectory"]["predictions_complex_csv"]),
            )
            summaries.append(ens_summary)
        else:
            missing = sorted(need.difference(set(produced.keys())))
            print(f"[PPB][ensemble] skipped; missing modes: {missing}")

    combined_path = out_dir / "compare_summary.json"
    combined_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[PPB] compare_summary={combined_path}")


if __name__ == "__main__":
    main()

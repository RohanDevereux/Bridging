from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..prep.record_views import (
    DEFAULT_INTERFACE_POLICY,
    SUPPORTED_INTERFACE_POLICIES,
    resolve_graph_view_variants,
)


def _parse_name_list(raw: str, *, allowed: set[str], label: str) -> list[str]:
    out: list[str] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"Unsupported {label}: {token}")
        out.append(token)
    if not out:
        raise ValueError(f"Expected at least one {label}.")
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize graph view datasets ahead of GraphVAE sweeps.")
    parser.add_argument("--records", required=True, help="Prepared graph_records.pt path.")
    parser.add_argument("--dataset", required=True, help="Dataset CSV with complex metadata.")
    parser.add_argument("--out-dir", required=True, help="Output directory that will contain per-policy view datasets.")
    parser.add_argument("--graph-views", default="full,interface")
    parser.add_argument("--interface-policies", default=DEFAULT_INTERFACE_POLICY)
    parser.add_argument(
        "--match-interface-subset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also materialize a full-view dataset filtered to the interface-retained complex IDs.",
    )
    parser.add_argument("--pdb-cache-root")
    parser.add_argument("--md-root")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    graph_views = _parse_name_list(args.graph_views, allowed={"full", "interface"}, label="graph view")
    if "interface" not in graph_views:
        raise ValueError("materialize_views requires graph_views to include 'interface'.")
    interface_policies = _parse_name_list(
        args.interface_policies,
        allowed=set(SUPPORTED_INTERFACE_POLICIES),
        label="interface policy",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    view_variants, view_reports = resolve_graph_view_variants(
        records_path=Path(args.records),
        dataset_csv=Path(args.dataset),
        graph_views=graph_views,
        interface_policies=interface_policies,
        view_root=out_dir,
        pdb_cache_root=(Path(args.pdb_cache_root) if args.pdb_cache_root else None),
        md_root=(Path(args.md_root) if args.md_root else None),
        match_interface_subset=bool(args.match_interface_subset),
        reuse_existing=False,
        progress_every=int(args.progress_every),
        log_prefix="[VIEWPREP]",
    )

    summary = {
        "records_path": str(Path(args.records)),
        "dataset_csv": str(Path(args.dataset)),
        "out_dir": str(out_dir),
        "graph_views": graph_views,
        "interface_policies": interface_policies,
        "match_interface_subset": bool(args.match_interface_subset),
        "view_variants": [
            {
                "interface_policy": variant["interface_policy"],
                "paths": {name: str(path) for name, path in variant["paths"].items()},
            }
            for variant in view_variants
        ],
        "view_reports": view_reports,
    }
    summary_path = out_dir / "view_materialization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[VIEWPREP] summary={summary_path}")


if __name__ == "__main__":
    main()

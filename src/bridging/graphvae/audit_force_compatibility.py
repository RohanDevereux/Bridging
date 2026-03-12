from __future__ import annotations

import argparse
import contextlib
import json
import signal
from collections import Counter
from pathlib import Path

from bridging.MD.paths import PDB_CACHE_DIR
from bridging.MD.prefetch_pdbs import ensure_pdb_cached

from .force_features import assess_force_query_compatibility
from .prepare import _select_complex_entries


class RowAuditTimeout(RuntimeError):
    pass


@contextlib.contextmanager
def _row_timeout(seconds: float):
    if seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):
        raise RowAuditTimeout(f"row audit timed out after {seconds:.1f}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit whether PPB query chain groups are representable for graph force features "
            "against raw/full/protein topology files, without running OpenMM force evaluation."
        )
    )
    p.add_argument("--dataset", required=True, help="Dataset CSV used for graph/force preparation.")
    p.add_argument("--md-root", required=True, help="Root with per-PDB MD outputs.")
    p.add_argument("--pdb-cache-root", default=str(PDB_CACHE_DIR), help="Cached raw PDB directory.")
    p.add_argument("--out-json", help="Optional JSON report path.")
    p.add_argument("--limit", type=int, default=0, help="Optional limit on selected complexes for debugging.")
    p.add_argument("--progress-every", type=int, default=50, help="Progress print interval.")
    p.add_argument(
        "--per-row-timeout-sec",
        type=float,
        default=0.0,
        help="Optional per-row timeout in seconds for pathological topology rows.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    dataset = Path(args.dataset)
    md_root = Path(args.md_root).expanduser()
    pdb_cache_root = Path(args.pdb_cache_root).expanduser()
    entries, select_report = _select_complex_entries(dataset)
    if args.limit and int(args.limit) > 0:
        entries = entries[: int(args.limit)]

    totals = {
        "dataset": str(dataset),
        "md_root": str(md_root),
        "pdb_cache_root": str(pdb_cache_root),
        "select_report": select_report,
        "n_selected": int(len(entries)),
        "n_done_with_topology": 0,
        "n_missing_md_dir": 0,
        "n_missing_topology_protein": 0,
        "n_missing_topology_full": 0,
        "n_compatible": 0,
        "n_incompatible": 0,
        "n_timed_out": 0,
        "compatibility_reason_counts": {},
        "incompatible_examples": [],
        "timed_out_examples": [],
    }
    reason_counts: Counter[str] = Counter()

    total = int(len(entries))
    for i, rec in enumerate(entries, start=1):
        pdb_id = str(rec["pdb_id"]).strip()
        md_dir = md_root / pdb_id
        protein_top = md_dir / "topology_protein.pdb"
        full_top = md_dir / "topology_full.pdb"

        if not md_dir.exists():
            totals["n_missing_md_dir"] += 1
            continue
        if not protein_top.exists():
            totals["n_missing_topology_protein"] += 1
            continue
        if not full_top.exists():
            totals["n_missing_topology_full"] += 1

        try:
            with _row_timeout(float(args.per_row_timeout_sec)):
                raw_pdb_path, _ = ensure_pdb_cached(pdb_id, cache_dir=pdb_cache_root)
                compat = assess_force_query_compatibility(
                    raw_pdb_path=raw_pdb_path,
                    protein_topology_pdb=protein_top,
                    full_topology_pdb=full_top if full_top.exists() else None,
                    ligand_group=str(rec["chains_1"]),
                    receptor_group=str(rec["chains_2"]),
                )
        except RowAuditTimeout:
            totals["n_done_with_topology"] += 1
            totals["n_incompatible"] += 1
            totals["n_timed_out"] += 1
            reason_counts["timeout"] += 1
            if len(totals["timed_out_examples"]) < 100:
                totals["timed_out_examples"].append(
                    {
                        "complex_id": str(rec["complex_id"]),
                        "pdb_id": pdb_id,
                        "reason": "timeout",
                        "timeout_sec": float(args.per_row_timeout_sec),
                    }
                )
            if args.progress_every > 0 and (i % int(args.progress_every) == 0 or i == total):
                print(
                    f"[AUDIT] {i}/{total} compatible={totals['n_compatible']} "
                    f"incompatible={totals['n_incompatible']} missing_md={totals['n_missing_md_dir']} "
                    f"missing_topology_protein={totals['n_missing_topology_protein']}",
                    flush=True,
                )
            continue

        totals["n_done_with_topology"] += 1

        if bool(compat.get("compatible", False)):
            totals["n_compatible"] += 1
        else:
            totals["n_incompatible"] += 1
            reason = str(compat.get("compatibility_reason", "unknown"))
            reason_counts[reason] += 1
            if len(totals["incompatible_examples"]) < 100:
                totals["incompatible_examples"].append(
                    {
                        "complex_id": str(rec["complex_id"]),
                        "pdb_id": pdb_id,
                        "reason": reason,
                        "missing_in_raw": list(compat.get("missing_in_raw", [])),
                        "missing_in_full": list(compat.get("missing_in_full", [])),
                        "missing_in_protein": list(compat.get("missing_in_protein", [])),
                        "raw_query_overlap": list(compat.get("raw_query_overlap", [])),
                        "md_ligand_group": list(compat.get("md_ligand_group", [])),
                        "md_receptor_group": list(compat.get("md_receptor_group", [])),
                    }
                )

        if args.progress_every > 0 and (i % int(args.progress_every) == 0 or i == total):
            print(
                f"[AUDIT] {i}/{total} compatible={totals['n_compatible']} "
                f"incompatible={totals['n_incompatible']} missing_md={totals['n_missing_md_dir']} "
                f"missing_topology_protein={totals['n_missing_topology_protein']}",
                flush=True,
            )

    totals["compatibility_reason_counts"] = dict(reason_counts)

    print(json.dumps(
        {
            "n_selected": totals["n_selected"],
            "n_done_with_topology": totals["n_done_with_topology"],
            "n_compatible": totals["n_compatible"],
            "n_incompatible": totals["n_incompatible"],
            "n_missing_md_dir": totals["n_missing_md_dir"],
            "n_missing_topology_protein": totals["n_missing_topology_protein"],
            "n_missing_topology_full": totals["n_missing_topology_full"],
            "compatibility_reason_counts": totals["compatibility_reason_counts"],
            "n_timed_out": totals["n_timed_out"],
        },
        indent=2,
    ))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(totals, indent=2), encoding="utf-8")
        print(f"[AUDIT] report={out}", flush=True)


if __name__ == "__main__":
    main()

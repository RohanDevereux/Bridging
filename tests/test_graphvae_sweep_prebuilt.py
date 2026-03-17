from pathlib import Path

import pandas as pd
import torch

from bridging.graphvae.prep import md_dynamics
from bridging.graphvae.prep.record_views import resolve_graph_view_variants


def test_load_protein_md_trajectory_uses_single_frame_loader(monkeypatch, tmp_path: Path) -> None:
    md_dir = tmp_path / "TEST"
    md_dir.mkdir(parents=True, exist_ok=True)
    (md_dir / "traj_protein.nc").write_bytes(b"nc")
    (md_dir / "topology_protein.pdb").write_text("END\n", encoding="utf-8")

    called: dict[str, object] = {}
    sentinel = object()

    def fake_load_frame(path: str, index: int, top: str):
        called["path"] = path
        called["index"] = index
        called["top"] = top
        return sentinel

    def fail_load(*args, **kwargs):
        raise AssertionError("md.load should not be called when max_frames=1")

    monkeypatch.setattr(md_dynamics.md, "load_frame", fake_load_frame)
    monkeypatch.setattr(md_dynamics.md, "load", fail_load)

    out = md_dynamics.load_protein_md_trajectory(md_dir, max_frames=1)
    assert out is sentinel
    assert called["index"] == 0
    assert str(called["path"]).endswith("traj_protein.nc")
    assert str(called["top"]).endswith("topology_protein.pdb")


def test_resolve_graph_view_variants_reuses_prebuilt_views(monkeypatch, tmp_path: Path) -> None:
    records_path = tmp_path / "graph_records.pt"
    torch.save(
        [
            {
                "complex_id": "TEST__A__B",
                "pdb_id": "TEST",
                "split": "train",
                "dG": -1.0,
                "node_feature_names": ["f1"],
                "edge_feature_names": ["distance"],
                "node_features": torch.tensor([[1.0]], dtype=torch.float32),
                "edge_features": torch.zeros((1, 1), dtype=torch.float32),
                "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                "node_chain_id": ["A"],
                "node_position": [1],
            }
        ],
        records_path,
    )

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"PDB": "TEST", "Chains_1": "A", "Chains_2": "B", "Subgroup": "OTHER"}]
    ).to_csv(dataset_csv, index=False)

    prebuilt_root = tmp_path / "views"
    policy_dir = prebuilt_root / "md_closest_pair_patch"
    policy_dir.mkdir(parents=True, exist_ok=True)

    interface_path = policy_dir / "graph_records_interface.pt"
    torch.save(torch.load(records_path, map_location="cpu"), interface_path)
    (policy_dir / "graph_view_interface_report.json").write_text(
        '{"graph_view":"interface","interface_policy":"md_closest_pair_patch","records_in":1,"records_considered":1,"records_out":1,"n_skipped_by_subset":0,"missing_metadata":[],"n_missing_metadata":0,"n_failed_view":0,"status_counts":{"ok":1},"retained_complex_ids":["TEST__A__B"],"n_report_rows_total":1,"n_report_rows_sampled":1,"report_rows":[{"complex_id":"TEST__A__B","status":"ok"}],"elapsed_seconds":0.0}',
        encoding="utf-8",
    )

    def fail_materialize(*args, **kwargs):
        raise AssertionError("materialize_graph_view_records should not run when prebuilt views are provided")

    monkeypatch.setattr("bridging.graphvae.prep.record_views.materialize_graph_view_records", fail_materialize)

    view_variants, view_reports = resolve_graph_view_variants(
        records_path=records_path,
        dataset_csv=dataset_csv,
        graph_views=["interface"],
        interface_policies=["md_closest_pair_patch"],
        view_root=prebuilt_root,
        pdb_cache_root=None,
        md_root=None,
        match_interface_subset=True,
        reuse_existing=True,
    )

    assert view_variants[0]["paths"]["interface"] == interface_path
    assert view_reports["md_closest_pair_patch"]["interface"]["records_out"] == 1

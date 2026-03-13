from pathlib import Path

import pandas as pd
import torch

from bridging.graphvae.record_views import materialize_graph_view_records


def _write_raw_pdb(path: Path) -> None:
    lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  CA  GLY A   2      20.000   0.000   0.000  1.00 20.00           C",
        "ATOM      3  CA  SER B   1       8.000   0.000   0.000  1.00 20.00           C",
        "ATOM      4  CA  THR B   2      35.000   0.000   0.000  1.00 20.00           C",
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_record() -> dict:
    return {
        "complex_id": "TEST__A__B",
        "pdb_id": "TEST",
        "split": "train",
        "dG": -10.0,
        "node_feature_names": ["f1"],
        "edge_feature_names": ["distance"],
        "node_features": torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32),
        "edge_features": torch.zeros((2, 1), dtype=torch.float32),
        "edge_index": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        "node_chain_id": ["A", "A", "B", "B"],
        "node_position": [1, 2, 1, 2],
    }


def test_interface_view_uses_ppb10_patch_without_cross_partner_edges(tmp_path: Path) -> None:
    records_path = tmp_path / "graph_records.pt"
    torch.save([_make_record()], records_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"PDB": "TEST", "Chains_1": "A", "Chains_2": "B", "Subgroup": "OTHER"}]
    ).to_csv(dataset_csv, index=False)

    pdb_cache_root = tmp_path / "pdb_cache"
    pdb_cache_root.mkdir()
    _write_raw_pdb(pdb_cache_root / "TEST.pdb")

    out_path, report = materialize_graph_view_records(
        records_path=records_path,
        dataset_csv=dataset_csv,
        graph_view="interface",
        out_dir=tmp_path / "views",
        pdb_cache_root=pdb_cache_root,
        interface_policy="ppb10_patch",
    )

    transformed = torch.load(out_path, map_location="cpu")
    assert len(transformed) == 1
    assert report["records_in"] == 1
    assert report["records_out"] == 1

    row = report["report_rows"][0]
    assert row["status"] == "ok"
    assert row["interface_source"] == "ppb10_patch"
    assert row["n_inter_partner_edges_full"] == 0

    rec = transformed[0]
    assert rec["graph_view"] == "interface"
    assert rec["n_nodes_view"] >= 2
    kept = set(zip(rec["node_chain_id"], rec["node_position"]))
    assert ("A", 1) in kept
    assert ("B", 1) in kept


def test_interface_view_drops_complex_when_no_ppb_interface_residues(tmp_path: Path) -> None:
    records_path = tmp_path / "graph_records.pt"
    torch.save([_make_record()], records_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"PDB": "TEST", "Chains_1": "A", "Chains_2": "B", "Subgroup": "OTHER"}]
    ).to_csv(dataset_csv, index=False)

    pdb_cache_root = tmp_path / "pdb_cache"
    pdb_cache_root.mkdir()
    far_lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  CA  GLY A   2      20.000   0.000   0.000  1.00 20.00           C",
        "ATOM      3  CA  SER B   1      50.000   0.000   0.000  1.00 20.00           C",
        "ATOM      4  CA  THR B   2      70.000   0.000   0.000  1.00 20.00           C",
        "TER",
        "END",
    ]
    (pdb_cache_root / "TEST.pdb").write_text("\n".join(far_lines) + "\n", encoding="utf-8")

    out_path, report = materialize_graph_view_records(
        records_path=records_path,
        dataset_csv=dataset_csv,
        graph_view="interface",
        out_dir=tmp_path / "views",
        pdb_cache_root=pdb_cache_root,
        interface_policy="ppb10_patch",
    )

    transformed = torch.load(out_path, map_location="cpu")
    assert transformed == []
    assert report["records_in"] == 1
    assert report["records_out"] == 0
    assert report["n_failed_view"] == 1
    assert report["report_rows"][0]["status"] == "no_ppb_interface_residues"


def test_closest_pair_patch_keeps_complex_without_ppb10_interface(tmp_path: Path) -> None:
    records_path = tmp_path / "graph_records.pt"
    torch.save([_make_record()], records_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"PDB": "TEST", "Chains_1": "A", "Chains_2": "B", "Subgroup": "OTHER"}]
    ).to_csv(dataset_csv, index=False)

    pdb_cache_root = tmp_path / "pdb_cache"
    pdb_cache_root.mkdir()
    far_lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  CA  GLY A   2      20.000   0.000   0.000  1.00 20.00           C",
        "ATOM      3  CA  SER B   1      50.000   0.000   0.000  1.00 20.00           C",
        "ATOM      4  CA  THR B   2      70.000   0.000   0.000  1.00 20.00           C",
        "TER",
        "END",
    ]
    (pdb_cache_root / "TEST.pdb").write_text("\n".join(far_lines) + "\n", encoding="utf-8")

    out_path, report = materialize_graph_view_records(
        records_path=records_path,
        dataset_csv=dataset_csv,
        graph_view="interface",
        out_dir=tmp_path / "views",
        pdb_cache_root=pdb_cache_root,
        interface_policy="closest_pair_patch",
    )

    transformed = torch.load(out_path, map_location="cpu")
    assert len(transformed) == 1
    assert report["records_out"] == 1
    row = report["report_rows"][0]
    assert row["status"] == "ok"
    assert row["interface_source"] == "closest_pair_patch"
    assert row["closest_pair_ca_distance_angstrom"] > 10.0

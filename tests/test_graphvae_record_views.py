from pathlib import Path

import pandas as pd
import pytest
import torch

from bridging.graphvae.prep.record_views import materialize_graph_view_records


def _write_pdb(path: Path, *, chain_1: str = "A", chain_2: str = "B", close: bool = True) -> None:
    b1 = 8.0 if close else 50.0
    b2 = 35.0 if close else 70.0
    lines = [
        f"ATOM      1  CA  ALA {chain_1}   1       0.000   0.000   0.000  1.00 20.00           C",
        f"ATOM      2  CA  GLY {chain_1}   2      20.000   0.000   0.000  1.00 20.00           C",
        f"ATOM      3  CA  SER {chain_2}   1       {b1:0.3f}   0.000   0.000  1.00 20.00           C",
        f"ATOM      4  CA  THR {chain_2}   2      {b2:0.3f}   0.000   0.000  1.00 20.00           C",
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_record(*, chains: tuple[str, str, str, str] = ("A", "A", "B", "B")) -> dict:
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
        "node_chain_id": list(chains),
        "node_position": [1, 2, 1, 2],
    }


def _write_md_first_frame(md_root: Path, *, chain_1: str = "X", chain_2: str = "Y", close: bool = True) -> None:
    md = pytest.importorskip("mdtraj")
    md_dir = md_root / "TEST"
    md_dir.mkdir(parents=True, exist_ok=True)
    top_path = md_dir / "topology_protein.pdb"
    _write_pdb(top_path, chain_1=chain_1, chain_2=chain_2, close=close)
    traj = md.load(str(top_path))
    traj.save_netcdf(str(md_dir / "traj_protein.nc"))


def test_interface_view_uses_ppb10_patch_without_cross_partner_edges(tmp_path: Path) -> None:
    records_path = tmp_path / "graph_records.pt"
    torch.save([_make_record()], records_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"PDB": "TEST", "Chains_1": "A", "Chains_2": "B", "Subgroup": "OTHER"}]
    ).to_csv(dataset_csv, index=False)

    pdb_cache_root = tmp_path / "pdb_cache"
    pdb_cache_root.mkdir()
    _write_pdb(pdb_cache_root / "TEST.pdb")

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
    assert report["status_counts"]["ok"] == 1

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
    _write_pdb(pdb_cache_root / "TEST.pdb", close=False)

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
    _write_pdb(pdb_cache_root / "TEST.pdb", close=False)

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


def test_closest_pair_patch_remaps_raw_partner_chains_to_md_topology(tmp_path: Path) -> None:
    records_path = tmp_path / "graph_records.pt"
    torch.save([_make_record(chains=("X", "X", "Y", "Y"))], records_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"PDB": "TEST", "Chains_1": "A", "Chains_2": "B", "Subgroup": "OTHER"}]
    ).to_csv(dataset_csv, index=False)

    pdb_cache_root = tmp_path / "pdb_cache"
    pdb_cache_root.mkdir()
    _write_pdb(pdb_cache_root / "TEST.pdb", chain_1="A", chain_2="B", close=True)

    md_root = tmp_path / "md_root"
    md_dir = md_root / "TEST"
    md_dir.mkdir(parents=True, exist_ok=True)
    _write_pdb(md_dir / "topology_protein.pdb", chain_1="X", chain_2="Y", close=True)

    out_path, report = materialize_graph_view_records(
        records_path=records_path,
        dataset_csv=dataset_csv,
        graph_view="interface",
        out_dir=tmp_path / "views",
        pdb_cache_root=pdb_cache_root,
        md_root=md_root,
        interface_policy="closest_pair_patch",
    )

    transformed = torch.load(out_path, map_location="cpu")
    assert len(transformed) == 1
    assert report["records_out"] == 1
    row = report["report_rows"][0]
    assert row["status"] == "ok"
    assert row["partner_chain_source"] == "raw_to_md_chain_map"
    assert row["graph_to_raw_chain_source"] == "md_to_raw_chain_map"


def test_md_closest_pair_patch_uses_md_first_frame_mapping(tmp_path: Path) -> None:
    records_path = tmp_path / "graph_records.pt"
    torch.save([_make_record(chains=("X", "X", "Y", "Y"))], records_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"PDB": "TEST", "Chains_1": "A", "Chains_2": "B", "Subgroup": "OTHER"}]
    ).to_csv(dataset_csv, index=False)

    pdb_cache_root = tmp_path / "pdb_cache"
    pdb_cache_root.mkdir()
    _write_pdb(pdb_cache_root / "TEST.pdb", chain_1="A", chain_2="B", close=False)

    md_root = tmp_path / "md_root"
    _write_md_first_frame(md_root, chain_1="X", chain_2="Y", close=True)

    out_path, report = materialize_graph_view_records(
        records_path=records_path,
        dataset_csv=dataset_csv,
        graph_view="interface",
        out_dir=tmp_path / "views",
        pdb_cache_root=pdb_cache_root,
        md_root=md_root,
        interface_policy="md_closest_pair_patch",
    )

    transformed = torch.load(out_path, map_location="cpu")
    assert len(transformed) == 1
    assert report["records_out"] == 1
    row = report["report_rows"][0]
    assert row["status"] == "ok"
    assert row["interface_source"] == "md_closest_pair_patch"
    assert row["partner_chain_source"] == "raw_to_md_chain_map"

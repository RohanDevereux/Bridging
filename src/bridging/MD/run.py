import json
import traceback

import pandas as pd

from .paths import DATA_CSV, MD_OUT_DIR, PDB_CACHE_DIR, resolve_dataset
from .prepare_complex import (
    load_and_fix,
    select_chains,
    drop_non_protein_residues,
    mark_disulfides,
    cysteine_variants,
    solvate,
)
from .simulate import run_simulation


def _chain_ids(chains_1, chains_2):
    return list(dict.fromkeys(list(chains_1) + list(chains_2)))


def run_all(dataset_path=None):
    dataset_path = resolve_dataset(dataset_path, DATA_CSV)
    df = pd.read_csv(dataset_path)
    MD_OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        row = row._asdict()
        pdb_id = str(row["PDB"]).upper()
        chains_1 = str(row["Chains_1"])
        chains_2 = str(row["Chains_2"])
        chain_ids = _chain_ids(chains_1, chains_2)
        temp_k = float(row["Temp_K"])
        ph = float(row["pH"]) if "pH" in row and row["pH"] == row["pH"] else 7.0

        out_dir = MD_OUT_DIR / pdb_id
        done_file = out_dir / "DONE"
        if done_file.exists():
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {idx}/{total} {pdb_id} ({dataset_path.name})")

        meta = {
            "pdb_id": pdb_id,
            "chains_1": chains_1,
            "chains_2": chains_2,
            "temp_k": temp_k,
            "pH": ph,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        try:
            pdb_file = PDB_CACHE_DIR / f"{pdb_id}.pdb"
            fixer = load_and_fix(pdb_file)
            modeller = select_chains(fixer.topology, fixer.positions, chain_ids)
            modeller = drop_non_protein_residues(modeller)
            modeller = mark_disulfides(modeller)
            variants = cysteine_variants(modeller.topology)
            forcefield, modeller = solvate(modeller, ph, variants=variants)
            run_simulation(forcefield, modeller, out_dir, temp_k)
            done_file.write_text("ok\n", encoding="utf-8")
            print(f"[OK] {pdb_id}")
        except Exception as exc:
            (out_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[FAIL] {pdb_id}: {exc}")


if __name__ == "__main__":
    run_all()

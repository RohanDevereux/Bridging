import json
import re
import traceback

import pandas as pd

from .paths import PPB_DATA_CSV, PPB_MD_OUT_DIR, PDB_CACHE_DIR, resolve_dataset
from .prepare_complex import (
    load_and_fix,
    select_chains,
    drop_non_protein_residues,
    solvate,
)
from .simulate import run_simulation


def _parse_chain_group(value):
    if pd.isna(value):
        return []
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(value))
    return list(dict.fromkeys(list(cleaned)))


def run_all(dataset_path=None):
    dataset_path = resolve_dataset(dataset_path, PPB_DATA_CSV)
    df = pd.read_csv(dataset_path)
    PPB_MD_OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        row = row._asdict()
        pdb_id = str(row["PDB"]).upper()
        chains_1 = str(row["Chains_1"])
        chains_2 = str(row["Chains_2"])
        chain_ids = _parse_chain_group(chains_1) + _parse_chain_group(chains_2)
        chain_ids = list(dict.fromkeys(chain_ids))
        temp_k = int(row["Temp_K"])
        ph = 7.0

        out_dir = PPB_MD_OUT_DIR / pdb_id
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
            forcefield, modeller = solvate(modeller, ph, pdb_path=pdb_file)
            run_simulation(forcefield, modeller, out_dir, temp_k)
            done_file.write_text("ok\n", encoding="utf-8")
            print(f"[OK] {pdb_id}")
        except Exception as exc:
            (out_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[FAIL] {pdb_id}: {exc}")


if __name__ == "__main__":
    run_all()

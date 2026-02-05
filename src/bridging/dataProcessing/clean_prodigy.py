'''
from pathlib import Path

from pandas import read_csv

package_dir = Path(__file__).resolve().parent.parent
data = read_csv(package_dir / "rawData" / "elife-07454-supp4.csv")
data["Baseline"] = data["ICs/NIS-based"]
data["Experimental"] = data["Binding_affinity"]
data["Error"] = data["Baseline"] - data["Experimental"]
cleaned_data = data[["PDB", "Baseline", "Experimental", "Error"]]
cleaned_data.to_csv(package_dir / "processedData" / "PRODIGY_Baseline.csv", index=False)
'''
from __future__ import annotations

from pathlib import Path
import pandas as pd

from bridging.dataProcessing.parse_table_s1 import load_table_s1


def main() -> None:
    package_dir = Path(__file__).resolve().parent.parent

    supp4_path = package_dir / "rawData" / "elife-07454-supp4.csv"
    table_s1_path = package_dir / "rawData" / "Supplementary_Table_S1.csv"

    supp4 = pd.read_csv(supp4_path)

    # eLife supp4 PDB values look like "1WEJ.PDB"
    supp4["PDB_ID"] = (
        supp4["PDB"].astype(str).str.upper().str.strip().str.replace(".PDB", "", regex=False)
    )

    # Your baseline/experimental choices
    supp4["Baseline_dG"] = supp4["ICs/NIS-based"]
    supp4["Experimental_dG"] = supp4["Binding_affinity"]
    supp4["Error_dG"] = supp4["Baseline_dG"] - supp4["Experimental_dG"]

    # Load/normalize Table S1
    s1 = load_table_s1(table_s1_path)

    # For the 79-complex eLife subset, PDB IDs are unique and match 1:1 with Table S1.
    s1_sub = s1[s1["PDB_ID"].isin(set(supp4["PDB_ID"]))].copy()

    merged = supp4.merge(
        s1_sub[
            [
                "PDB_ID",
                "complex_pdb",
                "class",
                "Chains_1_raw",
                "Chains_2_raw",
                "Temp_C",
                "Temp_K",
                "pH",
                "method",
                "Kd_M",
                "Kd_qualifier",
                "dG_kcal_mol",
                "dG_qualifier",
            ]
        ],
        on="PDB_ID",
        how="left",
        validate="one_to_one",
    )

    out_cols = [
        "PDB_ID",
        "complex_pdb",
        "Chains_1_raw",
        "Chains_2_raw",
        "Baseline_dG",
        "Experimental_dG",
        "Error_dG",
        "Temp_C",
        "Temp_K",
        "pH",
        "Kd_M",
        "Kd_qualifier",
        "dG_kcal_mol",
        "dG_qualifier",
        "method",
        "class",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    out = merged[out_cols].rename(
        columns={
            "PDB_ID": "PDB",
            "Chains_1_raw": "Chains_1",
            "Chains_2_raw": "Chains_2",
            "method": "Method",
            "class": "Class",
        }
    )

    # Normalize numeric fields and fill defaults for missing values.
    TEMP_DEFAULT_K = 298.15
    PH_DEFAULT = 7.0
    out["Temp_K"] = pd.to_numeric(out["Temp_K"], errors="coerce").fillna(TEMP_DEFAULT_K)
    out["pH"] = pd.to_numeric(out["pH"], errors="coerce").fillna(PH_DEFAULT)

    out_path = package_dir / "processedData" / "PRODIGY_Data.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

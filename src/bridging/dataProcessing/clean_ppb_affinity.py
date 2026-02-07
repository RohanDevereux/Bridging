from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def _parse_temp_k(value):
    if pd.isna(value):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return int(float(match.group(0)))


def main() -> None:
    package_dir = Path(__file__).resolve().parent.parent
    raw_path = package_dir / "rawData" / "PPB-Affinity.csv"
    out_path = package_dir / "processedData" / "PPB_Affinity_TCR_pMHC.csv"

    df = pd.read_csv(raw_path)

    df["Temp_K"] = df["Temperature(K)"].apply(_parse_temp_k)
    df = df[df["Temp_K"].notna()].copy()

    df["Subgroup"] = df["Subgroup"].astype(str).str.strip()
    df = df[df["Subgroup"] == "TCR-pMHC"].copy()

    df = df.head(100).copy()

    out = df[
        [
            "PDB",
            "Ligand Chains",
            "Receptor Chains",
            "Temp_K",
            "KD(M)",
            "Affinity Method",
            "Structure Method",
            "Resolution(A)",
            "Source Data Set",
            "Complex ID",
            "Subgroup",
        ]
    ].rename(
        columns={
            "Ligand Chains": "Chains_1",
            "Receptor Chains": "Chains_2",
        }
    )

    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


_AMBIENT_SENTINELS = {"ambient", "not stated", "not_stated", "na", ""}


def _parse_complex_pdb(complex_pdb: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse the benchmark notation like:
      1A2K_C:AB  -> PDB=1A2K, chains_1=C, chains_2=AB
      1WEJ_HL:F  -> PDB=1WEJ, chains_1=HL, chains_2=F

    Notation note: AB:C means component1 has chains A and B; component2 has chain C. :contentReference[oaicite:2]{index=2}
    """
    s = str(complex_pdb).strip()
    m = re.match(r"^([0-9A-Za-z]{4})_([^:]+):(.+)$", s)
    if not m:
        return None, None, None
    pdb = m.group(1).upper()
    left = m.group(2).strip()
    right = m.group(3).strip()
    return pdb, left, right


def _canon_chains(chains: str | None) -> str | None:
    """
    Canonicalize chain group by sorting alphanumeric characters.
    PDB chain IDs are typically single characters; groups like 'HL' represent {'H','L'}.
    """
    if chains is None or (isinstance(chains, float) and np.isnan(chains)):
        return None
    chars = [c for c in str(chains).strip() if c.isalnum()]
    return "".join(sorted(chars)) if chars else None


def _parse_temp_c(x) -> tuple[float, str | None]:
    """
    Table S1 uses numeric temperatures, and sometimes 'ambient' / 'not stated'.
    Benchmark note: ambient/not stated is treated as 25°C. :contentReference[oaicite:3]{index=3}
    """
    if pd.isna(x):
        return np.nan, None
    s = str(x).strip()
    s_l = s.lower()
    if s_l in _AMBIENT_SENTINELS:
        return 25.0, s
    try:
        return float(s), s
    except ValueError:
        return np.nan, s


def _parse_ph(x) -> tuple[float, str | None]:
    if pd.isna(x):
        return np.nan, None
    s = str(x).strip()
    s_l = s.lower()
    if s_l in {"not stated", "na", ""}:
        return np.nan, s
    try:
        return float(s), s
    except ValueError:
        return np.nan, s


def _parse_inequality_number(x) -> tuple[float, str | None, str | None]:
    """
    Handles values like '<1.4E-11' or '>15'.
    Returns: (numeric_value, qualifier '<'|'>'|None, raw_string)
    """
    if pd.isna(x):
        return np.nan, None, None
    raw = str(x).strip()
    qual = None
    s = raw
    if s.startswith("<") or s.startswith(">"):
        qual = s[0]
        s = s[1:].strip()
    try:
        val = float(s.replace("E", "e"))
    except ValueError:
        val = np.nan
    return val, qual, raw


def load_table_s1(csv_path: Path) -> pd.DataFrame:
    """
    Load and normalize Supplementary_Table_S1.csv into analysis-friendly columns.
    """
    df = pd.read_csv(csv_path)

    # Parse complex_pdb into PDB + chain groups
    parsed = df["complex_pdb"].apply(_parse_complex_pdb)
    df[["PDB_ID", "Chains_1_raw", "Chains_2_raw"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df["Chains_1"] = df["Chains_1_raw"].apply(_canon_chains)
    df["Chains_2"] = df["Chains_2_raw"].apply(_canon_chains)

    # Normalize temperature/pH
    temp_parsed = df["temp_c"].apply(_parse_temp_c)
    df[["Temp_C", "Temp_C_raw"]] = pd.DataFrame(temp_parsed.tolist(), index=df.index)
    df["Temp_K"] = df["Temp_C"] + 273.15

    ph_parsed = df["ph"].apply(_parse_ph)
    df[["pH", "pH_raw"]] = pd.DataFrame(ph_parsed.tolist(), index=df.index)

    # Normalize Kd and ΔG entries (handle inequalities)
    kd_parsed = df["kd_m"].apply(_parse_inequality_number)
    df[["Kd_M", "Kd_qualifier", "Kd_raw"]] = pd.DataFrame(kd_parsed.tolist(), index=df.index)

    dg_parsed = df["dg_kcal_mol"].apply(_parse_inequality_number)
    df[["dG_kcal_mol", "dG_qualifier", "dG_raw"]] = pd.DataFrame(dg_parsed.tolist(), index=df.index)

    return df


def main() -> None:
    package_dir = Path(__file__).resolve().parent.parent
    raw_path = package_dir / "rawData" / "Supplementary_Table_S1.csv"
    out_path = package_dir / "processedData" / "Kastritis2011_TableS1.csv"

    df = load_table_s1(raw_path)

    keep_cols = [
        "complex_pdb",
        "PDB_ID",
        "class",
        "Chains_1_raw",
        "Chains_2_raw",
        "Chains_1",
        "Chains_2",
        "Kd_M",
        "Kd_qualifier",
        "dG_kcal_mol",
        "dG_qualifier",
        "Temp_C",
        "Temp_K",
        "pH",
        "method",
        "reference",
        "dasa_a2",
        "i_rmsd_a",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

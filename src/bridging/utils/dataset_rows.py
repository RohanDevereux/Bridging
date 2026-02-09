from __future__ import annotations

import re

import pandas as pd

from .table import normalized_lookup


def parse_chain_group(value) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    tokens = re.findall(r"[A-Za-z0-9]", str(value))
    return list(dict.fromkeys(tokens))


def parse_complex_pdb(value: str) -> tuple[str | None, str | None, str | None]:
    s = str(value).strip()
    m = re.match(r"^([A-Za-z0-9]{4})_([^:]+):(.+)$", s)
    if not m:
        return None, None, None
    return m.group(1).upper(), m.group(2), m.group(3)


def row_chain_groups(row: dict) -> tuple[str | None, str | None]:
    lookup = normalized_lookup(row)
    if "chains1" in lookup and "chains2" in lookup:
        return str(row[lookup["chains1"]]), str(row[lookup["chains2"]])
    if "ligandchains" in lookup and "receptorchains" in lookup:
        return str(row[lookup["ligandchains"]]), str(row[lookup["receptorchains"]])
    if "complexpdb" in lookup:
        _, left, right = parse_complex_pdb(str(row[lookup["complexpdb"]]))
        return left, right
    return None, None


def row_pdb_id(row: dict) -> str | None:
    lookup = normalized_lookup(row)
    if "pdb" in lookup:
        value = str(row[lookup["pdb"]]).strip().upper()
        if re.fullmatch(r"[A-Z0-9]{4}", value):
            return value
    if "pdbid" in lookup:
        value = str(row[lookup["pdbid"]]).strip().upper()
        if re.fullmatch(r"[A-Z0-9]{4}", value):
            return value
    if "complexpdb" in lookup:
        pdb_id, _, _ = parse_complex_pdb(str(row[lookup["complexpdb"]]))
        if pdb_id is not None:
            return pdb_id
        text = str(row[lookup["complexpdb"]]).strip()
        if "_" in text:
            text = text.split("_", 1)[0]
        text = text.upper()
        if re.fullmatch(r"[A-Z0-9]{4}", text):
            return text
    if "complexid" in lookup:
        m = re.match(r"^([A-Za-z0-9]{4})", str(row[lookup["complexid"]]).strip())
        if m:
            return m.group(1).upper()
    return None

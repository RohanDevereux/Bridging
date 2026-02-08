from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ProdigyRequest:
    row_index: int
    pdb_id: str
    ligand_group: str
    receptor_group: str
    temperature_k: float | None

    @property
    def cache_key(self) -> str:
        return make_cache_key(
            self.pdb_id,
            self.ligand_group,
            self.receptor_group,
            self.temperature_k,
        )


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _norm_lookup(row: dict) -> dict[str, str]:
    return {_norm_col(k): k for k in row.keys()}


def _first_value(row: dict, lookup: dict[str, str], aliases: list[str]):
    for alias in aliases:
        key = lookup.get(alias)
        if key is None:
            continue
        value = row.get(key)
        if pd.isna(value):
            continue
        s = str(value).strip()
        if s:
            return s
    return None


def _parse_complex_pdb(value: str) -> tuple[str | None, str | None, str | None]:
    s = str(value).strip()
    m = re.match(r"^([A-Za-z0-9]{4})_([^:]+):(.+)$", s)
    if not m:
        return None, None, None
    return m.group(1).upper(), m.group(2), m.group(3)


def _parse_chain_ids(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    tokens = re.findall(r"[A-Za-z0-9]", str(value))
    unique = list(dict.fromkeys(tokens))
    return unique


def _format_group(chain_ids: list[str]) -> str:
    if not chain_ids:
        return ""
    return ",".join(chain_ids)


def _canonical_group(group: str) -> str:
    chain_ids = [c.upper() for c in _parse_chain_ids(group)]
    return ",".join(sorted(chain_ids))


def make_cache_key(
    pdb_id: str,
    ligand_group: str,
    receptor_group: str,
    temperature_k: float | None,
) -> str:
    if temperature_k is None or pd.isna(temperature_k):
        temp_part = "NA"
    else:
        temp_part = f"{float(temperature_k):.2f}"

    return "|".join(
        [
            str(pdb_id).strip().upper()[:4],
            _canonical_group(ligand_group),
            _canonical_group(receptor_group),
            temp_part,
        ]
    )


def _extract_temperature_k(row: dict, lookup: dict[str, str]) -> float | None:
    temp_k = _first_value(
        row,
        lookup,
        [
            "tempk",
            "temperaturek",
            "temperaturekelvin",
        ],
    )
    if temp_k is not None:
        try:
            return float(temp_k)
        except Exception:
            return None

    temp_c = _first_value(
        row,
        lookup,
        [
            "tempc",
            "temperaturec",
            "temperaturecelsius",
        ],
    )
    if temp_c is not None:
        try:
            return float(temp_c) + 273.15
        except Exception:
            return None

    return None


def parse_request_row(row: dict, row_index: int) -> ProdigyRequest:
    lookup = _norm_lookup(row)

    pdb_id = _first_value(row, lookup, ["pdb", "pdbid"])
    ligand_raw = _first_value(row, lookup, ["ligandchains", "chains1"])
    receptor_raw = _first_value(row, lookup, ["receptorchains", "chains2"])

    if not pdb_id:
        complex_pdb_value = _first_value(row, lookup, ["complexpdb"])
        if complex_pdb_value:
            parsed_pdb, parsed_lig, parsed_rec = _parse_complex_pdb(complex_pdb_value)
            pdb_id = parsed_pdb
            ligand_raw = ligand_raw or parsed_lig
            receptor_raw = receptor_raw or parsed_rec

    if (not ligand_raw or not receptor_raw) and lookup.get("complexpdb"):
        parsed_pdb, parsed_lig, parsed_rec = _parse_complex_pdb(str(row[lookup["complexpdb"]]))
        pdb_id = pdb_id or parsed_pdb
        ligand_raw = ligand_raw or parsed_lig
        receptor_raw = receptor_raw or parsed_rec

    if not pdb_id:
        complex_id = _first_value(row, lookup, ["complexid"])
        if complex_id:
            m = re.match(r"^([A-Za-z0-9]{4})", complex_id.strip())
            if m:
                pdb_id = m.group(1).upper()

    if not pdb_id:
        raise ValueError("missing PDB identifier")
    if not ligand_raw or not receptor_raw:
        raise ValueError("missing ligand/receptor chain groups")

    ligand_group = _format_group(_parse_chain_ids(ligand_raw))
    receptor_group = _format_group(_parse_chain_ids(receptor_raw))
    if not ligand_group or not receptor_group:
        raise ValueError("empty ligand/receptor chain groups after parsing")

    return ProdigyRequest(
        row_index=row_index,
        pdb_id=pdb_id.upper()[:4],
        ligand_group=ligand_group,
        receptor_group=receptor_group,
        temperature_k=_extract_temperature_k(row, lookup),
    )

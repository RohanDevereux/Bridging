from __future__ import annotations

import re

from bridging.utils.dataset_rows import parse_chain_group, row_chain_groups, row_pdb_id


def chain_group_token(value: str | None) -> str:
    chains = parse_chain_group(value)
    if not chains:
        return "NA"
    return "".join(chains).upper()


def canonical_complex_id(row: dict) -> str | None:
    pdb_id = row_pdb_id(row)
    if not pdb_id:
        return None
    left, right = row_chain_groups(row)
    left_token = chain_group_token(left)
    right_token = chain_group_token(right)
    return f"{pdb_id.upper()}__{left_token}__{right_token}"


def sanitize_filename_token(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]", "_", str(text))
    return re.sub(r"_+", "_", out).strip("_")


def primary_chain(value: str | None) -> str | None:
    chains = parse_chain_group(value)
    if not chains:
        return None
    return chains[0].upper()


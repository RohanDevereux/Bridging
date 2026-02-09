from __future__ import annotations

import re

import pandas as pd


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def normalized_lookup(row: dict) -> dict[str, str]:
    return {normalize_column_name(k): k for k in row.keys()}


def first_nonempty(
    row: dict,
    lookup: dict[str, str],
    aliases: list[str],
    *,
    as_text: bool = False,
):
    for alias in aliases:
        key = lookup.get(alias)
        if key is None:
            continue
        value = row.get(key)
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text:
            return text if as_text else value
    return None

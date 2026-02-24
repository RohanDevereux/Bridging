from __future__ import annotations

import math

from .dataset_rows import row_temperature_k
from .table import first_nonempty, normalized_lookup

R_KCAL_PER_MOL_K = 0.00198720425864083
DEFAULT_TEMP_K = 298.15


def split_name(value) -> str:
    s = str(value).strip().lower()
    if s in {"test", "val", "valid", "validation"}:
        return "test"
    return "train"


def to_float(value):
    if value is None:
        return None
    try:
        x = float(value)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def experimental_delta_g_kcalmol(row: dict) -> float | None:
    lookup = normalized_lookup(row)
    value = first_nonempty(
        row,
        lookup,
        [
            "deltagkcal",
            "experimentaldg",
            "dgkcalmol",
            "bindingaffinity",
        ],
    )
    out = to_float(value)
    if out is not None:
        return out

    kd_raw = first_nonempty(row, lookup, ["kdm", "kd"])
    kd = to_float(kd_raw)
    temp_k = row_temperature_k(row)
    if kd is None or kd <= 0:
        return None
    if temp_k is None:
        temp_k = DEFAULT_TEMP_K
    return R_KCAL_PER_MOL_K * float(temp_k) * math.log(kd)

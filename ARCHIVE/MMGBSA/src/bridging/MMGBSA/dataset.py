from __future__ import annotations

from dataclasses import dataclass
import math

from bridging.PRODIGY.dataset import make_cache_key as _make_cache_key
from bridging.PRODIGY.dataset import parse_request_row as _parse_prodigy_request_row


@dataclass(frozen=True)
class MMGBSARequest:
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


def make_cache_key(
    pdb_id: str,
    ligand_group: str,
    receptor_group: str,
    temperature_k: float | None,
) -> str:
    return _make_cache_key(pdb_id, ligand_group, receptor_group, temperature_k)


def parse_request_row(row: dict, row_index: int) -> MMGBSARequest:
    source_row_index = row.get("source_row_index", None)
    try:
        if source_row_index is not None and str(source_row_index).strip() != "":
            source_row_index = int(float(source_row_index))
        else:
            source_row_index = row_index
    except (TypeError, ValueError):
        source_row_index = row_index
    if isinstance(source_row_index, float) and not math.isfinite(source_row_index):
        source_row_index = row_index

    req = _parse_prodigy_request_row(row, row_index=int(source_row_index))
    return MMGBSARequest(
        row_index=req.row_index,
        pdb_id=req.pdb_id,
        ligand_group=req.ligand_group,
        receptor_group=req.receptor_group,
        temperature_k=req.temperature_k,
    )

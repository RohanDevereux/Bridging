from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Iterable


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class ChainSeqs:
    order: list[str]
    seqs: dict[str, str]
    residue_numbers: dict[str, tuple[int, ...]]


def _coerce_resseq(value) -> int | None:
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        pass
    matches = re.findall(r"-?\d+", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _line_chain_id(line: str) -> str:
    chain_id = ""
    if len(line) > 21:
        chain_id = str(line[21]).strip().upper()
    if chain_id:
        return chain_id
    resseq_token = line[22:27].strip() if len(line) >= 27 else line[22:].strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", resseq_token):
        return resseq_token[0].upper()
    return ""


def _load_chain_sequences(pdb_path: Path) -> ChainSeqs:
    order: list[str] = []
    seqs: dict[str, str] = {}
    residue_numbers: dict[str, tuple[int, ...]] = {}
    seen_residues: set[tuple[str, str, str, str]] = set()

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[0:6].strip().upper()
            if record not in {"ATOM", "HETATM"}:
                continue
            resname = line[17:20].strip().upper() if len(line) >= 20 else ""
            aa = AA3_TO_AA1.get(resname)
            if aa is None:
                continue
            chain_id = _line_chain_id(line)
            if not chain_id:
                continue
            resseq_token = line[22:27].strip() if len(line) >= 27 else line[22:].strip()
            icode = line[26].strip() if len(line) > 26 else ""
            residue_key = (chain_id, resseq_token, icode, resname)
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)
            if chain_id not in seqs:
                order.append(chain_id)
                seqs[chain_id] = ""
                residue_numbers[chain_id] = ()
            seqs[chain_id] = seqs[chain_id] + aa
            resid = _coerce_resseq(resseq_token)
            if resid is not None:
                residue_numbers[chain_id] = residue_numbers[chain_id] + (resid,)
    return ChainSeqs(order=order, seqs=seqs, residue_numbers=residue_numbers)


def load_chain_order(pdb_path: Path) -> list[str]:
    return list(_load_chain_sequences(pdb_path).order)


def _seq_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(a=a, b=b).ratio())


def _normalize_chain_ids(chain_ids: Iterable[str] | None) -> list[str]:
    if chain_ids is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for cid in chain_ids:
        norm = str(cid).strip().upper()
        if not norm or norm in seen:
            continue
        out.append(norm)
        seen.add(norm)
    return out


def _residue_overlap_score(raw_resids: tuple[int, ...], md_resids: tuple[int, ...]) -> float:
    if not raw_resids or not md_resids:
        return 0.0
    raw_set = set(int(x) for x in raw_resids)
    md_set = set(int(x) for x in md_resids)
    inter = len(raw_set.intersection(md_set))
    if inter <= 0:
        return 0.0
    return float(inter) / float(max(1, min(len(raw_set), len(md_set))))


def _relative_order_score(raw_idx: int, raw_total: int, md_idx: int, md_total: int) -> float:
    if raw_total <= 1 or md_total <= 1:
        return 1.0
    raw_pos = float(raw_idx) / float(max(1, raw_total - 1))
    md_pos = float(md_idx) / float(max(1, md_total - 1))
    return max(0.0, 1.0 - abs(raw_pos - md_pos))


def _pair_score(
    *,
    raw: ChainSeqs,
    md: ChainSeqs,
    raw_id: str,
    md_id: str,
    raw_rank: int,
    raw_total: int,
) -> tuple[float, dict]:
    raw_seq = raw.seqs.get(raw_id, "")
    md_seq = md.seqs.get(md_id, "")
    seq_score = _seq_score(raw_seq, md_seq)
    exact_seq = float(bool(raw_seq) and raw_seq == md_seq)
    residue_score = _residue_overlap_score(
        raw.residue_numbers.get(raw_id, ()),
        md.residue_numbers.get(md_id, ()),
    )
    direct_id = float(raw_id == md_id)
    order_score = _relative_order_score(
        raw_idx=raw_rank,
        raw_total=raw_total,
        md_idx=md.order.index(md_id),
        md_total=len(md.order),
    )
    total = (
        (6.0 * exact_seq)
        + (4.0 * seq_score)
        + (2.0 * residue_score)
        + (0.5 * direct_id)
        + (0.25 * order_score)
    )
    details = {
        "total": total,
        "seq_score": seq_score,
        "exact_seq": exact_seq,
        "residue_overlap": residue_score,
        "direct_id": direct_id,
        "order_score": order_score,
    }
    return total, details


def _best_query_assignment(
    *,
    raw: ChainSeqs,
    md: ChainSeqs,
    query_chains: list[str],
) -> tuple[dict[str, str], dict[str, dict]]:
    if len(query_chains) > len(md.order):
        raise RuntimeError(
            f"Cannot map {len(query_chains)} query chains onto {len(md.order)} MD chains."
        )

    raw_total = len(query_chains)
    pair_details: dict[tuple[str, str], dict] = {}
    for raw_rank, raw_id in enumerate(query_chains):
        for md_id in md.order:
            _score, details = _pair_score(
                raw=raw,
                md=md,
                raw_id=raw_id,
                md_id=md_id,
                raw_rank=raw_rank,
                raw_total=raw_total,
            )
            pair_details[(raw_id, md_id)] = details

    best_score = float("-inf")
    best_map: dict[str, str] = {}

    def _search(idx: int, used_md: set[str], current_score: float, current_map: dict[str, str]) -> None:
        nonlocal best_score, best_map
        if idx >= len(query_chains):
            if current_score > best_score:
                best_score = current_score
                best_map = dict(current_map)
            return

        raw_id = query_chains[idx]
        for md_id in md.order:
            if md_id in used_md:
                continue
            details = pair_details[(raw_id, md_id)]
            current_map[raw_id] = md_id
            used_md.add(md_id)
            _search(idx + 1, used_md, current_score + float(details["total"]), current_map)
            used_md.remove(md_id)
            current_map.pop(raw_id, None)

    _search(0, set(), 0.0, {})

    mapping_details = {
        raw_id: pair_details[(raw_id, md_id)]
        for raw_id, md_id in best_map.items()
    }
    return best_map, mapping_details


def build_raw_to_md_chain_map(
    raw_pdb: Path,
    md_topology_pdb: Path,
    *,
    query_chains: Iterable[str] | None = None,
) -> tuple[dict[str, str], list[str], dict]:
    raw = _load_chain_sequences(raw_pdb)
    md = _load_chain_sequences(md_topology_pdb)
    if not md.order:
        raise RuntimeError(f"No protein chains found in MD topology: {md_topology_pdb}")

    query_ids = _normalize_chain_ids(query_chains)
    missing_query = [cid for cid in query_ids if cid not in raw.seqs]

    chain_map: dict[str, str] = {}
    query_mapping_scores: dict[str, dict] = {}
    used_md: set[str] = set()
    if query_ids and not missing_query:
        query_map, query_mapping_scores = _best_query_assignment(
            raw=raw,
            md=md,
            query_chains=query_ids,
        )
        chain_map.update(query_map)
        used_md.update(query_map.values())

    pairs: list[tuple[float, str, str, dict]] = []
    for raw_id in raw.order:
        raw_seq = raw.seqs.get(raw_id, "")
        for md_id in md.order:
            md_seq = md.seqs.get(md_id, "")
            raw_rank = raw.order.index(raw_id)
            total, details = _pair_score(
                raw=raw,
                md=md,
                raw_id=raw_id,
                md_id=md_id,
                raw_rank=raw_rank,
                raw_total=len(raw.order),
            )
            pairs.append((total, raw_id, md_id, details))
    pairs.sort(key=lambda item: item[0], reverse=True)

    for score, raw_id, md_id, _details in pairs:
        if raw_id in chain_map or md_id in used_md:
            continue
        chain_map[raw_id] = md_id
        used_md.add(md_id)

    direct_overlap = sorted(set(raw.order).intersection(set(md.order)))
    map_scores = {
        raw_id: _seq_score(raw.seqs.get(raw_id, ""), md.seqs.get(md_id, ""))
        for raw_id, md_id in chain_map.items()
    }
    report = {
        "n_raw_chains": int(len(raw.order)),
        "n_md_chains": int(len(md.order)),
        "n_mapped": int(len(chain_map)),
        "direct_chain_id_overlap": direct_overlap,
        "mapping_scores": map_scores,
        "query_chains": list(query_ids),
        "missing_query_chains": list(missing_query),
        "query_mapping_scores": query_mapping_scores,
        "query_mapped": {cid: chain_map.get(cid, "") for cid in query_ids if cid in chain_map},
    }
    return chain_map, list(md.order), report


def remap_query_pair(
    *,
    query_chain_1: str,
    query_chain_2: str,
    chain_map: dict[str, str],
    md_chain_order: list[str],
    strict: bool = False,
) -> tuple[str, str]:
    if not md_chain_order:
        raise RuntimeError("Cannot remap query chains: empty MD chain list.")

    q1 = str(query_chain_1).strip().upper()
    q2 = str(query_chain_2).strip().upper()

    if strict:
        missing = [cid for cid in [q1, q2] if cid not in chain_map]
        if missing:
            raise RuntimeError(
                f"Could not remap query chains {missing}; available mapped={sorted(chain_map)} md={md_chain_order}"
            )
        r1 = chain_map[q1]
        r2 = chain_map[q2]
    else:
        r1 = chain_map.get(q1, q1 if q1 in md_chain_order else md_chain_order[0])
        r2 = chain_map.get(q2, q2 if q2 in md_chain_order else md_chain_order[0])

    if r1 == r2 and len(md_chain_order) > 1:
        for cid in md_chain_order:
            if cid != r1:
                r2 = cid
                break
    return r1, r2

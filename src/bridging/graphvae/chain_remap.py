from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from Bio.PDB import PDBParser


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


def _load_chain_sequences(pdb_path: Path) -> ChainSeqs:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(str(pdb_path.stem), str(pdb_path))
    model = next(structure.get_models())

    order: list[str] = []
    seqs: dict[str, str] = {}
    for chain in model.get_chains():
        chain_id = str(chain.id).strip().upper()
        chars: list[str] = []
        for residue in chain.get_residues():
            hetflag = str(residue.id[0]).strip()
            if hetflag not in ("", " "):
                continue
            resname = str(residue.resname).strip().upper()
            aa = AA3_TO_AA1.get(resname)
            if aa is not None:
                chars.append(aa)
        if chars:
            order.append(chain_id)
            seqs[chain_id] = "".join(chars)
    return ChainSeqs(order=order, seqs=seqs)


def _seq_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(a=a, b=b).ratio())


def build_raw_to_md_chain_map(raw_pdb: Path, md_topology_pdb: Path) -> tuple[dict[str, str], list[str], dict]:
    raw = _load_chain_sequences(raw_pdb)
    md = _load_chain_sequences(md_topology_pdb)
    if not md.order:
        raise RuntimeError(f"No protein chains found in MD topology: {md_topology_pdb}")

    pairs: list[tuple[float, str, str]] = []
    for raw_id in raw.order:
        raw_seq = raw.seqs.get(raw_id, "")
        for md_id in md.order:
            md_seq = md.seqs.get(md_id, "")
            pairs.append((_seq_score(raw_seq, md_seq), raw_id, md_id))
    pairs.sort(reverse=True)

    chain_map: dict[str, str] = {}
    used_md: set[str] = set()
    for score, raw_id, md_id in pairs:
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
    }
    return chain_map, list(md.order), report


def remap_query_pair(
    *,
    query_chain_1: str,
    query_chain_2: str,
    chain_map: dict[str, str],
    md_chain_order: list[str],
) -> tuple[str, str]:
    if not md_chain_order:
        raise RuntimeError("Cannot remap query chains: empty MD chain list.")

    q1 = str(query_chain_1).strip().upper()
    q2 = str(query_chain_2).strip().upper()

    r1 = chain_map.get(q1, q1 if q1 in md_chain_order else md_chain_order[0])
    r2 = chain_map.get(q2, q2 if q2 in md_chain_order else md_chain_order[0])

    if r1 == r2 and len(md_chain_order) > 1:
        for cid in md_chain_order:
            if cid != r1:
                r2 = cid
                break
    return r1, r2


from __future__ import annotations
# engine_rank.py
from typing import Dict, Iterable, Protocol, Tuple

class HasMeta(Protocol):
    @property
    def metadata(self) -> Dict[str, object]: ...

DocScore = Tuple[HasMeta, float]

def build_rank(pairs: Iterable[DocScore], name: str = "dense") -> Dict[str, int]:
    """
    (Document, score) iterable -> {chunk_id: 1-based rank}.
    If a chunk appears multiple times, keep the best (lowest) rank.
    """
    rank: Dict[str, int] = {}
    for i, (doc, _s) in enumerate(pairs, start=1):
        cid = doc.metadata.get("chunk_id")
        if not cid:
            continue
        # cid is object; ensure str for dict key
        scid = str(cid)
        if scid not in rank or i < rank[scid]:
            rank[scid] = i
    return rank

def build_score_map(pairs: Iterable[DocScore]) -> Dict[str, float]:
    """
    (Document, score) iterable -> {chunk_id: score} (only if chunk_id exists).
    """
    out: Dict[str, float] = {}
    for doc, s in pairs:
        cid = doc.metadata.get("chunk_id")
        if cid is None:
            continue
        out[str(cid)] = float(s)
    return out




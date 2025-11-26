
# C:\dev\GovernEdge_CLI\engine_query\engine_fusion.py

"""
This module provides simple fusion strategies for combining retrieval scores
from multiple sources (dense vectors, FTS/BM25, and SQL/structured data).
It supports weighted linear fusion and reciprocal-rank fusion (RRF),
helping balance signal strengths across heterogeneous retrievers.
"""

from __future__ import annotations
from typing import Dict
import logging

# --- logging setup ---
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def linear_fuse(dense: Dict[str, float], fts: Dict[str, float], w: Dict[str, float]) -> Dict[str, float]:
    """
    Linearly fuse dense and FTS scores using specified weights.
    """
    wd, wf = w.get("dense", 0.7), w.get("fts", 0.3)
    ids = set(dense) | set(fts)
    fused = {cid: wd * dense.get(cid, 0.0) + wf * fts.get(cid, 0.0) for cid in ids}
    logger.info("Linear fusion (2-way) → %d items (weights: dense=%.2f, fts=%.2f)", len(fused), wd, wf)
    return fused


def rrf_fuse(dense_rank: Dict[str, int], fts_rank: Dict[str, int], k: int = 60) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion (RRF): blends rankings from dense and FTS retrievers.
    Lower ranks (better positions) contribute higher scores.
    """
    ids = set(dense_rank) | set(fts_rank)
    fused = {
        cid: (1.0 / (k + dense_rank.get(cid, 1_000_000)))
            + (1.0 / (k + fts_rank.get(cid, 1_000_000)))
        for cid in ids
    }
    logger.info("RRF fusion → %d items (k=%d)", len(fused), k)
    return fused


def linear_fuse3(
    dense: Dict[str, float],
    fts: Dict[str, float],
    sql: Dict[str, float],
    w: Dict[str, float] = {"dense": 0.6, "fts": 0.25, "sql": 0.15},
) -> Dict[str, float]:
    """
    Linearly fuse scores from dense, FTS, and SQL retrievers.
    Default weights favor dense retrieval, with smaller contributions from FTS and SQL.
    """
    ids = set(dense) | set(fts) | set(sql)
    wd, wf, ws = w.get("dense", 0.6), w.get("fts", 0.25), w.get("sql", 0.15)
    fused = {
        cid: wd * dense.get(cid, 0.0)
           + wf * fts.get(cid, 0.0)
           + ws * sql.get(cid, 0.0)
        for cid in ids 
    }
    logger.info(
        "Linear fusion (3-way) → %d items (weights: dense=%.2f, fts=%.2f, sql=%.2f)",
        len(fused), wd, wf, ws
    )
    return fused


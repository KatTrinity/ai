# qe_api.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from config_base.config_k import KConfig
from engine_query.query_engine import QueryEngine  # your existing engine

@dataclass(frozen=True)
class QEOptions:
    prefilter_mode: str | None = None
    messy_words_lt: int = KConfig.MESSY_WORDS_LT
    pct: float = KConfig.PCT
    cap: int = KConfig.CAP
    floor: int = KConfig.FLOOR

@dataclass(frozen=True)
class QEResult:
    k_used: int
    corpus_n: int
    dense_hits: Sequence[Mapping[str, Any]]
    facet_filter: dict | None
    prefiltered: bool
    mode_hint: str | None

def run_qe_core(
    *,
    question: str,
    embed_key: str,
    ce_key: str,
    options: QEOptions = QEOptions(),
) -> QEResult:
    engine = QueryEngine()
    messy = len(question.split()) < options.messy_words_lt
    pf_mode = options.prefilter_mode or KConfig.PREFILTER_MODE

    prefiltered, facet_filter, mode_hint = engine.prefilter(question, pf_mode)

    k_used, corpus_n = engine.compute_k(
        embed_key=embed_key,
        messy=messy,
        prefiltered=prefiltered,
        pct=options.pct,
        cap=options.cap,
        floor=options.floor,
    )
    dense_hits = engine.search_retrieval(
        question=question,
        embed_key=embed_key,
        ce_key=ce_key,
        k=k_used,
        facet_filter=facet_filter,
    )
    return QEResult(
        k_used=k_used,
        corpus_n=corpus_n,
        dense_hits=dense_hits,
        facet_filter=facet_filter,
        prefiltered=prefiltered,
        mode_hint=mode_hint,
    )

# query_engine.py
# engine/query_engine.py
from __future__ import annotations
import time, logging, os, duckdb, time, json, socket , uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from langchain_core.documents import Document
from pathlib import Path
from statistics import mean

from langchain_core.messages import SystemMessage, HumanMessage

from config_base.config import Config
from config_base.config_k import KConfig
from engine_query.retriever_sql_duck import get_sql_master_data 
from engine_query.retriever_vector import get_vectorstore, count_corpus, to_chroma_filter
from engine_query.retriever_fts import fts_search_rows, normalize_bm25_to_pos
from engine_query.engine_cross_encoder import get_cross_encoder, choose_cpu_rerank, rerank_candidates
from engine_query.insert_chat_logs import insert_chat, upsert_prompt
from engine_query.engine_fusion import rrf_fuse, linear_fuse3
from engine_query.engine_load_models import get_embedder, get_llm, apply_query_prefix_for_embedder
from engine_query.engine_facets import facet_prefilter

from prepare_docs.db_io import get_conn
from utils.utils_k_dense import (
    resolve_collection_name,
    chroma_count,
    choose_k_dense, 
)

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- fusion helpers (placed BEFORE class QueryEngine) ---
def _build_rank(pairs: list[tuple], name: str = "dense") -> dict[str, int]:
    """[(Document, score)] -> {chunk_id: 1-based-rank}; keep best (lowest) rank on dups."""
    rank: dict[str, int] = {}
    for i, (doc, _s) in enumerate(pairs, start=1):
        cid = doc.metadata.get("chunk_id")
        if not cid:
            continue
        if cid not in rank or i < rank[cid]:
            rank[cid] = i
    return rank

def _build_score_map(pairs: list[tuple]) -> dict[str, float]:
    """[(Document, score)] -> {chunk_id: score} (only if chunk_id exists)."""
    return {
        doc.metadata.get("chunk_id"): float(s)
        for doc, s in pairs
        if doc.metadata.get("chunk_id")
    }

def _cfg_first(*names, default=None):
    """Return the first present Config attr among names, else default."""
    for n in names:
        if hasattr(Config, n):
            return getattr(Config, n)
    return default

def _as_str_path(v, default: str):
    if v is None or v == "":
        return str(Path(default))
    return str(Path(v))

DEFAULT_SYSTEM_PROMPT = (
    "You are a SAP S/4HANA and SAP ERP troubleshooting assistant. "
    "Use ONLY the provided context to answer. Prefer exact information "
    "from the documents over general SAP knowledge. "
    "If the context does not contain the answer, say so clearly. "
    "Keep answers short, technical, and correct."
)

def _read_prompt(val: Optional[str]) -> Optional[str]:
    """Allow SYSTEM_PROMPT_TEXT='@path/to/file.txt' to load from a file."""
    if not val:
        return None
    if isinstance(val, str) and val.startswith("@"):
        return Path(val[1:]).read_text(encoding="utf-8")
    return val

@dataclass
class RetrievalResult:
    # what the LLM/prompt builder and logger need
    final_docs: List[Any]                  # your Document objects
    scores_for_final: List[float]          # aligned to final_docs (dense proxy)
    #context_docs: List[Any]                # optional: top-N for prompt
    chosen_per_doc_cap: List[Any] 

    # timings
    dense_ms: int
    fts_ms: int
    sql_ms: int
    ce_ms: int
    total_ms: int
    # knobs actually used during the run
    k_dense: int
    k_rerank: int
    keep_top: int
    per_doc_cap: int
    prefiltered: int
    similarity_avg: int
    # debug/meta
    run_meta: Dict[str, Any] = field(default_factory=dict)

class QueryEngine:
    def __init__(
        self,
        *,
        chroma_dir_map: Dict[str, str],
        llm_map: Dict[str, str],
        embed_map: Dict[str, str],
        ce_map: Dict[str, str],
        db_path: str,
        md_db_path: str,
        chunks_table: str,
        fusion_weights: Dict[str, float],
        rrf_k: int,
        system_prompt_text: str | None = None, 
        use_rrf: bool = False,
    ):
        self._chroma_dir_map = chroma_dir_map or {}
        self._llm_map = llm_map or {}
        self._embed_map = embed_map or {}
        self._ce_map = ce_map or {}
        self.db_path = db_path
        self.md_db_path = md_db_path
        self.chunks_table = chunks_table
        self.fusion_weights = fusion_weights or {"dense": 0.55, "fts": 0.30, "sql": 0.15}
        self.rrf_k = int(rrf_k)
        self.use_rrf = bool(use_rrf) 
        self.system_prompt_text = system_prompt_text or DEFAULT_SYSTEM_PROMPT

        self._embedder_cache: Dict[str, Any] = {}
        self._vs_cache: Dict[str, Any] = {}
        self._llm_cache: Dict[str, Any] = {}

    @classmethod
    def from_env(cls) -> "QueryEngine":
        # tolerate DB_PATH / db_path; MASTER_DB_PATH / master_db_path; etc.
        db_path_val = _cfg_first("DB_PATH", "db_path",
                                 default=Path("database") / "chat_logs.sqlite")
        md_db_path_val = _cfg_first("MASTER_DB_PATH", "master_db_path", "MD_DB_PATH",
                                    default=db_path_val)
        
        # NEW: pull system prompt from env/config, allow @file
        sys_prompt_val = _cfg_first("SYSTEM_PROMPT_TEXT", "system_prompt_text", default=DEFAULT_SYSTEM_PROMPT)
        sys_prompt_val = _read_prompt(sys_prompt_val)
        
        return cls(
            chroma_dir_map=_cfg_first("CHROMA_DIR_MAP", "chroma_dir_map", default={}) or {},
            llm_map=_cfg_first("LLM_MODEL_MAP", "llm_model_map", default={}) or {},
            embed_map=_cfg_first("EMBEDDING_MODEL_MAP", "embedding_model_map", default={}) or {},
            ce_map=_cfg_first("CE_MODEL_MAP", "ce_model_map", default={}) or {},
            db_path=_as_str_path(db_path_val, "database/chat_logs.sqlite"),
            md_db_path=_as_str_path(md_db_path_val, _as_str_path(db_path_val, "database/chat_logs.sqlite")),
            chunks_table=_cfg_first("CHUNKS_TABLE", "chunks_table", default="doc_chunks"),
            fusion_weights=_cfg_first("FUSION_WEIGHTS", "fusion_weights",
                                      default={"dense": 0.55, "fts": 0.30, "sql": 0.15}) or
                           {"dense": 0.55, "fts": 0.30, "sql": 0.15},
            rrf_k=int(_cfg_first("RRF_K", "rrf_k", default=60)),
            system_prompt_text=sys_prompt_val,
        )
      
# ---- tiny helpers exposed to UI ----
    @property
    def llm_keys(self):   return list(self._llm_map.keys())
    @property
    def embed_keys(self): return list(self._embed_map.keys())
    @property
    def ce_keys(self):    return list(self._ce_map.keys())
    
# --- inside QueryEngine -------------------------------------------------------

    def compute_k( 
        self,
        *,
        embed_key: str,
        messy: bool,
        prefiltered: bool,
        pct: float,
        cap: int,
        floor: int,
    ) -> Tuple[int, int]:
        """
        Compute candidate-pool K based on corpus size and heuristics.
        Returns (k_used, corpus_size).
        """
        n = self.corpus_size(embed_key=embed_key) 
        k = choose_k_dense(
            n,
            pct=pct,
            cap=cap,
            floor=floor,
            messy_query=messy, # = None??
            prefiltered=prefiltered,
        )
        return max(1, int(k)), n

    def corpus_size(self, *, embed_key: str, strategy: str = "direct") -> int:
        """
        Count vectors in the collection for the given embedder key.

        strategy:
        - "cached": uses embedder cache + vs_cache (fast path)
        - don't used cached because of turn over
        - "direct": resolves collection + fresh chroma_count (bypass cache)
        """
        if strategy == "cached-no":
            return self._corpus_size_cached(embed_key)
        elif strategy == "direct":
            return self._corpus_size_direct(embed_key)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

# ---- private helpers ---------------------------------------------------------

    def _corpus_size_direct(self, embed_key: str) -> int:
        """Direct count via collection + chroma directory (no vs_cache)."""
        embed_model_name = self._embed_map[embed_key]
        emb = self.get_embedder(embed_key, embed_model_name)
        collection = resolve_collection_name(embed_key)
        chroma_dir = self._chroma_dir_map[embed_key]
        return chroma_count(collection, chroma_dir, emb)

    def _corpus_size_cached(self, embed_key: str) -> int:
        """Count using embedder cache + vector-store cache (fast path)."""
        emb = get_embedder(self._embedder_cache, self._embed_map[embed_key])
        return count_corpus(
            model_key=embed_key,
            embeddings=emb,
            cache=self._vs_cache,
            persist_dir=self._chroma_dir_map[embed_key],
            # collection_name=resolve_collection_name(embed_key)  # optional
        )

    def prefilter(self, question: str, mode: str):
        return facet_prefilter(question, mode)
    
# ---- core search
    def search_retrieval(
        self,
        *,
        question: str,
        embed_key: str,
        ce_key: str,
        k: int,
        facet_filter: dict | None = None,
    ) -> RetrievalResult:
        """
        Retrieval-only. No LLM. No logging. Returns unified hits.
        """
            # 1) dense/fts/fusion/rerank work you already do
            # 2) return a normalized list[dict] (each hit with id, score, text, meta)
        
        embed_model_name = self._embed_map[embed_key]
        embedder = get_embedder(self._embedder_cache, embed_model_name)
        vs       = get_vectorstore(cache=self._vs_cache, model_key=embed_key,embeddings=embedder,persist_dir=self._chroma_dir_map[embed_key],)


# --- Dense (Vector Store) with filter fallback ------
        q_dense = apply_query_prefix_for_embedder(question, embed_model_name)
        where   = to_chroma_filter(facet_filter)  # whatever builds {'$and': [{'object': {'$eq': ...}}, ...]}

        logger.info("[VEC] model=%s k=%s filter_raw=%r q_len=%d", embed_model_name, k, facet_filter, len(q_dense or ""))
        logger.info("[VEC] filter_chroma=%r", where)

        def _search_vec(_where):
            t0 = time.time()
            pairs = vs.similarity_search_with_score(q_dense, k=k, filter=_where) if _where else \
                    vs.similarity_search_with_score(q_dense, k=k)
            ms = int((time.time() - t0) * 1000)
            docs   = [d for d, _ in pairs]
            scores = [float(s) for _, s in pairs]
            by_id  = {d.metadata.get("chunk_id"): s for d, s in zip(docs, scores)}
            if docs:
                logger.info("[VEC] hits=%d time=%d ms score[min/avg/max]=[%.4f/%.4f/%.4f]",
                            len(docs), ms, min(scores), mean(scores), max(scores))
            else:
                logger.warning("[VEC] 0 hits time=%d ms (filter=%r)", ms, _where)
            return pairs, docs, scores, by_id, ms

        # try with filter
        dense_pairs, dense_docs, dense_scores, dense_by_id, dense_ms = _search_vec(where)

       # automatic fallback without filter if zero hits AND you had a filter
        if not dense_docs and where:
            logger.warning("[VEC] retrying WITHOUT filter (diagnostic fallback)")
            dense_pairs, dense_docs, dense_scores, dense_by_id, dense_ms = _search_vec(None)

        # (RE)BUILD rank and score maps from the FINAL dense_pairs
        dense_rank  = _build_rank(dense_pairs, name="dense")
        dense_by_id = _build_score_map(dense_pairs)  # keep in sync with pairs


# -----FTS ------------------ 
        question = "" if question is None else str(question)

        t_fts = time.perf_counter()
        fts_rows = fts_search_rows(self.db_path, question, k)   # (+ facet args if you use them)
        fts_ms = (time.perf_counter() - t_fts) * 1000.0

        fts_by_id = normalize_bm25_to_pos(fts_rows)
        fts_rank  = {r["chunk_id"]: i for i, r in enumerate(fts_rows, 1)}

        logger.info("FTS recall → %d rows in %.1f ms", len(fts_rows), fts_ms)
     
# --- SQL retriever (structured master-data) --- 

        logger.info(f"[SQL] md_db_path = {os.path.abspath(str(self.md_db_path)) if self.md_db_path else 'None'}")

        try:
            t_sql = time.time()
            sql_docs, sql_scores, sql_by_id = get_sql_master_data(str(self.md_db_path), question, k)
            logger.info("[SQL] done in %.0f ms → docs=%d", (time.time()-t_sql)*1000, len(sql_docs))
        except Exception as e:
            import traceback, z_other.streamlit_old as st
            tb = traceback.format_exc()
            logger.error("[SQL] fatal: %s", e)
            sql_docs, sql_scores, sql_by_id = [], {}, {}

        # pool size
        k = max(10, k)

        dbp = Path(self.md_db_path) if self.md_db_path else None
        if dbp and dbp.exists():
            # List tables/views (DuckDB)
            try:
                with duckdb.connect(database=str(dbp), read_only=True) as con:
                    tbls = (
                        con.execute("""
                            SELECT table_name
                            FROM information_schema.tables
                            WHERE table_schema = 'main'
                            AND table_type IN ('BASE TABLE','VIEW')
                            ORDER BY 1
                        """).fetchall()
                    )
                tbls = [t[0] for t in tbls]
                #logger.info(f"[SQL] md_db_path = {os.path.abspath(str(dbp))}")
                #logger.info(f"[SQL] tables/views: {sorted(tbls)[:25]} ... (total={len(tbls)})")
                logger.info(f"[SQL] checking tables...")
            except Exception as e:
                logger.warning(f"[SQL] failed to enumerate tables: {e}")

            # Run the DuckDB retriever
            t_sql = time.time()
            sql_docs, sql_scores, sql_by_id = get_sql_master_data(str(dbp), question, k)
            logger.info(f"[SQL] completed in {time.time()-t_sql:.3f}s "
                        f"→ {len(sql_docs)} docs, {len(sql_scores)} scores")
        else:
            logger.warning("[SQL] disabled: md_db_path invalid")
            sql_docs, sql_scores, sql_by_id = [], {}, {}

        sql_ms = int((time.time() - t_sql) * 1000)

        # --- Index by chunk_id for fusion (ensure mapping is complete) ---
        sql_by_id = {d.metadata["chunk_id"]: sql_scores.get(d.metadata["chunk_id"], 0.0) for d in sql_docs}


# --- Fusion (3-way) ---
        if self.use_rrf:
            # 1) RRF across dense+fts (rank-only inputs)
            rrf_scores = rrf_fuse(dense_rank, fts_rank, k=self.rrf_k)  # {chunk_id: score}

            # 2) Blend RRF result with SQL scores (simple linear blend)
            w_sql = float(self.fusion_weights.get("sql", 0.2))
            w_rrf = 1.0  # treat RRF as the base
            fused_scores = {cid: w_rrf * rrf_scores.get(cid, 0.0) + w_sql * sql_by_id.get(cid, 0.0)
                            for cid in set(rrf_scores) | set(sql_by_id)}
        else:
            # Standard linear 3-way (requires weights that include 'sql')
            # e.g., Config.FUSION_WEIGHTS = {"dense": 0.55, "fts": 0.30, "sql": 0.15}
            fused_scores = linear_fuse3(dense_by_id, fts_by_id, sql_by_id, self.fusion_weights)

         # Build per-cid score_json blobs once
        # ---- Build per-cid score_json once
        strategy_tag = {"strategy": "rrf", "rrf_k": self.rrf_k} if self.use_rrf else {"strategy": "linear3", "w": dict(self.fusion_weights)}
        score_json_map = {}
        for cid in set(dense_by_id) | set(fts_by_id) | set(sql_by_id) | set(fused_scores):
            score_json_map[cid] = {
                "dense": float(dense_by_id.get(cid, 0.0)),
                "fts":   float(fts_by_id.get(cid, 0.0)),
                "sql":   float(sql_by_id.get(cid, 0.0)),
                "fused": float(fused_scores.get(cid, 0.0)),
                **strategy_tag,
            }

        # ---- Hydrate the pool ONCE (dense -> FTS -> SQL) and attach score_json on each
        by_id_doc: dict[str, Document] = {}

        # dense docs first
        for d in dense_docs:
            cid = d.metadata.get("chunk_id")
            if not cid:
                continue
            d.metadata.setdefault("span_start", 0)
            d.metadata.setdefault("span_end", len(d.page_content or ""))
            d.metadata["score_json"] = score_json_map.get(cid, {})
            by_id_doc[cid] = d

        # FTS rows → Documents (use body_raw, not body)
        for r in fts_rows:
            cid = r["chunk_id"]
            if cid in by_id_doc:
                by_id_doc[cid].metadata.setdefault("score_json", score_json_map.get(cid, {}))

                continue
            doc = Document(
                page_content=r["body_raw"],
                metadata={
                    "chunk_id": cid,
                    "doc_id": r["doc_id"],
                    "title": r.get("title"),
                    "header_path": r.get("header_path"),
                    "span_start": 0,
                    "span_end": len(r["body_raw"] or ""),
                },
            )
            doc.metadata["score_json"] = score_json_map.get(cid, {})
            by_id_doc[cid] = doc

        # SQL docs (also ensure pre-existing entries get score_json)
        for d in sql_docs:
            cid = d.metadata.get("chunk_id")
            if not cid: 
                continue
            if cid in by_id_doc:
                by_id_doc[cid].metadata.setdefault("score_json", score_json_map.get(cid, {}))
            else:
                d.metadata["score_json"] = score_json_map.get(cid, {})
                by_id_doc[cid] = d

        # --- Final top-K selection from ONE fused map ---
        top_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
        candidates = [by_id_doc[cid] for cid in top_ids]

       # --- Cross-encoder rerank (optional) ---
        ce_ms = 0
        final_docs = candidates
        chosen_per_doc_cap = 3  # keep this in one place
        if ce_key:
            ce_model_name = self._ce_map.get(ce_key) or "BAAI/bge-reranker-base"
            ce = get_cross_encoder(ce_model_name)
            knobs    = choose_cpu_rerank(len(candidates))
            k_rerank = min(knobs["k_rerank"], len(candidates))
            keep_top = min(knobs["keep_top"], k)

            t_ce0 = time.time()
            rerank_pool = candidates[:k_rerank]

            final_docs, ranked = rerank_candidates(
                user_query=question,
                candidates=rerank_pool,
                ce=ce,
                batch_size=knobs["batch"],
                per_doc_cap=chosen_per_doc_cap,
                keep_top=keep_top,
            )
            ce_ms = int((time.time() - t_ce0) * 1000)
        else:
            k_rerank = len(candidates)
            keep_top = min(len(candidates), k)
            final_docs = candidates
            ranked = [(d, dense_by_id.get(d.metadata.get("chunk_id"))) for d in final_docs]  # optional

        # right after you compute `ranked`
        ranked_serializable = []
        for d, s in ranked:
            ranked_serializable.append({
                "chunk_id": d.metadata.get("chunk_id"),
                "doc_id": d.metadata.get("doc_id"),
                "ce_score": float(s) if s is not None else None,
            })

        scores_for_final: list[float | None] = [
            dense_by_id.get(d.metadata.get("chunk_id")) for d in final_docs
]
        logger.info(
            "Hybrid recall → dense=%s in %s ms; fts=%s in %s ms; sql=%s in %s ms; "
            "fused_pool=%s; ce(%s→%s) in %s ms",
            len(dense_docs), dense_ms,
            len(fts_rows),   fts_ms,
            len(sql_docs),   sql_ms,
            len(candidates),
            k_rerank, len(final_docs), ce_ms
        )

        # Total latency (sum the parts; guard None)
        def _n(x): return int(x) if isinstance(x, (int, float)) else 0
        total_ms = _n(dense_ms) + _n(fts_ms) + _n(sql_ms) + _n(ce_ms)

        # similarity over final docs (using dense score proxy)
        dense_vals = [s for s in scores_for_final if s is not None]
        similarity_avg = (sum(dense_vals) / len(dense_vals)) if dense_vals else None

        run_meta = {
            "app": {"version": "kb.v1", "build_id": os.getenv("BUILD_ID")},
            "env": {"host": socket.gethostname(), "pid": os.getpid()},
            "query": {"prefiltered": bool(facet_filter), "facet": facet_filter or {}},
            "retrievers": {
                "k_dense": k,
                "k_rerank": k_rerank,
                "keep_top": keep_top,
                "per_doc_cap": chosen_per_doc_cap,
                "ranked": ranked_serializable,
            },
            "timings_ms": {"dense": dense_ms, "fts": int(fts_ms), "sql": sql_ms, "ce": ce_ms},
            "weights": self.fusion_weights,
            "collection": self._chroma_dir_map.get(embed_key),
        } 

        #scores_for_final = [dense_by_id.get(d.metadata.get("chunk_id")) for d in final_docs]
        return RetrievalResult(
            final_docs=final_docs,
            scores_for_final=scores_for_final,
            #context_docs=context_docs,
            dense_ms=dense_ms,
            fts_ms=int(fts_ms),
            sql_ms=sql_ms,
            ce_ms=ce_ms,
            total_ms=total_ms,
            k_dense=k,
            k_rerank=k_rerank,
            keep_top=keep_top,
            per_doc_cap=chosen_per_doc_cap,
            prefiltered=bool(facet_filter),
            run_meta=run_meta,
            similarity_avg=similarity_avg,
            chosen_per_doc_cap=chosen_per_doc_cap

        )
        

##########################################################################################################

    def search(
        self,
        *,
        question: str,
        llm_key: str,
        embed_key: str,
        ce_key: str,
        k: int,
        session_id: str | None,
        facet_filter: dict | None = None,
    ) -> Tuple[str, str, List[Document], int]:
      
        # A) Retrieval
        retrieval_results = self.search_retrieval(
            question=question,
            embed_key=embed_key,
            ce_key=ce_key,
            k=k,
            facet_filter=facet_filter,
    )
       
        context_docs = retrieval_results.final_docs[:12]  
        # B) Prompt build (use rr.context_docs or rr.final_docs)
        context = "\n\n---\n\n".join(
        (getattr(d, "page_content", "") or getattr(d, "text", "") or "")
        for d in context_docs
    )
        # System prompt + user content built once
        system_prompt_text = self.system_prompt_text or ""
        user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        # Messages used for BOTH hashing + LLM
        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_content},
        ]

        # ------------- Prompt hash -------------
        with get_conn(self.db_path, ensure=True) as _conn_for_hash:
            prompt_hash = upsert_prompt(_conn_for_hash, system_prompt_text, messages)

        # For LLM: LC message objects
        lc_messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=user_content),
        ]

        # ------------- LLM call -------------
        llm_model_name = self._llm_map[llm_key]
        llm = get_llm(self._llm_cache, llm_model_name)

        # If ChatOllama accepts dict-style messages directly:
        answer = llm.invoke(lc_messages)

        logger.info("query_engine") 
        ce_model_name = self._ce_map.get(ce_key) if ce_key else None
        turn_id = None
        if session_id:
            try:

                # 🔧 normalize spans on all final docs before logging
                #for d in retrieval_results.final_docs:
                    #meta = getattr(d, "metadata", {}) or {}
                    #text = getattr(d, "page_content", "") or getattr(d, "text", "") or ""
                    #meta.setdefault("span_start", 0)
                    #meta.setdefault("span_end", len(text))
                    #d.metadata = meta  # in case it was a plain dict copy

                with get_conn(self.db_path, ensure=True) as log_conn:
                    turn_id = insert_chat(
                        conn=log_conn,
                        session_id=session_id,
                        question=question,
                        response=answer if isinstance(answer, str) else str(answer),
                        llm_key=llm_key,
                        embed_key=embed_key,
                        ce_key=ce_key,
                        ce_model=ce_model_name if ce_key else None,
                        top_k=k,                         # → k_dense
                        k_rerank=retrieval_results.k_rerank,
                        keep_top=retrieval_results.keep_top,               # final keep_top
                        per_doc_cap=retrieval_results.chosen_per_doc_cap,  # was None
                        collection=self._chroma_dir_map[embed_key],
                        latency_ms=retrieval_results.total_ms,
                        dense_ms=retrieval_results.dense_ms,
                        ce_ms=retrieval_results.ce_ms,
                        similarity_avg=retrieval_results.similarity_avg,
                        prefiltered=int(bool(facet_filter)),
                        docs=retrieval_results.final_docs,                 # ← log what you used
                        scores=retrieval_results.scores_for_final,         # ← aligned to final_docs
                        prompt_hash=prompt_hash,         # ensure this was computed earlier
                        run_meta=json.dumps(retrieval_results.run_meta, ensure_ascii=False),
                    )
            except Exception as e:
                logger.warning("chat logging failed: %s", e)

        return answer, turn_id

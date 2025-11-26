# retriever_sql_duck.py
"""
DuckDB retriever for structured master-data rows → LangChain Documents.

Features:
  • LIKE/INSTR across selected columns (auto-cast non-text → VARCHAR)
  • Token OR logic, case-insensitive, SQL-side relevance score
  • Numeric-token retry + safe top-N fallback
  • Works with:
      - core tables/views (md_material, md_customer, md_sales_order, etc.)
      - *_metadata tabs (field/description/data_element/domain)
      - md_row_kv skinny store
"""

from __future__ import annotations
import re, logging, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Sequence, Any, Tuple

import duckdb
from langchain_core.documents import Document

from config_base.config import Config  # unchanged

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
#if not logger.handlers:  # only add once
    #_h = logging.StreamHandler(sys.stdout)
    #_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    #logger.addHandler(_h)
    #logger.setLevel(logging.INFO)

MASTER_DB_PATH = Path(getattr(Config, "MASTER_DB_PATH", "database/master_data.duckdb"))

@dataclass(frozen=True)
class TableSpec:
    id_key: str
    like_cols: list[str]
    preview_cols: list[str]
    fabricate_id: bool = False

# --- Table specs (you can extend as needed) ---
TABLE_SPECS: dict[str, TableSpec] = {
    # core tables
    "md_material": TableSpec(
        id_key="material_id",
        like_cols=["material_id", "matnr", "mat_type", "ind_sector", "description"],
        preview_cols=["matnr", "mat_type", "ind_sector", "description"],
    ),
    "md_customer": TableSpec(
        id_key="customer_id",
        like_cols=["customer_id", "kunnr", "name", "country", "tax_class"],
        preview_cols=["kunnr", "name", "country", "tax_class"],
    ),
    "md_sales_order": TableSpec(
        id_key="vbeln",
        like_cols=["vbeln", "matnr", "country", "tax_code", "customer_id"],
        preview_cols=["vbeln", "matnr", "country", "tax_code", "customer_id"],
    ),
    # helpful views
    "vw_sales_tax_mismatch": TableSpec(
        id_key="vbeln",
        like_cols=["vbeln", "matnr", "mat_type", "country", "tax_code"],
        preview_cols=["vbeln", "matnr", "mat_type", "country", "tax_code"],
    ),
    "vw_missing_material_views": TableSpec(
        id_key="_id",  # fabricate "{matnr}::{required_view}"
        like_cols=["matnr", "mat_type", "required_view"],
        preview_cols=["matnr", "mat_type", "required_view"],
        fabricate_id=True,
    ),
    # skinny KV (optional, enabled if table exists)
    "md_row_kv": TableSpec(
        id_key="rowid",  # DuckDB can expose rowid via hidden column, but safer to fabricate
        like_cols=[
            "entity", "level", "business_key", "key_json",
            "key_matnr", "key_werks", "key_bukrs", "key_kokrs", "key_vkorg", "key_vtweg", "key_spras",
            "field_tech", "value", "scalar_category", "scalar_object", "tags_json", "source",
        ],
        preview_cols=["entity", "level", "business_key", "field_tech", "value", "source"],
        fabricate_id=True,  # we’ll synthesize a stable id
    ),
    # metadata tabs (pattern match added at runtime)
}

# ---------------------------------------------------------------------------
# DB helpers (DuckDB)
# ---------------------------------------------------------------------------
def _open_ro(md_db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    dbp = str(Path(md_db_path or MASTER_DB_PATH))
    return duckdb.connect(database=dbp, read_only=True)

def _fetch_records(conn, sql: str, params=None) -> list[dict]:
    cur = conn.execute(sql, params or [])
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    if not cols:
        return []  # e.g., DDL or PRAGMA
    return [dict(zip(cols, r)) for r in rows]

def _list_tables_and_views(md_db_path):
    sql = """
      SELECT table_schema, table_name
      FROM information_schema.tables
      WHERE table_type IN ('BASE TABLE','VIEW')
      ORDER BY 1,2
    """
    with _open_ro(md_db_path) as con:
        rows = _fetch_records(con, sql)
    # qualify to avoid collisions; keep simple name list too if you prefer
    return [r["table_name"] for r in rows]  # or [(r["table_schema"], r["table_name"])]

def _existing_cols(md_db_path, table):
    sql = """
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = ?
    """
    with _open_ro(md_db_path) as con:
        rows = _fetch_records(con, sql, [table])
    return {r["column_name"] for r in rows}

def _texty_cols(md_db_path, table):
    sql = """
      SELECT column_name, data_type
      FROM information_schema.columns
      WHERE table_name = ?
    """
    with _open_ro(md_db_path) as con:
        rows = _fetch_records(con, sql, [table])
    def _is_text(t: str) -> bool:
        t = (t or "").upper()
        return ("CHAR" in t) or ("TEXT" in t) or ("STRING" in t) or ("VARCHAR" in t)
    return {r["column_name"]: _is_text(r["data_type"]) for r in rows}


def _col_expr(col: str, texty: dict[str, bool]) -> str:
    return col if texty.get(col, True) else f"CAST({col} AS VARCHAR)"

# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------
def _tokenize(question: str, max_tokens: int = 5) -> Tuple[list[str], list[str]]:
    toks = [t for t in re.split(r"[^\w\-/_]+", question or "") if len(t) > 1][:max_tokens]
    nums = [t for t in toks if any(ch.isdigit() for ch in t)]
    return toks, nums

def _build_where_and_score(cols: list[str], texty: dict[str, bool], tokens: list[str]):
    where_groups, where_params = [], []
    score_terms, score_params = [], []
    for c in cols:
        expr = f"LOWER({_col_expr(c, texty)})"
        per_col = []
        for t in tokens:
            tl = t.lower()
            per_col.append(f"INSTR({expr}, ?) > 0")
            where_params.append(tl)
            score_terms.append("CASE WHEN INSTR({expr}, ?) > 0 THEN 1 ELSE 0 END".format(expr=expr))
            score_params.append(tl)
        if per_col:
            where_groups.append("(" + " OR ".join(per_col) + ")")
    where_sql = " OR ".join(where_groups) if where_groups else "1=0"
    score_sql = " + ".join(score_terms) if score_terms else "0"
    return where_sql, score_sql, where_params, score_params

def _fabricate_ids_if_needed(table: str, spec: TableSpec, rows: list[dict]) -> list[dict]:
    if not spec.fabricate_id:
        return rows
    out = []
    if table == "vw_missing_material_views":
        for r in rows:
            rid = f"{r.get('matnr')}::{r.get('required_view')}"
            out.append({**r, "_id": rid})
        return out
    if table == "md_row_kv":
        # Make a stable composite key: entity|level|business_key|field_tech|key_json hash
        for r in rows:
            rid = f"{r.get('entity')}|{r.get('level')}|{r.get('business_key')}|{r.get('field_tech')}|{r.get('key_json')}"
            out.append({**r, "_id": rid})
        return out
    return rows

# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------
def table_like_search(
    *, md_db_path: str | Path, table: str, like_cols: list[str],
    question: str, k: int = 20, exact_filters: Dict[str, Any] | None = None
) -> List[Dict]:
    raw = (question or "").strip()
    if not raw:
        return []

    avail = _existing_cols(md_db_path, table)
    cols = [c for c in like_cols if c in avail]
    if not cols:
        logger.warning("[SQL] %s: no overlap; requested=%s available=%s", table, like_cols, sorted(avail))
        return []

    texty = _texty_cols(md_db_path, table)
    toks, nums = _tokenize(raw)

    def _run(tokens: list[str], limit: int) -> list[dict]:
        if not tokens:
            return []
        where_sql, score_sql, where_params, score_params = _build_where_and_score(cols, texty, tokens)
        filt_sql, filt_params = "", []
        if exact_filters:
            parts = []
            for fk, fv in exact_filters.items():
                parts.append(f"{fk} = ?")
                filt_params.append(fv)
            if parts:
                filt_sql = " AND " + " AND ".join(parts)

        sql = f"""
          SELECT *, ({score_sql}) AS _score
          FROM {table}
          WHERE {where_sql}{filt_sql}
          ORDER BY _score DESC
          LIMIT ?
        """
        params = where_params + score_params + filt_params + [int(limit)]
        with _open_ro(md_db_path) as con:
            rows = _fetch_records(con, sql, params)
        logger.info("[SQL] %s hits=%d (cols=%s, toks=%s)", table, len(rows), cols, tokens)
        return rows

    # Pass 1: all tokens
    rows = _run(toks, k)
    if rows:
        return rows
    # Pass 2: numeric-only tokens
    rows = _run(nums, k) if nums else []
    if rows:
        return rows
    # Pass 3: tiny fallback
    logger.info("[SQL] %s fallback top-N", table)
    with _open_ro(md_db_path) as con:
        rows = _fetch_records(con, f"SELECT * FROM {table} LIMIT ?", [min(k, 10)])
    for r in rows:
        r["_fallback"] = 1
        r["_score"] = 0
    return rows

# ---------------------------------------------------------------------------
# Documents + scoring
# ---------------------------------------------------------------------------
def rows_to_documents(rows: List[Dict], table: str, id_key: str, preview_cols: Sequence[str] | None = None) -> List[Document]:
    docs: List[Document] = []
    pc = list(preview_cols or [])
    for r in rows:
        cid_val = r.get(id_key) or r.get("_id") or r.get("rowid")
        cid = str(cid_val) if cid_val is not None else "?"
        preview_pairs = [f"{c}={r.get(c)}" for c in pc if c in r]
        preview = ", ".join(preview_pairs) if preview_pairs else str({k: r[k] for k in list(r)[:6]})
        content = f"[{table}] {preview}"
        meta = {
            "chunk_id": f"{table}:{cid}",
            "doc_id":   cid,
            "table":    table,
            "row":      r,
            "title":    f"{table}:{cid}",
            "header_path": "",
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs

def score_sql_rows(rows: List[Dict], question: str, like_cols: Sequence[str], table: str, id_key: str) -> Dict[str, float]:
    term_l = (question or "").lower().strip()
    scores: Dict[str, float] = {}
    denom = float(max(1, len(like_cols)))
    for r in rows:
        hits = 0
        for c in like_cols:
            v = r.get(c)
            if isinstance(v, str) and term_l and term_l in v.lower():
                hits += 1
        base = min(1.0, hits / denom)
        penalty = 0.2 if r.get("_fallback") else 0.0
        sql_boost = min(0.3, 0.02 * float(r.get("_score", 0) or 0))
        key_id = r.get(id_key) or r.get("_id") or r.get("rowid")
        key = f"{table}:{key_id}"
        scores[key] = max(0.0, min(1.0, base - penalty + sql_boost))
    return scores

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_sql_retrieval(
    *, md_db_path: str | Path, question: str, k: int = 20,
    limit_tables: list[str] | None = None, log_counts_once: bool = True
) -> tuple[list[Document], dict[str, float], dict[str, float]]:
    dbp = Path(md_db_path)
    if not dbp.exists():
        logger.warning("[SQL] disabled: md_db_path invalid → %s", dbp)
        return [], {}, {}

    present = set(_list_tables_and_views(dbp))

    BASE_SPECS = TABLE_SPECS  # keep your original constant
    specs = dict(BASE_SPECS)  # local copy each call   
    
    # auto-discover without mutating global
    meta_tabs = [t for t in present if t.endswith("_metadata")]
    for t in meta_tabs:
        if t not in specs:
            specs[t] = TableSpec(
                id_key="field",
                like_cols=["field", "description", "data_element", "domain"],
                preview_cols=["field", "description", "domain"],
            )

    # Default target = all known specs that are present
    if limit_tables:
        target = [t for t in (limit_tables or specs.keys()) if t in present]
    else:
        target = [t for t in TABLE_SPECS if t in present]

    logger.info("[SQL] present=%d tables/views", len(present))
    if log_counts_once and logger.isEnabledFor(logging.DEBUG):
        with _open_ro(dbp) as con:
            for t in target[:50]:  # cap to avoid spam
                try:
                    n = _fetch_records(con, f"SELECT COUNT(*) AS n FROM {t}")[0]["n"]
                    logger.debug("[SQL] %s rows=%d", t, n)
                except Exception as e:
                    logger.debug("[SQL] %s count failed: %s", t, e)

    sql_docs: list[Document] = []
    sql_scores: dict[str, float] = {}

    for table in target:
        spec = TABLE_SPECS[table]
        # intersect requested like_cols with actual columns
        use_like_cols = [c for c in spec.like_cols if c in _existing_cols(dbp, table)]
        if not use_like_cols:
            continue

        rows = table_like_search(md_db_path=dbp, table=table, like_cols=use_like_cols, question=question, k=k)
        if not rows:
            continue

        rows = _fabricate_ids_if_needed(table, spec, rows)
        docs = rows_to_documents(rows, table=table, id_key=spec.id_key, preview_cols=spec.preview_cols)
        sc = score_sql_rows(rows, question=question, like_cols=use_like_cols, table=table, id_key=spec.id_key)

        sql_docs.extend(docs)
        sql_scores.update(sc)

    sql_by_id = {d.metadata["chunk_id"]: sql_scores.get(d.metadata["chunk_id"], 0.0) for d in sql_docs}
    logger.info("[SQL] finished → docs=%d, scored=%d", len(sql_docs), len(sql_scores))
    return sql_docs, sql_scores, sql_by_id


# Convenience wrapper (kept from your original)
def get_sql_master_data(md_db_path: str | Path, question: str, k: int = 20):
    return run_sql_retrieval(md_db_path=md_db_path, question=question, k=k)

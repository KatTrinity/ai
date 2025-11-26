# --- DB bootstrap ---
from pathlib import Path
import os, sys
import sqlite3, logging
from config_base.config import Config

# ✅ Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logger = logging.getLogger(__name__)

DB_PATH = Path(getattr(Config, "DB_PATH", Path("database") / "chat_logs.sqlite"))

# ----------------------- sql helpers --------------------------------------
PRAGMAS = (
    "PRAGMA foreign_keys=ON;",
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;", 
)

def _exec_block(conn: sqlite3.Connection, name: str, sql: str) -> None:
    block = (sql or "").strip()
    if not block:
        logging.debug(f"[DDL] {name}: empty, skipping")
        return
    try:
        conn.executescript(block)
        logger.info("✅ %s applied", name)
    except sqlite3.OperationalError as e:
        # bisect to the exact failing statement
        stmts = [s.strip() for s in block.split(";")]
        for i, st in enumerate(stmts, 1):
            if not st:
                continue
            try:
                conn.executescript(st + ";")
            except sqlite3.OperationalError as e2:
                logging.error(f"[DDL] {name} statement #{i} failed: {e2}\nSQL >>> {st};")
                raise
        raise

def ensure_db(conn: sqlite3.Connection) -> None:
    for p in PRAGMAS:
        conn.execute(p)
    conn.execute("BEGIN IMMEDIATE;")
    try:
        # 1) CORE (parents → children)
        _exec_block(conn, "CORE", DDL_CORE)

        # 2) CHAT (everything EXCEPT log_chat_sources)
        _exec_block(conn, "CHAT_BASE", DDL_CHAT_BASE)

        # 3) FTS last (it references doc_chunks)
        _exec_block(conn, "FTS", DDL_FTS)

        # 4) Finally, create the child table that FK-references parents
        _exec_block(conn, "CHAT_SOURCES", DDL_CHAT_SOURCES)

        # 5) 
        _exec_block(conn, "CHAT_VIEWS", DDL_VIEWS)

        # 6) User account and chat history
        _exec_block(conn, "USER_HISTORY", DDL_USERS)

        # 7) Doc Metrics
        _exec_block(conn, "DOC_METRICS", DDL_DOC_FRESHNESS)

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    # quick sanity
    rows = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name IN ('doc_ingest','doc_chunks','log_chat_turns','log_chat_sources')
        ORDER BY name
    """).fetchall()
    #logger.info("Tables present: %s", [r[0] for r in rows])

# Keep your get_conn as-is (just calls ensure_db)
def get_conn(db_path: str | Path | None = None, *, ensure: bool = True) -> sqlite3.Connection:
    p = Path(db_path) if db_path else DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Connecting to SQLite DB at: {p}")
    conn = sqlite3.connect(p, check_same_thread=False)
    for pr in PRAGMAS:
        conn.execute(pr)
    if ensure:
        ensure_db(conn)
    return conn 

DDL_CORE = """

-- Files discovered & canonicalized 
CREATE TABLE IF NOT EXISTS doc_ingest (
  doc_id           INTEGER PRIMARY KEY AUTOINCREMENT,

  -- File identity
  file_name        TEXT        NOT NULL,
  file_path        TEXT        NOT NULL UNIQUE,  -- absolute, normalized
  folder_resource  TEXT,                         -- optional logical bucket
  file_size_bytes  INTEGER,                      -- optional: for change detection
  file_mtime_ns    INTEGER,                      -- optional: precise mtime (ns)

  -- Hashes for idempotence
  file_hash        TEXT        NOT NULL,         -- full-file hash (frontmatter + body)
  fm_hash          TEXT,                         -- hash of canonical FM JSON
  body_hash        TEXT,                         -- hash of cleaned body (after noindex strip)

  ------------------------------------------------------------------
  -- Canonicalized front-matter (promoted fields + raw JSON snapshot)
  ------------------------------------------------------------------

  -- From frontmatter: doc_id (SAP Note/KBA ID)
  sap_doc_id       TEXT        NOT NULL DEFAULT '',   -- e.g. "540392"

  -- From frontmatter: doc_title
  title            TEXT        NOT NULL DEFAULT '',   -- display/canonical title

  -- Language + coarse classification
  language          TEXT        NOT NULL DEFAULT 'EN', -- e.g. "EN"
  doc_type          TEXT        NOT NULL DEFAULT '',   -- SAP Note, KBA, FAQ
  doc_category      TEXT        NOT NULL DEFAULT '',   -- FAQ, How-To, Troubleshooting
  doc_utility_score REAL,                         --- how useful this doc has been historically

  -- Component + parsed hierarchy
  component                TEXT        NOT NULL DEFAULT '',  -- e.g. "PP-SFC-EXE-GM"
  component_area           TEXT        NOT NULL DEFAULT '',  -- e.g. "PP"
  component_sub_area       TEXT        NOT NULL DEFAULT '',  -- e.g. "SFC"
  component_process        TEXT        NOT NULL DEFAULT '',  -- e.g. "EXE"
  component_sub_process    TEXT        NOT NULL DEFAULT '',  -- e.g. "GM"

  -- Versioning + dates (using your date_* preference)
  version                  INTEGER,
  version_previous         INTEGER,
  date_published           TEXT,                          -- ISO string "YYYY-MM-DD"
  date_previous_published  TEXT,                          -- ISO string or NULL
  date_download            TEXT,                          -- when *you* grabbed it
  date_freshness_score     REAL DEFAULT 0.0,
  date_ingested_iso        TEXT, 
  is_outdated_flag         INTEGER,

  -- LLM/human enrichment from frontmatter
  summary          TEXT        NOT NULL DEFAULT '',       -- short technical summary
  change_summary   TEXT        NOT NULL DEFAULT '',       -- version diff summary (human/LLM)

  -- High-signal LLM fields for faceting/search
  keyword_tags     TEXT        NOT NULL DEFAULT '[]',     -- JSON array of tags
  codes            TEXT        NOT NULL DEFAULT '[]',     -- JSON array of codes (t-codes, msg IDs, tables)

  -- Full canonical FM as JSON (for lineage/debugging)
  fm_json          TEXT        NOT NULL DEFAULT '{}',

  ----------------------------------------------------
  -- Canonicalized body (post noindex, ready for NLP)
  ----------------------------------------------------
  body_md                TEXT        NOT NULL DEFAULT '',       -- cleaned body markdown
  body_word_count        INTEGER     DEFAULT 0,
  noindex_removed        INTEGER     DEFAULT 0,                 -- 0/1 flag if noindex stripped
  has_contradiction_flag INTEGER,
  text_quality_score     REAL,

  -- Timestamps
  created_at       TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
  updated_at       TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,


  -- (Optional) JSON validity checks (SQLite json1)

  CHECK (json_valid(keyword_tags)),
  CHECK (json_valid(codes)),
  CHECK (json_valid(fm_json))
  
);

  -- identity / change detection
  CREATE UNIQUE INDEX IF NOT EXISTS idx_ingest_path      ON doc_ingest(file_path);
  CREATE INDEX IF NOT EXISTS idx_ingest_filehash         ON doc_ingest(file_hash);

  -- SAP-level identity
  CREATE INDEX IF NOT EXISTS idx_ingest_sap_doc_id       ON doc_ingest(sap_doc_id);

  -- common filters / facets
  CREATE INDEX IF NOT EXISTS idx_ingest_component        ON doc_ingest(component);
  CREATE INDEX IF NOT EXISTS idx_ingest_doc_type         ON doc_ingest(doc_type);
  CREATE INDEX IF NOT EXISTS idx_ingest_doc_category     ON doc_ingest(doc_category);
  CREATE INDEX IF NOT EXISTS idx_ingest_language         ON doc_ingest(language);

  -- title lookups
  CREATE INDEX IF NOT EXISTS idx_ingest_title            ON doc_ingest(title);

  -- Keep updated_at fresh
  CREATE TRIGGER IF NOT EXISTS trg_ingest
  AFTER UPDATE ON doc_ingest
  FOR EACH ROW
  BEGIN
    UPDATE doc_ingest
    SET updated_at = CURRENT_TIMESTAMP
    WHERE doc_id = NEW.doc_id;
  END;

-- Per-model embedding bookkeeping (used later by the builder)
CREATE TABLE IF NOT EXISTS doc_embedding_state (
  doc_id        INTEGER NOT NULL,
  model_key     TEXT    NOT NULL,
  content_hash  TEXT    NOT NULL,
  embedded_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (doc_id, model_key),
  FOREIGN KEY (doc_id) REFERENCES doc_ingest(doc_id) ON DELETE CASCADE
);

-- chunk_embedding_state: add FK + helpful indexes
CREATE TABLE IF NOT EXISTS chunk_embedding_state (
  chunk_id     TEXT NOT NULL,
  model_key    TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  embedded_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (chunk_id, model_key),
  FOREIGN KEY (chunk_id) REFERENCES doc_chunks(chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunk_state_model   ON chunk_embedding_state(model_key);
CREATE INDEX IF NOT EXISTS idx_chunk_state_content ON chunk_embedding_state(content_hash);

-- Optional: section layer for H1/H2/H3 grouping (broad recall)
CREATE TABLE IF NOT EXISTS doc_sections (
  section_id   TEXT PRIMARY KEY,      -- e.g., "{doc_id}:S0003"
  doc_id       INTEGER NOT NULL,
  header_path  TEXT NOT NULL,         -- "H1 > H2 > H3"
  span_start   INTEGER NOT NULL,      -- char offset in canonical body
  span_end     INTEGER NOT NULL,
  updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (doc_id) REFERENCES doc_ingest(doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON doc_sections(doc_id);

-- Micro-chunks (raw text lives here)
CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id            TEXT PRIMARY KEY,   -- e.g., "{doc_id}:S0003:C0012"
  doc_id              INTEGER NOT NULL,
  title               TEXT,
  section_id          TEXT,               -- nullable if you skip section layer
  header_path         TEXT NOT NULL,
  body_raw            TEXT NOT NULL,      -- canonical raw chunk text
  chunk_hash_raw      TEXT NOT NULL,      -- SHA-256 over header+body_raw
  chunk_quality_score REAL, 
  is_near_duplicate   INTEGER, 
  has_symptom_resolution_pair INTEGER,
  token_count                 INTEGER,            -- optional
  updated_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (doc_id)    REFERENCES doc_ingest(doc_id) ON DELETE CASCADE,
  FOREIGN KEY (section_id) REFERENCES doc_sections(section_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON doc_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON doc_chunks(chunk_hash_raw);

-- spaCy cache: keep your naming, but ensure a compound index is present
CREATE TABLE IF NOT EXISTS doc_nlp_cache (
  chunk_id     TEXT NOT NULL,
  nlp_version  TEXT NOT NULL,
  text_hash    TEXT NOT NULL,
  cleaned_text TEXT NOT NULL,
  pos_json     TEXT,
  updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (chunk_id, nlp_version),
  FOREIGN KEY (chunk_id) REFERENCES doc_chunks(chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_nlp_texthash ON doc_nlp_cache(text_hash);
-- (optional) queries by version benefit from this too:
CREATE INDEX IF NOT EXISTS idx_nlp_version ON doc_nlp_cache(nlp_version);

-- Canonicalized body after FM strip (one row per doc)
CREATE TABLE IF NOT EXISTS doc_canonical (
  doc_id         INTEGER PRIMARY KEY,
  canonical_body TEXT NOT NULL,
  updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (doc_id) REFERENCES doc_ingest(doc_id) ON DELETE CASCADE
);
"""

DDL_CHAT_BASE = """
-- One row per Q/A turn
CREATE TABLE IF NOT EXISTS log_chat_turns (
  turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id     TEXT NOT NULL,
  question       TEXT NOT NULL,
  response       TEXT,
  prompt_hash    TEXT,
  run_meta       TEXT,

  -- model selections (use *key naming consistently)
  llm_key        TEXT,
  embed_key      TEXT,
  ce_key         TEXT,
  ce_model       TEXT,

  -- retrieval settings & timings
  k_dense        INTEGER,
  k_rerank       INTEGER,
  corpus_n       INTEGER,
  pct_used       REAL,
  retrieval_sanity_flag INTEGER,
  keep_top       INTEGER,
  per_doc_cap    INTEGER,
  collection     TEXT,
  latency_ms     INTEGER DEFAULT 0,
  dense_ms       INTEGER DEFAULT 0,
  ce_ms          INTEGER DEFAULT 0,
  ce_used        INTEGER,

  prefiltered    INTEGER NOT NULL DEFAULT 0 CHECK (prefiltered IN (0,1)),
  similarity_avg REAL,
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_turns_session_time ON log_chat_turns(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_turns_prefiltered  ON log_chat_turns(prefiltered);

CREATE TABLE IF NOT EXISTS eval_targets (
  turn_id    INTEGER PRIMARY KEY REFERENCES log_chat_turns(turn_id) ON DELETE CASCADE,
  target     TEXT NOT NULL,
  dataset    TEXT,
  added_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS log_prompts (
  prompt_hash   TEXT PRIMARY KEY,
  system_prompt TEXT,
  prompt_text   TEXT,
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
);

-- one row per rater (session) per turn
CREATE TABLE IF NOT EXISTS log_chat_ratings (
  turn_id     INTEGER NOT NULL,
  session_id  TEXT    NOT NULL,
  rate_stars  INTEGER NOT NULL CHECK (rate_stars BETWEEN 1 AND 5),
  rated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (turn_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_ratings_turn        ON log_chat_ratings(turn_id);
CREATE INDEX IF NOT EXISTS idx_ratings_rated_at    ON log_chat_ratings(rated_at);
CREATE INDEX IF NOT EXISTS idx_turns_turn          ON log_chat_turns(turn_id);
CREATE INDEX IF NOT EXISTS idx_turns_model_created ON log_chat_turns(llm_key, created_at);
"""

DDL_VIEWS = """
-- Convenience views to search arrays (SQLite json1) 
CREATE VIEW IF NOT EXISTS v_doc_tags AS
  SELECT d.doc_id, lower(json_each.value) AS tag
  FROM doc_ingest d, json_each(d.keyword_tags);

CREATE VIEW IF NOT EXISTS v_doc_categories AS
  SELECT d.doc_id, lower(json_each.value) AS category
  FROM doc_ingest d, json_each(d.category);

CREATE VIEW IF NOT EXISTS v_doc_codes AS
  SELECT d.doc_id, upper(json_each.value) AS tcode
  FROM doc_ingest d, json_each(d.codes);

CREATE VIEW IF NOT EXISTS v_doc_error_codes AS
  SELECT d.doc_id, upper(json_each.value) AS error_code
  FROM doc_ingest d, json_each(d.error_codes);

  -- Base join: one row per (rated) turn x rater
CREATE VIEW IF NOT EXISTS vw_chat_ratings AS
SELECT
  r.turn_id,
  r.session_id              AS rater_session_id,
  r.rate_stars,
  r.rated_at,

  t.session_id              AS turn_session_id,
  t.created_at,
  t.llm_key,
  t.embed_key,
  t.latency_ms,
  t.similarity_avg
FROM log_chat_ratings r
JOIN log_chat_turns  t ON t.turn_id = r.turn_id;

-- Pre-agg sums so Python can compute std fast
CREATE VIEW IF NOT EXISTS vw_ratings_by_model_sums AS
SELECT
  llm_key,
  COUNT(*)                AS n,
  AVG(rate_stars)         AS mean_stars,
  SUM(rate_stars)         AS sum_stars,
  SUM(rate_stars*rate_stars) AS sumsq_stars,
  MIN(rated_at)           AS first_rating_at,
  MAX(rated_at)           AS last_rating_at
FROM vw_chat_ratings_enriched
GROUP BY llm_key;

CREATE VIEW IF NOT EXISTS vw_fusion_sources AS
SELECT
  s.turn_id,
  s.rank,
  s.doc_id,
  s.chunk_id,
  s.used_in_prompt,

  -- scalar scores
  s.ce_score,
  s.dense_score AS dense_score_logged,

  -- score_json components
  CAST(json_extract(s.score_json, '$.dense') AS REAL)  AS dense_score,
  CAST(json_extract(s.score_json, '$.fts')   AS REAL)  AS fts_score,
  CAST(json_extract(s.score_json, '$.sql')   AS REAL)  AS sql_score,
  CAST(json_extract(s.score_json, '$.fused') AS REAL)  AS fused_score,

  json_extract(s.score_json, '$.strategy')           AS fusion_strategy,
  json_extract(s.score_json, '$.rrf_k')              AS rrf_k,
  json_extract(s.score_json, '$.w.dense')            AS w_dense,
  json_extract(s.score_json, '$.w.fts')              AS w_fts,
  json_extract(s.score_json, '$.w.sql')              AS w_sql,

  -- join to turn metadata
  t.session_id,
  t.question,
  t.response,
  t.llm_key,
  t.embed_key,
  t.ce_key,
  t.latency_ms,
  t.dense_ms,
  t.ce_ms,
  t.similarity_avg,
  t.prefiltered,
  t.created_at
FROM log_chat_sources s
JOIN log_chat_turns  t
  ON t.turn_id = s.turn_id;

CREATE VIEW IF NOT EXISTS vw_fusion_by_turn AS
SELECT
  turn_id,
  llm_key,
  embed_key,

  COUNT(*) AS total_sources,
  SUM(CASE WHEN dense_score > 0 THEN 1 ELSE 0 END) AS dense_hits,
  SUM(CASE WHEN fts_score   > 0 THEN 1 ELSE 0 END) AS fts_hits,
  SUM(CASE WHEN sql_score   > 0 THEN 1 ELSE 0 END) AS sql_hits,

  -- overlap buckets
  SUM(CASE WHEN dense_score > 0 AND fts_score = 0 AND sql_score = 0 THEN 1 ELSE 0 END) AS dense_only,
  SUM(CASE WHEN dense_score = 0 AND fts_score > 0 AND sql_score = 0 THEN 1 ELSE 0 END) AS fts_only,
  SUM(CASE WHEN dense_score = 0 AND fts_score = 0 AND sql_score > 0 THEN 1 ELSE 0 END) AS sql_only,

  SUM(CASE WHEN dense_score > 0 AND fts_score > 0 AND sql_score = 0 THEN 1 ELSE 0 END) AS dense_fts_overlap,
  SUM(CASE WHEN dense_score > 0 AND fts_score = 0 AND sql_score > 0 THEN 1 ELSE 0 END) AS dense_sql_overlap,
  SUM(CASE WHEN dense_score = 0 AND fts_score > 0 AND sql_score > 0 THEN 1 ELSE 0 END) AS fts_sql_overlap,
  SUM(CASE WHEN dense_score > 0 AND fts_score > 0 AND sql_score > 0 THEN 1 ELSE 0 END) AS all_three,

  -- restrict to what actually went into the prompt
  SUM(CASE WHEN used_in_prompt=1 THEN 1 ELSE 0 END) AS used_in_prompt
FROM vw_fusion_sources
GROUP BY turn_id, llm_key, embed_key;


-- Average CE scores per turn
DROP VIEW IF EXISTS vw_ce_by_turn;

CREATE VIEW IF NOT EXISTS vw_ce_by_turn AS
SELECT
  turn_id,
  AVG(ce_score) AS avg_ce_score,
  AVG(CASE WHEN used_in_prompt = 1 THEN ce_score END) AS avg_ce_used
FROM log_chat_sources
GROUP BY turn_id;

CREATE VIEW IF NOT EXISTS stale_docs_for_review AS
SELECT
  i.doc_id,
  i.title,
  i.component,
  f.last_pub_date,
  f.age_days,
  f.bucket,
  f.refreshed_at
FROM doc_ingest i
JOIN doc_freshness_metrics f
  ON f.doc_id = i.doc_id
WHERE f.bucket IN ('stale_1_3y', 'legacy_3y_plus')
ORDER BY f.age_days DESC;  -- oldest at top

CREATE TABLE IF NOT EXISTS system_maintenance (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

"""

DDL_CHAT_SOURCES = """
-- One row per cited chunk for that turn (create LAST)
CREATE TABLE IF NOT EXISTS log_chat_sources (
  turn_id        INTEGER NOT NULL,
  rank           INTEGER NOT NULL,        -- 1..N after rerank
  doc_id         INTEGER,                 -- from doc_ingest
  chunk_id       TEXT,                    -- "{doc_id}:Sxxx:Cxxxx"
  section_id     TEXT,                    -- "{doc_id}:Sxxx"
  file_path      TEXT,
  title          TEXT,
  header_path    TEXT,
  span_start     INTEGER,
  span_end       INTEGER,
  ce_score       REAL,
  dense_score    REAL,
  used_in_prompt INTEGER NOT NULL DEFAULT 1,
  score_json     TEXT,
  utility_delta  REAL,
  used_in_final_answer INTEGER,

  PRIMARY KEY (turn_id, rank),
  FOREIGN KEY (turn_id)  REFERENCES log_chat_turns(turn_id) ON DELETE CASCADE,
  FOREIGN KEY (doc_id)   REFERENCES doc_ingest(doc_id)      ON DELETE SET NULL,
  FOREIGN KEY (chunk_id) REFERENCES doc_chunks(chunk_id)    ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_doc ON log_chat_sources(doc_id);
"""

DDL_FTS = """ 
-- SQLite FTS5 mirror over RAW chunks (keyword/BM25 path)
CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
  title,
  header_path,
  body_raw,
  chunk_id UNINDEXED,
  doc_id   UNINDEXED,
  tokenize='porter'
);

-- Change tracking for FTS (because FTS5 has no ON CONFLICT)
CREATE TABLE IF NOT EXISTS doc_chunks_fts_state (
  chunk_id   TEXT PRIMARY KEY,
  chunk_hash TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (chunk_id) REFERENCES doc_chunks(chunk_id) ON DELETE CASCADE
);
"""

DDL_USERS = """

-- Application users (for accounts & memory scoping)
CREATE TABLE IF NOT EXISTS users (
  user_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  username     TEXT NOT NULL UNIQUE,          -- e.g. "kat", "admin"
  display_name TEXT,                          -- pretty name for UI
  role         TEXT NOT NULL DEFAULT 'user',  -- 'admin' or 'user'
  is_active    INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
  created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER IF NOT EXISTS trg_users_updated
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
  UPDATE users
  SET updated_at = CURRENT_TIMESTAMP
  WHERE user_id = NEW.user_id;
END;

-- Long-term user memory entries (Phase 1)
CREATE TABLE IF NOT EXISTS user_memory (
  memory_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id      INTEGER NOT NULL,              -- FK to users.user_id
  agent_scope  TEXT NOT NULL DEFAULT 'global',-- e.g. 'global', 'governedge', 'dqscan'
  memory_type  TEXT NOT NULL DEFAULT 'fact',  -- 'fact','pref','identity','task'
  label        TEXT,                          -- short label for UI
  content      TEXT NOT NULL,                 -- the actual memory text
  importance   INTEGER NOT NULL DEFAULT 1,    -- 1–5 weight
  is_active    INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
  source       TEXT,                          -- 'manual','agent','import'
  vector_id    TEXT,                          -- optional: ID in Chroma/DuckDB
  last_used_at TIMESTAMP,
  created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_memory_user       ON user_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_user_memory_active     ON user_memory(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_user_memory_scope_type ON user_memory(agent_scope, memory_type);

CREATE TRIGGER IF NOT EXISTS trg_user_memory_updated
AFTER UPDATE ON user_memory
FOR EACH ROW
BEGIN
  UPDATE user_memory
  SET updated_at = CURRENT_TIMESTAMP
  WHERE memory_id = NEW.memory_id;
END;

-- Map Streamlit/LightAgent session IDs to users
CREATE TABLE IF NOT EXISTS user_sessions (
  session_id  TEXT PRIMARY KEY,          -- same string you're already using
  user_id     INTEGER NOT NULL,
  started_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id);

CREATE TRIGGER IF NOT EXISTS trg_user_sessions_touch
AFTER UPDATE ON user_sessions
FOR EACH ROW
BEGIN
  UPDATE user_sessions
  SET last_seen_at = CURRENT_TIMESTAMP
  WHERE session_id = NEW.session_id;
END;

"""
DDL_DOC_FRESHNESS = """ 

CREATE TABLE IF NOT EXISTS doc_freshness_metrics (
  doc_id        INTEGER PRIMARY KEY
                REFERENCES doc_ingest(doc_id) ON DELETE CASCADE,
  last_pub_date TEXT NOT NULL,
  age_days      INTEGER NOT NULL,
  bucket        TEXT NOT NULL,
  refreshed_at  TEXT NOT NULL
);

"""


if __name__ == "__main__":
    import argparse
    #from pathlib import Path
    #from config_base.config import Config

    # to run independantly
    # (.venv) PS C:\dev\GovernEdge_CLI> python -m prepare_docs.db_io --db C:\dev\GovernEdge_CLI\database\chat_logs.sqlite

    ap = argparse.ArgumentParser(description="Bootstrap GovernEdge DB schema")
    ap.add_argument("--db", required=True, help="Path to SQLite DB file")
    ap.add_argument("--reset", action="store_true", help="Delete and recreate the DB file")
    args = ap.parse_args()

    db_path = Path(args.db)

    if args.reset:
        # nuke main + WAL + SHM if present
        for ext in ("", "-wal", "-shm"):
            f = Path(str(db_path) + ext)
            if f.exists():
                f.unlink()
                print(f"🗑️ Removed {f}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        ensure_db(conn)

    print(f"✅ Database schema ensured at {db_path}")






PRAGMAS = (
    "PRAGMA foreign_keys=ON;",
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;",
)
DDL_CORE = """
BEGIN;
DROP TABLE IF EXISTS doc_chunks_fts;
DROP TABLE IF EXISTS doc_chunks_fts_state;

CREATE VIRTUAL TABLE doc_chunks_fts USING fts5(
  title,
  header_path,
  body_raw,
  chunk_id UNINDEXED,
  doc_id   UNINDEXED,
  tokenize='porter'
);

CREATE TABLE doc_chunks_fts_state (
  chunk_id   TEXT PRIMARY KEY,
  chunk_hash TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO doc_chunks_fts (title, header_path, body_raw, chunk_id, doc_id)
SELECT title, header_path, body_raw, chunk_id, doc_id FROM doc_chunks;
COMMIT;
"""
def ensure_db(conn):
    for p in PRAGMAS:
        conn.execute(p)
    conn.executescript(DDL_CORE)
    conn.commit()

if __name__ == "__main__":
    import argparse, sqlite3
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Bootstrap GovernEdge DB schema")
    ap.add_argument("--db", required=True, help="Path to SQLite DB file")
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        ensure_db(conn)
        print(f"✅ Database schema ensured at {db_path}")
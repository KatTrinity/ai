
# What “good” looks like
# journal_mode=WAL, foreign_keys=1
# doc_ingest ≈ number of source files
# doc_canonical equals doc_ingest
# doc_chunks > 0
# doc_chunks_fts equals doc_chunks (or close if you filtered)
# “Chunks missing in FTS: 0”
# “FTS state mismatches: 0”
# FTS search returns a few snippets for your test term

import sqlite3, sys
from pathlib import Path

db = Path(sys.argv[1] if len(sys.argv) > 1 else r".\database\chat_logs.sqlite")

q = lambda c, sql, *a: c.execute(sql, a).fetchone()[0]
with sqlite3.connect(str(db)) as conn:
    conn.row_factory = sqlite3.Row

    # PRAGMAs
    conn.execute("PRAGMA foreign_keys=ON;") 
    jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
    fk = q(conn, "PRAGMA foreign_keys")
    print(f"PRAGMA journal_mode={jm}, foreign_keys={fk}")

    # Table counts
    counts = {
        "doc_ingest": q(conn, "SELECT COUNT(*) FROM doc_ingest"),
        "doc_canonical":  q(conn, "SELECT COUNT(*) FROM doc_canonical"),
        "doc_chunks":     q(conn, "SELECT COUNT(*) FROM doc_chunks"),
        "doc_chunks_fts": q(conn, "SELECT COUNT(*) FROM doc_chunks_fts"),
        "fts_state":      q(conn, "SELECT COUNT(*) FROM doc_chunks_fts_state"),
        "nlp_cache":      q(conn, "SELECT COUNT(*) FROM doc_nlp_cache"),
    }
    print("Row counts:", counts)

    # 1:1 checks
    missing_canon = q(conn, """
        SELECT COUNT(*) FROM doc_ingest d
        LEFT JOIN doc_canonical c USING(doc_id)
        WHERE c.doc_id IS NULL
    """)
    print("Missing canonical bodies:", missing_canon)

    # Any chunks that did NOT make it to FTS?
    missing_fts = q(conn, """
        SELECT COUNT(*) FROM doc_chunks c
        LEFT JOIN doc_chunks_fts f ON f.chunk_id = c.chunk_id
        WHERE f.chunk_id IS NULL
    """)
    print("Chunks missing in FTS:", missing_fts)

    # FTS state hash mismatches (should be 0)
    mismatch = q(conn, """
        SELECT COUNT(*) FROM doc_chunks c
        LEFT JOIN doc_chunks_fts_state s ON s.chunk_id = c.chunk_id
        WHERE s.chunk_hash IS NULL OR s.chunk_hash != c.chunk_hash_raw
    """)
    print("FTS state mismatches:", mismatch)

    # Peek a few chunks
    for r in conn.execute("""
        SELECT d.doc_id, d.file_name, COALESCE(d.title, '') AS title, c.chunk_id,
               substr(c.body_raw,1,80) AS preview
        FROM doc_chunks c JOIN doc_ingest d ON d.doc_id = c.doc_id
        ORDER BY d.doc_id, c.chunk_id LIMIT 5
    """):
        print(dict(r))

    # Optional: quick FTS match (edit the term if you like)
    term = sys.argv[2] if len(sys.argv) > 2 else None
    if term:
        print(f"\nFTS search for: {term!r}")
        for r in conn.execute("""
            SELECT title,
                   snippet(doc_chunks_fts, 2, '[', ']', ' … ', 10) AS snip,
                   doc_id, chunk_id
            FROM doc_chunks_fts
            WHERE doc_chunks_fts MATCH ?
            LIMIT 5
        """, (term,)):
            print(dict(r))

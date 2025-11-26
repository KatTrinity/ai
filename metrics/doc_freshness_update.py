# metrics/freshness_maintenance.py
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from prepare_docs.db_io import get_conn, DB_PATH as DEFAULT_DB_PATH

FRESHNESS_KEY = "freshness_metrics_last_run"

def _ensure_tables(conn: sqlite3.Connection) -> None:
    # doc_freshness_metrics is already created by ensure_db via DDL_DOC_FRESHNESS
    # so we only need system_maintenance here (unless you move that into db_io too)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS system_maintenance (
      key        TEXT PRIMARY KEY,
      value      TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    """)

def _refresh_doc_freshness_metrics(conn: sqlite3.Connection) -> None:
    conn.execute("""
    INSERT INTO doc_freshness_metrics (
      doc_id, last_pub_date, age_days, bucket, refreshed_at
    )
    SELECT
      d.doc_id,
      d.date_published AS last_pub_date,
      CAST(julianday('now') - julianday(d.date_published) AS INTEGER) AS age_days,
      CASE
        WHEN (julianday('now') - julianday(d.date_published)) <=  90      THEN 'new_90_days'
        WHEN (julianday('now') - julianday(d.date_published)) <= 365      THEN 'fresh_365_days'
        WHEN (julianday('now') - julianday(d.date_published)) <= 3*365    THEN 'stale_3y'
        ELSE 'legacy_3y_plus'
      END AS bucket,
      datetime('now') AS refreshed_at
    FROM doc_ingest d
    WHERE d.date_published IS NOT NULL
    ON CONFLICT(doc_id) DO UPDATE SET
      last_pub_date = excluded.last_pub_date,
      age_days      = excluded.age_days,
      bucket        = excluded.bucket,
      refreshed_at  = excluded.refreshed_at;
    """)
    conn.execute("""
    INSERT INTO system_maintenance(key, value, updated_at)
    VALUES (?, datetime('now'), datetime('now'))
    ON CONFLICT(key) DO UPDATE SET
      value      = excluded.value,
      updated_at = excluded.updated_at;
    """, (FRESHNESS_KEY,))

def chk_freshness_metrics(
    db_path: str | Path | None = None,
    max_age_hours: int = 24,
) -> None:
    dbp = str(db_path or DEFAULT_DB_PATH)

    # ensure_db() is invoked here through get_conn(..., ensure=True),
    # so doc_freshness_metrics already exists.
    with get_conn(dbp, ensure=True) as conn:
        _ensure_tables(conn)  # only ensures system_maintenance now

        cur = conn.execute(
            "SELECT value FROM system_maintenance WHERE key = ?",
            (FRESHNESS_KEY,),
        )
        row = cur.fetchone()
        if row is None:
            _refresh_doc_freshness_metrics(conn)
            return

        last_run_str = row[0]
        try:
            last_run = datetime.fromisoformat(last_run_str)
        except Exception:
            _refresh_doc_freshness_metrics(conn)
            return

        if datetime.utcnow() - last_run >= timedelta(hours=max_age_hours):
            _refresh_doc_freshness_metrics(conn)
        # else: no-op

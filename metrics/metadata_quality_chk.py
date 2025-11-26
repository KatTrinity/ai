

###################################

# place holder sample. this means nothing inherently 

####################################

# metrics/metadata_metrics.py
from pathlib import Path
import sqlite3
import json

import streamlit as st

from prepare_docs.db_io import get_conn, DB_PATH


def _fetch_one(conn, sql, params=()):
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else 0


def render_metadata_metrics(db_path: Path | str = DB_PATH) -> None:
    st.markdown("### Metadata Health & Drift")

    with get_conn(db_path) as conn:
        # ---- Doc-level aggregates ----
        total_docs = _fetch_one(conn, "SELECT COUNT(*) FROM doc_ingest")
        docs_with_canonical = _fetch_one(
            conn,
            "SELECT COUNT(*) FROM doc_canonical WHERE canonical_body IS NOT NULL"
        )
        docs_missing_title = _fetch_one(
            conn,
            "SELECT COUNT(*) FROM doc_ingest WHERE title IS NULL OR trim(title) = ''"
        )
        docs_trivial_fm = _fetch_one(
            conn,
            "SELECT COUNT(*) FROM doc_ingest WHERE fm_json = '{}' OR fm_json IS NULL"
        )
        docs_tiny_body = _fetch_one(
            conn,
            "SELECT COUNT(*) FROM doc_ingest WHERE body_word_count IS NOT NULL AND body_word_count < 50"
        )

        # ---- Chunk-level aggregates ----
        total_chunks = _fetch_one(conn, "SELECT COUNT(*) FROM doc_chunks")
        chunks_missing_section = _fetch_one(
            conn,
            "SELECT COUNT(*) FROM doc_chunks WHERE section_id IS NULL"
        )
        low_quality_chunks = _fetch_one(
            conn,
            "SELECT COUNT(*) FROM doc_chunks WHERE chunk_quality_score IS NOT NULL AND chunk_quality_score < 0.4"
        )
        near_dups = _fetch_one(
            conn,
            "SELECT COUNT(*) FROM doc_chunks WHERE is_near_duplicate = 1"
        )
        sr_chunks = _fetch_one(
            conn,
            "SELECT COUNT(*) FROM doc_chunks WHERE has_symptom_resolution_pair = 1"
        )

    # ---- Top-level summary ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Docs total", total_docs)
    col2.metric("Docs with canonical body", docs_with_canonical)
    col3.metric("Docs missing title", docs_missing_title)

    col4, col5, col6 = st.columns(3)
    col4.metric("Docs w/ trivial FM", docs_trivial_fm)
    col5.metric("Docs w/ tiny body (<50 words)", docs_tiny_body)
    col6.metric("Total chunks", total_chunks)

    col7, col8, col9 = st.columns(3)
    col7.metric("Chunks missing section_id", chunks_missing_section)
    col8.metric("Low quality chunks (score < 0.4)", low_quality_chunks)
    col9.metric("Near-duplicate chunks", near_dups)

    st.markdown("---")
    st.markdown("#### Symptom/Resolution chunks")
    st.write(f"Chunks detected with symptom+resolution pattern: **{sr_chunks}**")

    # You can add a table of the 'worst' docs/chunks later if you want.

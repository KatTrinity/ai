from pathlib import Path
import pandas as pd
import streamlit as st

from config_base.config import Config
from metrics.common import query_df

DB_PATH = Path(getattr(Config, "DB_PATH", Path("database") / "chat_logs.sqlite"))


# ---------- Helpers ------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_total_chunks() -> int:
    df = query_df("SELECT COUNT(*) AS n_chunks FROM doc_chunks")
    return int(df.iloc[0]["n_chunks"]) if not df.empty else 0


@st.cache_data(show_spinner=False)
def load_nlp_cache_stats():
    """
    Returns:
      - one-row stats DataFrame (global)
      - per-version coverage DataFrame
    """
    # global stats
    global_df = query_df(
        """
        SELECT
          (SELECT COUNT(*) FROM doc_chunks) AS total_chunks,
          COUNT(DISTINCT chunk_id) AS cached_chunks,
          SUM(
            CASE WHEN updated_at >= datetime('now','-7 day')
                 THEN 1 ELSE 0 END
          ) AS cached_recent_rows,
          COUNT(DISTINCT CASE
            WHEN updated_at >= datetime('now','-7 day')
            THEN chunk_id END
          ) AS cached_recent_chunks
        FROM doc_nlp_cache
        """
    )

    # per nlp_version coverage
    per_version_df = query_df(
        """
        SELECT
          nlp_version,
          COUNT(DISTINCT chunk_id) AS cached_chunks,
          MIN(updated_at) AS first_cached_at,
          MAX(updated_at) AS last_cached_at
        FROM doc_nlp_cache
        GROUP BY nlp_version
        ORDER BY last_cached_at DESC
        """,
        parse_dates=["first_cached_at", "last_cached_at"],
    )

    return global_df, per_version_df


@st.cache_data(show_spinner=False)
def load_fts_state_stats():
    """
    Returns one-row DataFrame with FTS state coverage + mismatches.
    """
    df = query_df(
        """
        SELECT
          -- chunk universe
          (SELECT COUNT(*) FROM doc_chunks) AS total_chunks,

          -- chunks with a state row
          (SELECT COUNT(*) FROM doc_chunks_fts_state) AS fts_state_rows,

          -- chunks whose FTS state was updated in last 7 days
          (SELECT COUNT(*) FROM doc_chunks_fts_state
           WHERE updated_at >= datetime('now','-7 day')) AS fts_recent_rows,

          -- chunks that exist but have NO FTS state row
          (SELECT COUNT(*) FROM doc_chunks c
           LEFT JOIN doc_chunks_fts_state s
             ON s.chunk_id = c.chunk_id
           WHERE s.chunk_id IS NULL) AS missing_fts_rows,

          -- FTS rows pointing at non-existent chunks (zombie state)
          (SELECT COUNT(*) FROM doc_chunks_fts_state s
           LEFT JOIN doc_chunks c
             ON c.chunk_id = s.chunk_id
           WHERE c.chunk_id IS NULL) AS orphan_fts_rows,

          -- FTS rows where hash is stale vs current chunk_hash_raw
          (SELECT COUNT(*) FROM doc_chunks_fts_state s
           JOIN doc_chunks c
             ON c.chunk_id = s.chunk_id
           WHERE s.chunk_hash <> c.chunk_hash_raw) AS stale_hash_rows
        """
    )
    return df


# ---------- Streamlit page -----------------------------------------------

def render_nlp_fts_validity():
    st.markdown("## 🧠 NLP & 🧾 FTS Cache Validity")

    total_chunks = load_total_chunks()
    if total_chunks == 0:
        st.info("No chunks found in doc_chunks yet.")
        return

    # ----- NLP cache coverage --------------------------------------------
    st.subheader("NLP cache coverage (doc_nlp_cache)")

    nlp_global_df, nlp_versions_df = load_nlp_cache_stats()
    if nlp_global_df.empty:
        st.info("doc_nlp_cache is empty – nothing cached yet.")
    else:
        row = nlp_global_df.iloc[0]
        total_chunks = int(row["total_chunks"])
        cached_chunks = int(row["cached_chunks"])
        cached_recent_rows = int(row["cached_recent_rows"]) if not pd.isna(row["cached_recent_rows"]) else 0
        cached_recent_chunks = int(row["cached_recent_chunks"]) if not pd.isna(row["cached_recent_chunks"]) else 0

        pct_cached = cached_chunks / total_chunks if total_chunks else 0.0
        pct_recent = cached_recent_chunks / total_chunks if total_chunks else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total chunks", f"{total_chunks}")
        c2.metric("Cached in NLP", f"{cached_chunks} ({pct_cached:.0%})")
        c3.metric("Chunks cached in last 7 days", f"{cached_recent_chunks} ({pct_recent:.0%})")

        st.write("Per NLP version coverage:")
        if not nlp_versions_df.empty:
            st.dataframe(nlp_versions_df)
        else:
            st.caption("No per-version stats yet (no rows in doc_nlp_cache).")

    # ----- FTS mirror state coverage -------------------------------------
    st.subheader("FTS mirror state coverage (doc_chunks_fts_state)")

    fts_df = load_fts_state_stats()
    if fts_df.empty:
        st.info("doc_chunks_fts_state is empty – FTS mirror hasn't run yet.")
        return

    fr = fts_df.iloc[0]
    total_chunks_fts = int(fr["total_chunks"])
    fts_state_rows = int(fr["fts_state_rows"])
    fts_recent_rows = int(fr["fts_recent_rows"])
    missing_fts_rows = int(fr["missing_fts_rows"])
    orphan_fts_rows = int(fr["orphan_fts_rows"])
    stale_hash_rows = int(fr["stale_hash_rows"])

    pct_fts_state = fts_state_rows / total_chunks_fts if total_chunks_fts else 0.0
    pct_recent_fts = fts_recent_rows / total_chunks_fts if total_chunks_fts else 0.0

    c4, c5, c6 = st.columns(3)
    c4.metric("Chunks with FTS state", f"{fts_state_rows} ({pct_fts_state:.0%})")
    c5.metric("FTS rows updated in last 7 days", f"{fts_recent_rows} ({pct_recent_fts:.0%})")
    c6.metric("Missing FTS rows", f"{missing_fts_rows}")

    c7, c8 = st.columns(2)
    c7.metric("Orphan FTS rows (no chunk)", f"{orphan_fts_rows}")
    c8.metric("Stale hash rows", f"{stale_hash_rows}")

    st.caption(
        "- **Missing FTS rows**: chunks in `doc_chunks` with no entry in `doc_chunks_fts_state`.\n"
        "- **Orphan FTS rows**: state rows whose `chunk_id` no longer exists.\n"
        "- **Stale hash rows**: `chunk_hash` in state != `chunk_hash_raw` in `doc_chunks` (needs re-mirror)."
    )


# Optional: run standalone
if __name__ == "__main__":
    render_nlp_fts_validity()

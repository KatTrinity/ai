from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from config_base.config import Config
from metrics.common import query_df

DB_PATH = Path(getattr(Config, "DB_PATH", Path("database") / "chat_logs.sqlite"))


# --- Helpers -------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_similarity_turns(limit: int = 1000) -> pd.DataFrame:
    """
    Per-turn similarity + avg CE scores.
    """
    df = query_df(
        """
        SELECT
          t.turn_id,
          t.session_id,
          t.question,
          t.response,
          t.similarity_avg,
          c.avg_ce_score,
          c.avg_ce_used,
          t.llm_key,
          t.embed_key,
          t.ce_key,
          t.latency_ms,
          t.ce_ms,
          t.created_at
        FROM log_chat_turns t
        LEFT JOIN vw_ce_by_turn c
          ON c.turn_id = t.turn_id
        ORDER BY t.created_at DESC
        LIMIT ?
        """,
        params=(limit,),
        parse_dates=["created_at"],
    )
    return df


@st.cache_data(show_spinner=False)
def load_ratings_with_similarity() -> pd.DataFrame:
    """
    One row per rating, with similarity + CE averages per turn.
    """
    df = query_df(
        """
        SELECT
          r.turn_id,
          r.rater_session_id,
          r.rate_stars,
          r.rated_at,
          r.similarity_avg,
          r.llm_key,
          r.embed_key,
          c.avg_ce_score,
          c.avg_ce_used
        FROM vw_chat_ratings r
        LEFT JOIN vw_ce_by_turn c
          ON c.turn_id = r.turn_id
        ORDER BY r.rated_at DESC
        """,
        parse_dates=["rated_at"],
    )
    return df


# --- Streamlit page ------------------------------------------------------

def render_similarity_quality():
    st.markdown("## 📏 Similarity & Quality Metrics")

    # ---------- Per-turn similarity & CE drift ----------
    st.subheader("Per-turn similarity & CE scores")

    turns_df = load_similarity_turns(limit=2000)
    if turns_df.empty:
        st.info("No chat turns logged yet.")
        return

    # Basic global stats
    global_avg_sim = turns_df["similarity_avg"].mean()
    global_avg_ce  = turns_df["avg_ce_score"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg similarity (all turns)", f"{global_avg_sim:.3f}")
    col2.metric("Avg CE score (all turns)", f"{global_avg_ce:.3f}")
    col3.metric("Total turns", len(turns_df))

    # Line-ish view over time
    st.write("Similarity and CE score over time (most recent first):")
    time_df = (
        turns_df[["created_at", "similarity_avg", "avg_ce_score"]]
        .sort_values("created_at")
        .rename(
            columns={
                "similarity_avg": "Similarity",
                "avg_ce_score": "Avg CE score",
            }
        )
    )
    st.line_chart(time_df.set_index("created_at"))

    # Optional: breakdown by model
    st.subheader("By model / embed combo")
    grouped = (
        turns_df.groupby(["llm_key", "embed_key"])
        .agg(
            n_turns=("turn_id", "nunique"),
            avg_similarity=("similarity_avg", "mean"),
            avg_ce_score=("avg_ce_score", "mean"),
        )
        .reset_index()
        .sort_values("n_turns", ascending=False)
    )
    st.dataframe(grouped)


    # ---------- Correlation: stars ⭐ vs similarity / CE ----------
    st.subheader("Correlation between ratings and similarity")

    ratings_df = load_ratings_with_similarity()
    if ratings_df.empty:
        st.info("No ratings yet, so no correlation to compute.")
        return

    # Drop rows missing scores
    corr_df = ratings_df.dropna(subset=["rate_stars", "similarity_avg", "avg_ce_score"])

    if len(corr_df) < 3:
        st.info("Not enough rated samples to compute meaningful correlations yet.")
        return

    corr_sim = corr_df["rate_stars"].corr(corr_df["similarity_avg"])
    corr_ce  = corr_df["rate_stars"].corr(corr_df["avg_ce_score"])

    c1, c2 = st.columns(2)
    c1.metric("Corr(stars, similarity)", f"{corr_sim:.3f}")
    c2.metric("Corr(stars, avg CE)",     f"{corr_ce:.3f}")

    st.write("Sample of rated turns:")
    st.dataframe(
        corr_df[
            [
                "turn_id",
                "rate_stars",
                "similarity_avg",
                "avg_ce_score",
                "llm_key",
                "embed_key",
            ]
        ].head(25)
    )


# Optional: standalone run
if __name__ == "__main__":
    render_similarity_quality()

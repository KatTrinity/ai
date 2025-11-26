from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st

from config_base.config import Config
from metrics.common import query_df  # assuming this defaults to DB_PATH inside

DB_PATH = Path(getattr(Config, "DB_PATH", Path("database") / "chat_logs.sqlite"))


# --- Helpers -------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_fusion_flat(limit: int = 5000) -> pd.DataFrame:
    """
    Load flattened fusion rows from the view.
    """
    # If query_df knows DB_PATH globally, you can use it here.
    # Otherwise, swap for raw sqlite3.connect(DB_PATH).
    df = query_df(
        """
        SELECT
          turn_id,
          session_id,
          question,
          response,
          rank,
          used_in_prompt,
          dense_score,
          fts_score,
          sql_score,
          fused_score,
          ce_score,
          fusion_strategy,
          w_dense,
          w_fts,
          w_sql,
          llm_key,
          embed_key,
          ce_key,
          created_at
        FROM vw_fusion_sources
        ORDER BY created_at DESC, rank ASC
        LIMIT ?
        """,
        params=(limit,),
        parse_dates=["created_at"],
    )
    return df


def add_weight_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add alternate fused scores for some what-if weight combos.
    """
    # make sure we have numeric columns
    for col in ("dense_score", "fts_score", "sql_score"):
        if col not in df.columns:
            df[col] = 0.0

    weight_sets = [
        (0.8, 0.15, 0.05),
        (0.4, 0.4, 0.2),
        (0.55, 0.30, 0.15),  # your current default, probably
    ]

    for w_dense, w_fts, w_sql in weight_sets:
        col_name = f"fused_test_{w_dense:.2f}_{w_fts:.2f}_{w_sql:.2f}"
        df[col_name] = (
            w_dense * df["dense_score"].fillna(0)
            + w_fts   * df["fts_score"].fillna(0)
            + w_sql   * df["sql_score"].fillna(0)
        )

    return df


# --- Streamlit page ------------------------------------------------------

def render_fusion_perform():
    st.markdown("## 🔀 Fusion Performance")

    df = load_fusion_flat(limit=5000)
    if df.empty:
        st.info("No fusion data yet. Run some queries through the engine first.")
        return

    df = add_weight_experiments(df)

    # --- Global coverage summary ----------------------------------------
    st.subheader("Source coverage (overall)")

    total_sources = len(df)
    dense_hits = (df["dense_score"] > 0).sum()
    fts_hits   = (df["fts_score"]   > 0).sum()
    sql_hits   = (df["sql_score"]   > 0).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total chunks", total_sources)
    col2.metric("Dense coverage", f"{dense_hits / total_sources:.0%}")
    col3.metric("FTS coverage",   f"{fts_hits   / total_sources:.0%}")
    col4.metric("SQL coverage",   f"{sql_hits   / total_sources:.0%}")

    # --- Per-turn dominance ---------------------------------------------
    st.subheader("Per-turn retriever dominance")

    # classify which retriever "dominates" a chunk
    def dominant(row):
        scores = {
            "dense": row["dense_score"] or 0.0,
            "fts":   row["fts_score"] or 0.0,
            "sql":   row["sql_score"] or 0.0,
        }
        # all zero?
        if all(v == 0 for v in scores.values()):
            return "none"
        return max(scores, key=scores.get)

    df["dominant"] = df.apply(dominant, axis=1)

    per_turn = (
        df.groupby("turn_id")
          .agg(
              total_chunks=("turn_id", "size"),
              dense_only=("dominant", lambda s: (s == "dense").sum()),
              fts_only=("dominant",   lambda s: (s == "fts").sum()),
              sql_only=("dominant",   lambda s: (s == "sql").sum()),
          )
          .reset_index()
    )

    per_turn["dense_only_pct"] = per_turn["dense_only"] / per_turn["total_chunks"]
    per_turn["fts_only_pct"]   = per_turn["fts_only"]   / per_turn["total_chunks"]
    per_turn["sql_only_pct"]   = per_turn["sql_only"]   / per_turn["total_chunks"]

    st.dataframe(
        per_turn[["turn_id", "total_chunks", "dense_only_pct", "fts_only_pct", "sql_only_pct"]]
        .sort_values("turn_id", ascending=False)
    )

    # --- Weight sensitivity (quick peek) --------------------------------
    st.subheader("Weight sensitivity (what-if)")

    # compare your stored fused_score to one alternative combo
    alt_col = [c for c in df.columns if c.startswith("fused_test_")][0]
    corr = df[["fused_score", alt_col]].corr().iloc[0, 1]

    st.write(
        f"Correlation between current fused score and `{alt_col}`: **{corr:.3f}** "
        "(1.0 = identical ranking, closer to 0 = very different)."
    )

    # mini preview table
    st.write("Sample chunks with original vs alt fused scores:")
    st.dataframe(
        df[["turn_id", "rank", "dense_score", "fts_score", "sql_score", "fused_score", alt_col]]
        .head(20)
    )


# Optional: run standalone
if __name__ == "__main__":
    render_fusion_perform()


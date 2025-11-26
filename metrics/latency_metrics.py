from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from config_base.config import Config
from metrics.common import query_df

DB_PATH = Path(getattr(Config, "DB_PATH", Path("database") / "chat_logs.sqlite"))


# --- Helpers -------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_latency_rows(limit: int = 2000) -> pd.DataFrame:
    """
    Load per-turn latency info from log_chat_turns.
    """
    df = query_df(
        """
        SELECT
          turn_id,
          session_id,
          question,
          latency_ms,
          dense_ms,
          ce_ms,
          llm_key,
          embed_key,
          ce_key,
          created_at
        FROM log_chat_turns
        ORDER BY created_at DESC
        LIMIT ?
        """,
        params=(limit,),
        parse_dates=["created_at"],
    )
    return df


def compute_stage_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dense_share, ce_share, other_share columns based on latency_ms.
    """
    df = df.copy()

    # Ensure numeric and non-null
    for col in ["latency_ms", "dense_ms", "ce_ms"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Avoid divide-by-zero
    denom = df["latency_ms"].replace(0, np.nan)

    df["dense_share"] = df["dense_ms"] / denom
    df["ce_share"]    = df["ce_ms"] / denom

    # Other = remainder (LLM + overhead + anything not in dense/CE)
    df["other_share"] = 1.0 - df["dense_share"].fillna(0) - df["ce_share"].fillna(0)

    # Clip to [0, 1] just in case of minor numeric weirdness
    for col in ["dense_share", "ce_share", "other_share"]:
        df[col] = df[col].clip(lower=0.0, upper=1.0)

    return df


def _p95(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(np.percentile(series, 95))


# --- Streamlit page ------------------------------------------------------

def render_latency_metrics():
    st.markdown("## ⏱ Latency Metrics")

    df = load_latency_rows(limit=5000)
    if df.empty:
        st.info("No chat turns logged yet.")
        return

    df = compute_stage_shares(df)

    # ---------- Overall latency stats ----------
    st.subheader("Overall latency stats")

    avg_latency = df["latency_ms"].mean()
    p95_latency = _p95(df["latency_ms"])

    avg_dense = df["dense_ms"].mean()
    p95_dense = _p95(df["dense_ms"])

    avg_ce = df["ce_ms"].mean()
    p95_ce = _p95(df["ce_ms"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg latency (ms)", f"{avg_latency:.1f}")
    c2.metric("p95 latency (ms)", f"{p95_latency:.1f}")
    c3.metric("Total turns", len(df))

    c4, c5, c6 = st.columns(3)
    c4.metric("Avg dense_ms", f"{avg_dense:.1f}")
    c5.metric("p95 dense_ms", f"{p95_dense:.1f}")
    c6.metric("Avg ce_ms", f"{avg_ce:.1f}")

    # ---------- Stage shares ----------
    st.subheader("Stage share of latency (dense / CE / other)")

    share_means = df[["dense_share", "ce_share", "other_share"]].mean()

    s1, s2, s3 = st.columns(3)
    s1.metric("Dense share (avg)", f"{share_means['dense_share']:.0%}")
    s2.metric("CE share (avg)",    f"{share_means['ce_share']:.0%}")
    s3.metric("Other share (avg)", f"{share_means['other_share']:.0%}")

    # Small table preview
    st.write("Sample per-turn breakdown:")
    st.dataframe(
        df[
            [
                "turn_id",
                "latency_ms",
                "dense_ms",
                "ce_ms",
                "dense_share",
                "ce_share",
                "other_share",
                "llm_key",
                "embed_key",
            ]
        ].head(30)
    )

    # ---------- Trend over time ----------
    st.subheader("Latency over time")

    time_df = (
        df[["created_at", "latency_ms", "dense_ms", "ce_ms"]]
        .sort_values("created_at")
        .rename(
            columns={
                "latency_ms": "Total latency (ms)",
                "dense_ms": "Dense stage (ms)",
                "ce_ms": "CE stage (ms)",
            }
        )
    )
    st.line_chart(time_df.set_index("created_at"))

    # ---------- By model / embed ----------
    st.subheader("Latency by model / embed combo")

    model_df = (
        df.groupby(["llm_key", "embed_key"])
        .agg(
            n_turns=("turn_id", "nunique"),
            avg_latency_ms=("latency_ms", "mean"),
            p95_latency_ms=("latency_ms", _p95),
            avg_dense_ms=("dense_ms", "mean"),
            avg_ce_ms=("ce_ms", "mean"),
            dense_share=("dense_share", "mean"),
            ce_share=("ce_share", "mean"),
            other_share=("other_share", "mean"),
        )
        .reset_index()
        .sort_values("avg_latency_ms", ascending=False)
    )

    st.dataframe(model_df)


# Optional: run this page standalone
if __name__ == "__main__":
    render_latency_metrics()

# evidently_07_text_report.py
from __future__ import annotations
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from metrics.common import query_df

from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import TextLength   # add more later (Sentiment, etc.)
from evidently.presets import TextEvals
from evidently.metrics import QuantileValue, ValueDrift  # replaces old ColumnQuantile/ColumnDrift


def render_metrics(db_path: Path, *, limit: int = 1000, height: int = 900) -> None:
    """Streamlit wrapper that renders the Evidently 0.7 TextEvals + a couple numeric metrics."""
    st.subheader("Text Evals")

    df = query_df(
            """
            SELECT turn_id, session_id, question, response,
                   latency_ms, dense_ms, similarity_avg, prefiltered, created_at
            FROM log_chat_turns
            ORDER BY created_at ASC
            LIMIT ?
            """,
            params=(int(limit),),
            parse_dates=["created_at"]
        )

    if df.empty:
        st.info("No rows found in chat logs yet.")
        return
    
    #st.caption(f"{len(df)} rows • newest at bottom")
    #st.dataframe(df.tail(50), use_container_width=True, height=height)

    # Split into reference/current so presets have something to compare
    cut = max(10, int(len(df) * 0.5))
    ref_df, cur_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    if len(ref_df) == 0 or len(cur_df) == 0:
        st.warning("Not enough data to create reference/current split yet.")
        st.dataframe(df.head(20))
        return

    # Map columns + compute descriptors
    definition = DataDefinition(
        text_columns=["question", "response"],
        numerical_columns=["latency_ms", "dense_ms", "similarity_avg"],
    )

    descriptors = [
        TextLength("response", alias="RespLen"),
        TextLength("question", alias="Qlen"),
    ]

    ref = Dataset.from_pandas(ref_df, data_definition=definition, descriptors=descriptors)
    cur = Dataset.from_pandas(cur_df, data_definition=definition, descriptors=descriptors)

    report = Report(
        metrics=[
            TextEvals(),                                   # summarizes the descriptors above
            QuantileValue(column="latency_ms", quantile=0.90),
            ValueDrift(column="RespLen"),                  # drift on descriptor column
        ],
        include_tests=True,
    )

    # Run and render inside Streamlit
    try:
        result = report.run(current_data=cur, reference_data=ref)
        try:
            html = result.get_html()
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            result.save_html(str(tmp_path))
            html = tmp_path.read_text(encoding="utf-8")

        st.components.v1.html(html, height=height, scrolling=True)
        st.download_button(
            "Download Evidently report",
            data=html,
            file_name="evidently_text_eval.html",
            mime="text/html",
        )
    except Exception as e:
        st.error(f"Evidently report failed: {e}")
        st.dataframe(df.head(20))


# Optional: run this module directly with `streamlit run evidently_07_text_report.py`
if __name__ == "__main__":
    DB_PATH = Path("database") / "chat_logs.sqlite"
    render_metrics(DB_PATH)

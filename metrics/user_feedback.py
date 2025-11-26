from pathlib import Path
from config_base.config import Config
import pandas as pd
import streamlit as st
from prepare_docs.db_io import get_conn
from metrics.common import query_df
#with get_conn(self.db_path, ensure=True) as log_conn:
#with get_conn(self.db_path, ensure=True) as _conn_for_hash:
DB_PATH = Path(getattr(Config, "DB_PATH", Path("database") / "chat_logs.sqlite"))

# metrics/user_feedback.py

def load_enriched(limit: int | None = 2000) -> pd.DataFrame:
    df = query_df("""
        SELECT
          turn_id, rater_session_id, rate_stars, rated_at,
          created_at, llm_key, embed_key, latency_ms, similarity_avg
        FROM vw_chat_ratings
        ORDER BY rated_at ASC
    """, parse_dates=["rated_at", "created_at"])
    if limit:
        df = df.tail(int(limit))
    return df

def summarize(df: pd.DataFrame):
    overall = (
        df.agg(
            mean_stars=("rate_stars", "mean"),
            std_stars =("rate_stars", "std"),
            n=("rate_stars", "count"),
        )
    )

    by_model = (
        df.groupby("llm_key", dropna=False)
          .agg(mean_stars=("rate_stars","mean"),
               std_stars =("rate_stars","std"),
               n=("rate_stars","count"))
          .reset_index()
          .sort_values(["mean_stars","n"], ascending=[False, False])
    )

    corr_overall = df[["rate_stars","latency_ms","similarity_avg"]].corr(method="pearson")

    # stars histogram 0–5
    hist = (df["rate_stars"]
              .value_counts()
              .reindex([0,1,2,3,4,5], fill_value=0)
              .rename_axis("stars")
              .reset_index(name="count"))

    return overall, by_model, corr_overall, hist

def render_user_feedback():
    st.markdown("### ⭐ User Feedback (Star Ratings)")
    limit = st.slider("Rows to include", 5, 100, 50, step=5)
    df = load_enriched(limit=limit)

    if df.empty:
        st.info("No ratings yet.")
        return

    overall, by_model, corr_overall, hist = summarize(df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Overall**")
        st.dataframe(overall, use_container_width=True)
    with c2:
        st.markdown("**By model**")
        st.dataframe(by_model, use_container_width=True)

    st.markdown("**Ratings distribution**")
    st.bar_chart(hist.set_index("stars")["count"])

    with st.expander("Correlations (stars vs latency/similarity)"):
        st.dataframe(corr_overall, use_container_width=True)

    # Quick filters to explore a specific model
    models = [m for m in df["llm_key"].dropna().unique()]
    if models:
        choose = st.selectbox("Inspect a model", options=["(all)"] + sorted(models))
        if choose != "(all)":
            sub = df[df["llm_key"] == choose]
            s_overall, s_by_model, s_corr, s_hist = summarize(sub)
            st.markdown(f"**{choose}**")
            st.dataframe(s_overall, use_container_width=True)
            st.bar_chart(s_hist.set_index("stars")["count"])
            with st.expander(f"Correlations for {choose}"):
                st.dataframe(s_corr, use_container_width=True)


# Optional: run this module directly with `streamlit run evidently_07_text_report.py`
if __name__ == "__main__":
    DB_PATH = Path("database") / "chat_logs.sqlite"
    render_user_feedback()
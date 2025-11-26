
import pandas as pd
import streamlit as st
import logging
from pathlib import Path
from config_base.config import Config
from prepare_docs.db_io import get_conn

logger = logging.getLogger(__name__)

DB_PATH = Path(getattr(Config, "DB_PATH", Path("database") / "chat_logs.sqlite"))

@st.cache_data(show_spinner=False)
def query_df(sql: str, params: tuple = (), parse_dates: list[str] | None = None) -> pd.DataFrame:
    with get_conn(ensure=False) as c:                      # ← ensure=False in tabs
        return pd.read_sql_query(sql, c, params=params, parse_dates=parse_dates or [])


# eval_export.py
import sqlite3, json
import pandas as pd

def build_rag_eval_df(db_path: str, run_id: str) -> pd.DataFrame:
    """
    Build a dataset for Evidently from your logs.
    Assumes you keep a table of query runs and which chunks were retrieved.
    Adapt the SELECTs to your actual tables.
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Example schema assumptions:
    # - query_runs(run_id, question, response, target)  -- target optional
    # - query_retrievals(run_id, chunk_id, rank, score)
    # - doc_chunks_v2(chunk_id, chunk_text)
    q = con.execute("""
        SELECT r.run_id, r.question AS Question, r.response AS Response, r.target AS Target
        FROM query_runs r
        WHERE r.run_id = ?
    """, (run_id,)).fetchone()
    if not q:
        raise ValueError(f"run_id {run_id} not found")

    rows = con.execute("""
        SELECT rc.chunk_id, rc.rank, rc.score, c.chunk_text
        FROM query_retrievals rc
        JOIN doc_chunks_v2 c ON c.chunk_id = rc.chunk_id
        WHERE rc.run_id = ?
        ORDER BY rc.rank ASC
    """, (run_id,)).fetchall()

    contexts = [r["chunk_text"] for r in rows]  # list[str] for multi-context
    df = pd.DataFrame([{
        "Question": q["Question"],
        "Context": contexts if contexts else "",
        "Response": q["Response"],
        "Target": q["Target"] if "Target" in q.keys() else None
    }])
    con.close()
    return df

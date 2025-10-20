import sqlite3, sys

DB_PATH = r"C:\dev\GovernEdge_CLI\database\chat_logs.sqlite"
MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "nomic"

with sqlite3.connect(DB_PATH) as conn:
    conn.execute("DELETE FROM chunk_embedding_state WHERE model_key = ?", (MODEL_KEY,))
    conn.commit()
    conn.execute("VACUUM")
    print(f"✅ Cleared embeddings for {MODEL_KEY}")


# python llm_core_tst\vector_index_tst\clear_embed.py nomic
# python llm_core_tst\vector_index_tst\clear_embed.py snow
# python llm_core_tst\vector_index_tst\clear_embed.py minilm

# python C:\dev\GovernEdge_CLI\llm_core_tst\vector_index_tst\clear_embed.py 


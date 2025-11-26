import os
import matplotlib.pyplot as plt
from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment="GovernEdge Architecture", format="png")
dot.attr(rankdir="LR", size="8")

# ... build your graph ...

# Option B: write directly to a .png (recent graphviz versions)
# dot.render(outfile=rf"{outdir}\governedge_architecture.png", cleanup=True)


# Ingestion pipeline nodes
dot.node("discover_hash", "discover_hash.py\n(Stage 1: Discovery + Hash)")
dot.node("frontmatter", "frontmatter_canon.py\n(Stage 2: Front‑matter)")
dot.node("chunker", "chunker.py\n(Stage 3: Chunking)")
dot.node("taxonomy", "apply_taxonomy.py\n(Stage 4: Taxonomy)")
dot.node("spacy", "spacy_clean.py + nlp_cache.py\n(Stage 5: spaCy Clean Cache)")
dot.node("fts", "fts_mirror.py\n(Stage 6: FTS Mirror)")

# DB IO and schema
dot.node("db", "db_io.py\nSQLite schema + helpers", shape="cylinder")
dot.node("master_db", "seed_master_data_test.py\nMaster Data DB", shape="cylinder")

# Vector indexing
dot.node("vector_index_loader", "vector_index_loader.py\nVectorstore Loader")
dot.node("retriever_vector", "retriever_vector.py\nVector Retriever")
dot.node("retriever_fts", "retriever_fts.py\nFTS Retriever")
dot.node("retriever_sql", "retriever_sql.py\nSQL Retriever")

# Query engine components
dot.node("query_engine", "query_engine.py\nHybrid Query Engine")
dot.node("engine_fusion", "engine_fusion.py\nFusion Strategies")
dot.node("engine_facets", "engine_facets.py\nFacet Prefilter")
dot.node("engine_cross_encoder", "engine_cross_encoder.py\nCross Encoder")
dot.node("engine_load_models", "engine_load_models.py\nModel Loader")

# Orchestrator
dot.node("ingest_docs", "ingest_docs.py\nPipeline Orchestrator")

# Flow edges
dot.edges([
    ("discover_hash", "frontmatter"),
    ("frontmatter", "chunker"),
    ("chunker", "taxonomy"),
    ("taxonomy", "spacy"),
    ("spacy", "fts"),
    ("fts", "db"),
])

# DB connections
dot.edge("db", "query_engine", label="doc_chunks + FTS")
dot.edge("master_db", "query_engine", label="md_* tables")

# Vector pipeline
dot.edge("vector_index_loader", "retriever_vector")
dot.edge("retriever_vector", "query_engine")
dot.edge("retriever_fts", "query_engine")
dot.edge("retriever_sql", "query_engine")
dot.edge("engine_fusion", "query_engine")
dot.edge("engine_facets", "query_engine")
dot.edge("engine_cross_encoder", "query_engine")
dot.edge("engine_load_models", "query_engine")

# Orchestrator arrow
dot.edge("ingest_docs", "discover_hash", style="dashed")
dot.edge("ingest_docs", "fts", style="dashed")
dot.edge("ingest_docs", "taxonomy", style="dashed")

# Render to file
# output_path = "C:\dev\GovernEdge_CLI\config_base"
#dot.render(output_path, format="png", cleanup=True)

outdir = r"C:\dev\GovernEdge_CLI\config_base"
os.makedirs(outdir, exist_ok=True)

# Option A: filename + directory (Graphviz creates .gv and .png)
dot.render(filename="governedge_architecture",
           directory=outdir,
           cleanup=True)   # removes the .gv after rendering

outdir + ".png"

# Create a simplified high‑level architecture PNG for presentations


dot = Digraph('GovernEdge_HighLevel', format='png')
dot.attr(rankdir='LR', fontsize='12', labelloc='t', label='GovernEdge — High‑Level Architecture')

# Nodes
dot.node('ingest', 'Ingestion\n(discovery → FM → chunk → taxonomy → spaCy → FTS)', shape='box')
dot.node('storage', 'Storage\n(SQLite + Chroma)', shape='box')
dot.node('retrieval', 'Retrieval\n(Vector • FTS • SQL)', shape='box')
dot.node('engine', 'Query Engine\n(Facets • Fusion • Cross‑Encoder)', shape='box')
dot.node('ui', 'Streamlit UI', shape='box')

# Edges
dot.edge('ingest', 'storage', label='persist')
dot.edge('storage', 'retrieval', label='indexes + data')
dot.edge('retrieval', 'engine', label='top‑k candidates')
dot.edge('engine', 'ui', label='answers + citations')

# Render
out = "C:\dev\GovernEdge_CLI\config_base"
# Option A: filename + directory (Graphviz creates .gv and .png)
dot.render(filename="governedge_arch",
           directory=outdir,
           cleanup=True)   # removes the .gv after rendering

out + '.png'

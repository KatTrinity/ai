
# C:\dev\GovernEdge_CLI\prepare_docs\chunker.py

"""
chunker.py
----------
Stage 3 of the pipeline: take canonicalized docs and split into
(H1/H2) sections and micro‑chunks with stable IDs + per‑chunk hashes.

Inputs  (per Document):
  - d.page_content  : canonical raw text (front-matter already stripped)
  - d.metadata      : MUST include "doc_id" (int/str)

Outputs (List[Document]):
  - page_content    : raw chunk text
  - metadata        : {
        doc_id, section_index, section_id,
        chunk_index, chunk_id, header_path,
         chunk_hash_raw
    }

Why a SECOND hash?
- File hash (discover stage) = "did the file change?"
- Chunk hash (here)         = "which chunk(s) changed?"
  This powers idempotent updates to:
    * doc_chunks (raw)
    * doc_nlp_cache (cleaned text keyed by chunk_id + nlp_version)
    * doc_chunks_fts + _state (keyword index mirror)
"""

"""
Handles splitting documents into hierarchical sections + micro-chunks,
and rebuilding chunks in the database with stable IDs + hashes.
"""

import os, re, hashlib, logging, sqlite3, string
from typing import List
from pathlib import Path
from langchain.schema import Document

# ---------------- logging ----------------
logger = logging.getLogger(__name__)

# --------- Chunking defaults (char-based) ----------
# Rough heuristic: ~5-6 characters per token in SAP-ish text.
SAP_CHUNK_SIZE_CHARS = 900    # ~150–200 tokens
SAP_CHUNK_OVERLAP_CHARS = 150 # ~25–35 token overlap

# ---------------- heading markers ----------------
_H1 = "# "
_H2 = "## " 

def estim_token_count(text: str) -> int:
    # rough: word-based
    return len((text or "").split())

def _sha(text: str) -> str:
    """Stable SHA-256 for change detection at chunk granularity."""
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()

def _in_code_fence(s: str, start: int, end: int) -> bool:
    """
    True if [start:end) lies inside any supported code/block fence.
    Fence types include:
      - ```
      - ::: 
      - ''' (rare, but SAP sometimes emits)
    
    Logic:
      Count the number of opening fences before `end` minus before `start`.
      If the number is odd → we're inside a block.
    """

    # All fence strings the splitter should treat as “block mode”
    fences = ["```", ":::","'''"]

    def count_fences(upto: int) -> int:
        c = 0
        for f in fences:
            c += s[:upto].count(f)
        return c

    return (count_fences(end) - count_fences(start)) % 2 == 1


def assess_chunk_quality(text: str) -> dict:
    if not text:
        return dict(
            chunk_quality_score=0.0,
            is_near_duplicate=0,
            has_symptom_resolution_pair=0,
        )

    total = len(text)
    alpha = sum(c.isalpha() for c in text)
    allowed = set(string.ascii_letters + string.digits +
                  string.punctuation + " \n\t")
    weird = sum(1 for c in text if c not in allowed)

    weird_ratio = weird / max(total, 1)
    alpha_ratio = alpha / max(total, 1)

    # Simple length flags
    too_short = total < 40
    too_long  = total > 3000

    # Cheap "possible near-duplicate" check
    # (you can replace this later with SimHash)
    near_dup = 1 if weird_ratio < 0.001 and alpha_ratio > 0.90 and total < 100 else 0

    # Compress into a 0–1 score
    # (totally tunable)
    score = 1.0
    if too_short:
        score -= 0.5
    if too_long:
        score -= 0.2
    score -= weird_ratio * 2.0
    score = max(0.0, min(1.0, score))

    return dict(
        chunk_quality_score=score,
        is_near_duplicate=near_dup,
    )

def load_documents(directory: str = "data_tst") -> List[Document]:
    """
    Convenience loader for quick experiments.
    In the full pipeline, you usually pass in Documents built from the DB,
    with doc_id set and front-matter already stripped.
    """
    from langchain_community.document_loaders import TextLoader

    docs: List[Document] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.lower().endswith((".txt", ".md")):
                continue
            filepath = os.path.join(root, filename)
            try:
                loader = TextLoader(filepath, encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                logger.warning("⚠️ Failed to load %s: %s", filename, e)
    logger.info("Loaded %d documents from %s", len(docs), directory)
    return docs


def split_documents(
    docs: List[Document],
    *,
    chunk_size: int = SAP_CHUNK_SIZE_CHARS,      # ≈300–500 tokens worth of chars
    chunk_overlap: int = SAP_CHUNK_OVERLAP_CHARS,    # ~15% overlap to improve recall
    max_heading_indent: int = 3  # allow up to 3 leading spaces before '#'
) -> List[Document]:
    """
    Split each Document into H1/H2 sections, then window into micro-chunks.

    Stable IDs:
      section_id = "{doc_id}:S{sec_idx:03d}"
      chunk_id   = "{section_id}:C{local_idx:04d}"

    chunk_hash_raw:
      SHA-256 over (header_path + "\n" + chunk_text).
      If either header path or text changes, hash changes.
    """
    out: List[Document] = []

    for d in docs:
        text = (d.page_content or "").replace("\r\n", "\n").replace("\r", "\n")
        n = len(text)

        # -------- Pass 1: carve H1/H2 sections --------
        sections = []
        cur_headers: list[str] = []
        buf_start = 0
        pos = 0  # absolute char position in the doc

        def flush(end_pos: int):
            """Close out the current section buffer."""
            nonlocal buf_start
            if end_pos > buf_start:
                sec_text = text[buf_start:end_pos]
                sections.append((" > ".join(cur_headers), buf_start, end_pos, sec_text))

        for ln in text.splitlines(keepends=True):
            ln_stripped = ln.lstrip()
            indent = len(ln) - len(ln_stripped)

            if indent <= max_heading_indent and ln_stripped.startswith(_H1):
                flush(pos)
                cur_headers = [f"H1:{ln_stripped[len(_H1):].strip()}"]
                buf_start = pos
            elif indent <= max_heading_indent and ln_stripped.startswith(_H2):
                flush(pos)
                if cur_headers and cur_headers[0].startswith("H1:"):
                    cur_headers = [cur_headers[0], f"H2:{ln_stripped[len(_H2):].strip()}"]
                else:
                    cur_headers = [f"H2:{ln_stripped[len(_H2):].strip()}"]
                buf_start = pos

            pos += len(ln) 

        flush(n)

        # If no headings, treat whole doc as a single section
        if not sections:
            sections = [("", 0, n, text)]

        # -------- Pass 2: window each section into chunks --------
        sec_idx = 0
        for header_path, s_start, s_end, sec_text in sections:
            sec_text = sec_text.strip("\n")
            if not sec_text:
                sec_idx += 1
                continue

            doc_id = d.metadata.get("doc_id")
            if doc_id is None:
                raise ValueError("split_documents requires metadata['doc_id'] on each input Document.")

            section_id = f"{doc_id}:S{sec_idx:03d}"

            start = 0
            local_idx = 0
            while start < len(sec_text):
                end = min(len(sec_text), start + chunk_size)

                # Avoid ending inside a code fence by extending to the closing ```
                if _in_code_fence(sec_text, start, end):
                    nxt = sec_text.find("```", end)
                    if nxt != -1:
                        end = min(len(sec_text), nxt + 3)

                chunk = sec_text[start:end].strip()
                if not chunk:
                    break

                chunk_id = f"{section_id}:C{local_idx:04d}"
                chash = _sha(header_path + "\n" + chunk)  # stable hash

                token_count = estim_token_count(chunk)

                meta = dict(d.metadata)
                meta.update({
                    "section_index": sec_idx,
                    "section_id": section_id,
                    "chunk_index": local_idx,
                    "chunk_id": chunk_id,
                    "header_path": header_path,
                    "chunk_hash_raw": chash,
                    "token_count": token_count,
                })

                out.append(Document(page_content=chunk, metadata=meta))
                local_idx += 1

                if end >= len(sec_text):
                    break
                # Overlap ensures redundancy across chunk boundaries
                start = max(0, end - chunk_overlap)

            sec_idx += 1

    logger.info("Split %d documents into %d chunks", len(docs), len(out))
    return out

def rebuild_chunks_for_doc(
    conn: sqlite3.Connection,
    doc_id: int,
    char_window: int = SAP_CHUNK_SIZE_CHARS,
    overlap: int = SAP_CHUNK_OVERLAP_CHARS,
) -> None:
    """
    Rebuild chunks for one doc_id:
      - Load canonical body from doc_ingest/doc_canonical
      - Run split_documents()
      - Delete prior chunks
      - Insert new rows into doc_chunks
    """
    cur = conn.cursor()

    # pull canonical body metadata
    row = cur.execute(
        "SELECT file_name, title, fm_json FROM doc_ingest WHERE doc_id=?",
        (doc_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"doc_id {doc_id} not found in doc_ingest")

    file_name, title, fm_json = row
    body_row = cur.execute(
        "SELECT canonical_body FROM doc_canonical WHERE doc_id=?",
        (doc_id,),
    ).fetchone()
    if not body_row:
        raise ValueError(
            f"doc_id {doc_id} has no canonical_body — did you run frontmatter_canon?"
        )
    body_text = body_row[0]

    # Build one Document with metadata
    d = Document(
        page_content=body_text,
        metadata={
            "doc_id": doc_id,
            "file_name": file_name,
            "title": title,          # if you already parsed a front-matter title
            # add anything else you want propagated into chunks
        },
    )

    chunks = split_documents([d], chunk_size=char_window, chunk_overlap=overlap)

    # Clear old rows
    cur.execute("DELETE FROM doc_chunks WHERE doc_id=?", (doc_id,))

    # Prepare rows for bulk insert
    rows = []
    for ch in chunks:
        text = ch.page_content or ""
        meta = ch.metadata or {}

        stats = assess_chunk_quality(text)

        rows.append((
            meta["chunk_id"],
            meta["doc_id"],
            meta.get("title"),
            #meta.get("section_id"),                 # NEW: keep section_id in sync with schema. might use later. add to insert
            meta.get("header_path") or "",
            text,
            meta["chunk_hash_raw"],
            meta.get("token_count"),
            stats["chunk_quality_score"],
            stats["is_near_duplicate"],
        ))

    cur.executemany(
        """
        INSERT INTO doc_chunks (
            chunk_id,
            doc_id,
            title,
            header_path,
            body_raw,
            chunk_hash_raw,
            token_count,
            chunk_quality_score,
            is_near_duplicate,
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    conn.commit()
    logger.info("✅ doc_id=%s: %d chunks rebuilt", doc_id, len(chunks))

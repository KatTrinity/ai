
# C:\dev\GovernEdge_CLI\prepare_docs\spacy_clean.py

"""
spacy_clean.py

Pipeline for cleaning text chunks with spaCy (tokenizer-only):
- Uses spaCy English tokenizer + stopwords (no tagger/lemma/parser/NER).
- Tokenizes while preserving code-like tokens (IDs, ALLCAPS, numbers).
- Preserves fenced code blocks and inline `code`.
- Drops stop words.
- Optionally stores a simple token dump in pos_json (no real POS).
- Caches cleaned text into SQLite.
"""

import os, re, json, sqlite3, argparse, logging
from pathlib import Path
from contextlib import closing

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

# ✅ use package-relative imports
from .nlp_cache import fetch_chunks_to_process, upsert_clean_batch, sha256
from prepare_docs.db_io import get_conn  # uses your PRAGMAs + ensure_db

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spacy_clean")

# ---- Load spaCy tokenizer only ----
try:
    # Blank English pipeline: tokenizer + vocab, no tagger/parser/ner/lemmatizer
    nlp = English()

    # Register stopwords on the vocab so tok.is_stop works
    for w in STOP_WORDS:
        lex = nlp.vocab[w]
        lex.is_stop = True

    logger.info("Initialized spaCy English tokenizer (blank) with stopwords")
except Exception as e:
    raise SystemExit(f"Failed to initialize spaCy English tokenizer: {e}")

# ---------------- config ----------------
# We no longer rely on POS tags; we only use stopwords.
JOINABLE = {"-", "/", "_"}  # ID joiners like MM-01, S/4HANA, GL_Account_100

# regexes
FENCE_BLOCK_RE = re.compile(r"(```.*?```)", re.S)   # preserve fenced blocks
INLINE_CODE_RE = re.compile(r"`([^`]+)`")           # preserve inline `code`
WS_MULTI_RE     = re.compile(r"\s+")
CODE_RE = re.compile(
    r"^(?:[A-Z0-9]{2,}|[A-Za-z0-9]*\d[A-Za-z0-9]*)$"
)  # preserve rawish tokens (IDs, tcodes, etc.)

# Optional extra words to drop beyond spaCy stopwords (if you want)
DROP_WORDS = {
    "the", "a", "an",
    "it", "they", "them",
    "this", "that", "these", "those",
}


# ---------------- token merging ----------------
def _merge_alnum_joiners(doc):
    """
    Merge sequences like [ALNUM] ('-'|'/'|'_') [ALNUM] into one token.
    Example: "GL" "-" "100" → "GL-100"

    Returns:
        List of merged token texts.
    """
    out = []
    i = 0
    while i < len(doc):
        t = doc[i]
        if (
            t.text in JOINABLE
            and 0 < i < len(doc) - 1
            and doc[i - 1].text.isalnum()
            and doc[i + 1].text.isalnum()
        ):
            merged = out.pop() + t.text + doc[i + 1].text
            out.append(merged)
            i += 2
            continue
        out.append(t.text)
        i += 1
    return out


# ---------------- cleaning core ----------------
def _clean_plain(text: str, keep_pos: bool = False):
    """
    Clean one text block with spaCy tokenizer-only:
    - Drop stopwords (spaCy STOP_WORDS + DROP_WORDS).
    - Lowercase normal tokens.
    - Preserve ALLCAPS/IDs/codes defined by CODE_RE (just lowercased).
    - Optionally store a simple token dump in pos_json (no real POS).
    """
    doc = nlp(text)
    raw_tokens = _merge_alnum_joiners(doc)

    cleaned = []
    pos_dump = [] if keep_pos else None
    j = 0

    for i, tok in enumerate(doc):
        raw = tok.text
        if j >= len(raw_tokens):
            break
        current = raw_tokens[j]

        # Handle joiners (already merged into a previous/next token)
        if (
            raw in JOINABLE
            and i > 0
            and i < len(doc) - 1
            and doc[i - 1].text.isalnum()
            and doc[i + 1].text.isalnum()
        ):
            # This character was folded into a merged token; skip it.
            continue

        # Align with merged token sequence
        if raw.isalnum() or raw == current:
            t = current

            # Skip punctuation / spaces
            if tok.is_punct or tok.is_space:
                j += 1
                continue

            # Drop stopwords
            if tok.is_stop or tok.text.lower() in DROP_WORDS:
                j += 1
                continue

            # Preserve IDs/codes/tcodes as-is (just lowercased)
            if CODE_RE.match(t):
                cleaned.append(t.lower())
            else:
                # We don't have lemmas, so just lowercase token
                cleaned.append(t.lower())

            if keep_pos:
                # Note: no POS in tokenizer-only mode; we just store the token text.
                pos_dump.append({"t": t})

            j += 1
            continue

        # If token didn’t align with current merged chunk, skip it
        continue

    out_text = WS_MULTI_RE.sub(" ", " ".join(cleaned)).strip()
    pos_json = json.dumps(pos_dump, ensure_ascii=False) if keep_pos else None
    return out_text, pos_json


# ---------------- cleaning wrapper ----------------
def spacy_clean_keep_code(text: str, keep_pos: bool = False):
    """
    Clean text but preserve code blocks:
    - Fenced ```blocks``` passed through unchanged
    - Inline `code` segments passed through unchanged
    - Only non-code text is cleaned via _clean_plain
    """
    text = text.replace("\r\n", "\n")

    # Split by fenced code blocks first, keep the fences
    parts = FENCE_BLOCK_RE.split(text)

    cleaned_segments = []
    last_pos_json = None  # we keep simple behavior for keep_pos

    # Regex to split inline code, keeping the `code` segments
    inline_split_re = re.compile(r"(`[^`]+`)")

    for part in parts:
        if not part:
            continue

        # 1) Fenced code block: pass through unchanged
        if part.startswith("```"):
            cleaned_segments.append(part)
            continue

        # 2) Non-fenced section: handle inline code + normal text
        subparts = inline_split_re.split(part)

        for sub in subparts:
            if not sub:
                continue

            # Inline code: pass through unchanged
            if inline_split_re.fullmatch(sub):
                cleaned_segments.append(sub)
                continue

            # Normal text: clean with spaCy tokenizer
            cleaned, pos_json = _clean_plain(sub, keep_pos=keep_pos)
            if cleaned:
                cleaned_segments.append(cleaned)
            if keep_pos:
                # For now we just keep the last non-empty pos_json
                last_pos_json = pos_json

    final_text = WS_MULTI_RE.sub(" ", " ".join(cleaned_segments)).strip()
    return final_text, (last_pos_json if keep_pos else None)



# ---------------- main runner ----------------
def spacy_main(db_path: str, nlp_version: str, keep_pos: bool, batch_size: int = 200):
    """
    Run spaCy cleaning across all chunks, update cache table.

    Args:
        db_path: path to SQLite database
        nlp_version: version key for invalidation (e.g., "spacy.v2.tokenizer")
        keep_pos: whether to store a simple token dump JSON
        batch_size: commit interval
    """
    with get_conn(db_path, ensure=True) as conn:
        rows = fetch_chunks_to_process(conn, nlp_version)
        to_write = []
        total_updated = total_skipped = 0

        for r in rows:
            cid = r["chunk_id"]
            raw = r["raw_text"] or ""
            cleaned, pos_json = spacy_clean_keep_code(raw, keep_pos=keep_pos)

            # hash of cleaned text for cache invalidation
            th = sha256(cleaned)

            if r["cached_hash"] == th:
                total_skipped += 1
                continue

            to_write.append((cid, th, cleaned, pos_json))
            if len(to_write) >= batch_size:
                upsert_clean_batch(conn, nlp_version, to_write)
                conn.commit()
                total_updated += len(to_write)
                to_write.clear()

        # final flush (after the loop)
        if to_write:
            upsert_clean_batch(conn, nlp_version, to_write)
            conn.commit()
            total_updated += len(to_write)

        logger.info(
            "✅ spaCy tokenizer cache update complete → updated=%d, skipped=%d",
            total_updated,
            total_skipped,
        )


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to your SQLite DB")
    ap.add_argument(
        "--nlp-version",
        default="spacy.v2.tokenizer",
        help="Version key for cache invalidation",
    )
    ap.add_argument(
        "--keep-pos",
        action="store_true",
        help="Store simple token dump JSON (no real POS)",
    )
    args = ap.parse_args()

    spacy_main(args.db, args.nlp_version, keep_pos=args.keep_pos)


    #python -m prepare_docs.spacy_clean --db "C:\dev\GovernEdge_CLI\database\chat_logs.sqlite" --nlp-version spacy.v1 --keep-pos


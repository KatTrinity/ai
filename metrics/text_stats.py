import regex as re
from collections import Counter
import spacy
from pathlib import Path
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load spaCy only once (fast enough on CPU for your pipeline)
_nlp = spacy.load("en_core_web_sm", disable=["ner"])  # NER optional for speed


def analyze_doc(text: str, *, top_n: int = 20) -> dict:
    """
    Analyze a single SAP document and return useful linguistic + domain metrics.
    Designed for per-doc pre-ingest diagnostics (lightweight, CPU-only).

    Returns a dict like:
    {
        "n_tokens": int,
        "n_unique_tokens": int,
        "type_token_ratio": float,
        "top_words": [(word, count), ...],
        "top_bigrams": [(('word1','word2'), count), ...],
        "tcodes": [...],
        "tables": [...],
        "avg_sentence_length": float,
        "noun_phrases_sample": [...]
    }
    """

    # ---------------------------------------------------------------------
    # 1. Basic tokenization (fast, using regex)
    # ---------------------------------------------------------------------
    tokens = re.findall(r"\p{L}+", text.lower())
    n_tokens = len(tokens)
    n_unique_tokens = len(set(tokens))
    ttr = n_unique_tokens / n_tokens if n_tokens else 0.0


    # ---------------------------------------------------------------------
    # 2. Unigram & bigram frequencies
    # ---------------------------------------------------------------------
    unigram_counts = Counter(tokens)
    top_words = unigram_counts.most_common(top_n)

    def build_ngrams(tok_list, n):
        return list(zip(*[tok_list[i:] for i in range(n)]))

    bigrams = build_ngrams(tokens, 2)
    trigrams = build_ngrams(tokens, 3)
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)
    top_bigrams = bigram_counts.most_common(top_n)
    top_trigrams = trigram_counts.most_common(top_n)


    # ---------------------------------------------------------------------
    # 3. SAP-specific patterns (quick regex passes)
    # ---------------------------------------------------------------------
    # tcode: uppercase 2–10 chars/digits (simple heuristic)
    tcodes = re.findall(r"\b[A-Z0-9]{2,10}\b", text)

    # table names (MARA, MARC, EKPO, etc.)
    tables = re.findall(r"\b[A-Z]{4,4}\b", text)

    #notes = re.findall(r"\b[A-Z0-9_]{4,10}\b", text)

    # ---------------------------------------------------------------------
    # 4. Sentence-level & phrase-level analysis (spaCy)
    # ---------------------------------------------------------------------
    doc = _nlp(text)

    tokens_2 = [t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space]

    #tokens_3 = [t.text for t in doc]
    #pos_tags = [(t.text, t.pos_, t.tag_) for t in doc]

    trigrams_2 = list(zip(tokens_2, tokens_2[1:], tokens_2[2:]))
    trigrams_2_counts = Counter(trigrams_2)
    top_trigrams_2 = trigrams_2_counts.most_common(top_n)

    # sentence lengths
    sent_lens = [len([t for t in sent if not t.is_space]) for sent in doc.sents]
    avg_sent_len = sum(sent_lens) / len(sent_lens) if sent_lens else 0

    # noun phrase sample (first N phrases)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks][:top_n]


    # ---------------------------------------------------------------------
    #  Plots
    # ---------------------------------------------------------------------

    counts = Counter(tokens)
    freq_df = pd.DataFrame(
        {"token": list(counts.keys()), "freq": list(counts.values())}
    ).sort_values("freq", ascending=False)

    freq_df["rank"] = range(1, len(freq_df) + 1)

    plt.loglog(freq_df["rank"], freq_df["freq"], marker=".")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf curve")
    #plt.show()

    vocab_sizes = []
    token_counts = []

    seen = set()
    running_tokens = 0

    for doc_tokens in ["tokens"]:
        for tok in doc_tokens:
            running_tokens += 1
            seen.add(tok)
        token_counts.append(running_tokens)
        vocab_sizes.append(len(seen))

    plt.plot(token_counts, vocab_sizes)
    plt.xlabel("Total tokens")
    plt.ylabel("Vocabulary size")
    plt.title("Heaps curve")
    #plt.show()

    # ---------------------------------------------------------------------
    # 5. Bundle results
    # ---------------------------------------------------------------------
    return {
        "n_tokens": n_tokens,
        "n_unique_tokens": n_unique_tokens,
        "type_token_ratio": round(ttr, 4),
        "top_words": top_words,
        "top_bigrams": top_bigrams,
        "top_trigrams": top_trigrams,
        "top_trigrams_2" : top_trigrams_2,
        "tcodes": tcodes,
        "tables": tables,
        #"notes": notes,
        "avg_sentence_length": round(avg_sent_len, 2),
        "noun_phrases_sample": noun_phrases,
    }

def load_and_analyze(path: str | Path) -> dict:
    """
    Read a file from disk and run analyze_doc() on it.
    Supports plain .txt, .md, .html, SAP Note exports, etc.
    """

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8", errors="ignore")

    return analyze_doc(text)

result = load_and_analyze(r"C:\dev\GovernEdge_CLI\bad_ques.txt")

print(["n_tokens"])
print(result["n_tokens"])
print(["n_unique_tokens"])
print(result["n_unique_tokens"])
print(["type_token_ratio"])
print(result["type_token_ratio"])

print(["*****************"])

print(["avg_sentence_length"])
print(result["avg_sentence_length"])
print(["top_words"])
#print(result["top_words"])
print(["noun_phrases_sample"])
#print(result["noun_phrases_sample"])

print(["*****************"])

print(["top_bigrams"])
#print(result["top_bigrams"])
print(["top_trigrams"])
print(result["top_trigrams"])
print(["top_trigrams_2"])
print(result["top_trigrams_2"])

print(["*****************"])

print(["tables"])
print(result["tables"])
print(["tcodes"])
print(result["tcodes"])



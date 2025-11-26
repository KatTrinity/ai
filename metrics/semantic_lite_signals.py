from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


# Load a small, CPU-friendly embedding model once.
# You can swap this to match your eventual embedder.
_EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str, min_len: int = 40) -> List[str]:
    """
    Split on blank lines; keep only reasonably long paragraphs.
    """
    raw_parts = text.split("\n\n")
    paras: List[str] = []
    for part in raw_parts:
        p = part.strip()
        if len(p) >= min_len:
            paras.append(p)
    return paras or [text.strip()]  # fallback: whole doc as 1 paragraph


def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using the shared model. Returns (n, d) array.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # MiniLM has 384 dims
    embs = _EMBED_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs.astype(np.float32)


# ---------------------------------------------------------------------------
# 1) Paragraph similarity scoring (within one doc)
# ---------------------------------------------------------------------------

@dataclass
class ParagraphRedundancyResult:
    paragraphs: List[str]
    redundant_pairs: List[Tuple[int, int, float]]  # (i, j, similarity)
    similarity_matrix_shape: Tuple[int, int]


def analyze_paragraph_similarity(path: str | Path,
                                 threshold: float = 0.9) -> Dict:
    """
    For a single doc:
      - split into paragraphs
      - embed each paragraph
      - compute cosine similarity
      - return pairs above `threshold` as potential duplicates

    threshold ~0.9 is strict (only very similar paragraphs).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8", errors="ignore")
    paragraphs = _split_paragraphs(text)

    embs = _embed_texts(paragraphs)
    if embs.shape[0] < 2:
        result = ParagraphRedundancyResult(
            paragraphs=paragraphs,
            redundant_pairs=[],
            similarity_matrix_shape=embs.shape,
        )
        return asdict(result)

    sim = cosine_similarity(embs)  # (n, n)

    redundant_pairs: List[Tuple[int, int, float]] = []
    n = sim.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s >= threshold:
                redundant_pairs.append((i, j, round(s, 4)))

    result = ParagraphRedundancyResult(
        paragraphs=paragraphs,
        redundant_pairs=redundant_pairs,
        similarity_matrix_shape=sim.shape,
    )
    return asdict(result)


# ---------------------------------------------------------------------------
# 2) Topic clustering (multi-doc)
# ---------------------------------------------------------------------------

@dataclass
class TopicClusteringResult:
    files: List[str]
    labels: List[int]
    cluster_centers: List[List[float]]


def cluster_documents(paths: List[str | Path],
                      n_clusters: int = 5) -> Dict:
    """
    Cluster documents by topic using averaged paragraph embeddings.

    Returns:
      - cluster label per file (0..n_clusters-1)
      - cluster centers (for inspection / future distance checks).
    """
    texts: List[str] = []
    files: List[str] = []

    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        texts.append(text)
        files.append(str(p))

    if not texts:
        return TopicClusteringResult(files=[], labels=[], cluster_centers=[]).__dict__

    # Represent each doc as the mean of its paragraph embeddings
    doc_embs: List[np.ndarray] = []
    for text in texts:
        paras = _split_paragraphs(text)
        embs = _embed_texts(paras)
        doc_emb = embs.mean(axis=0) if embs.size > 0 else np.zeros((384,), dtype=np.float32)
        doc_embs.append(doc_emb)

    X = np.vstack(doc_embs)
    k = min(n_clusters, X.shape[0])  # can't have more clusters than docs

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    result = TopicClusteringResult(
        files=files,
        labels=list(map(int, labels)),
        cluster_centers=kmeans.cluster_centers_.tolist(),
    )
    return asdict(result)


# ---------------------------------------------------------------------------
# 3) Outlier detection (multi-doc)
# ---------------------------------------------------------------------------

@dataclass
class OutlierDetectionResult:
    files: List[str]
    anomaly_scores: List[float]
    is_outlier: List[bool]


def detect_outlier_documents(paths: List[str | Path],
                             contamination: float = 0.1) -> Dict:
    """
    Use IsolationForest on doc embeddings to flag outlier docs.
    contamination ~ fraction of points expected to be outliers (0.0–0.5).

    Returns:
      - anomaly_scores: larger = more anomalous
      - is_outlier: boolean flag per file
    """
    texts: List[str] = []
    files: List[str] = []

    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        texts.append(text)
        files.append(str(p))

    if not texts:
        return OutlierDetectionResult(files=[], anomaly_scores=[], is_outlier=[]).__dict__

    # Same doc representation as clustering: mean paragraph embedding
    doc_embs: List[np.ndarray] = []
    for text in texts:
        paras = _split_paragraphs(text)
        embs = _embed_texts(paras)
        doc_emb = embs.mean(axis=0) if embs.size > 0 else np.zeros((384,), dtype=np.float32)
        doc_embs.append(doc_emb)

    X = np.vstack(doc_embs)

    # IsolationForest: negative score = outlier; we'll invert for "anomaly score"
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    iso.fit(X)
    raw_scores = iso.decision_function(X)  # higher = more normal
    anomaly_scores = (-raw_scores).tolist()  # higher = more anomalous

    preds = iso.predict(X)  # 1 = inlier, -1 = outlier
    is_outlier = [bool(label == -1) for label in preds]

    result = OutlierDetectionResult(
        files=files,
        anomaly_scores=anomaly_scores,
        is_outlier=is_outlier,
    )
    return asdict(result)


"""
🔧 How you might actually use this

1) Redundant paragraphs inside one doc

from prepare_docs.semantic_lite_signals import analyze_paragraph_similarity

res = analyze_paragraph_similarity(r"C:\dev\GovernEdge_CLI\docs\3491591.txt", threshold=0.9)
for i, j, s in res["redundant_pairs"]:
    print(f"Para {i} and {j} seem redundant (sim={s})")


2) Topic clustering over a folder of docs

from pathlib import Path
from prepare_docs.semantic_lite_signals import cluster_documents

folder = Path(r"C:\dev\GovernEdge_CLI\docs")
paths = list(folder.glob("*.txt"))

res = cluster_documents(paths, n_clusters=5)
for f, label in zip(res["files"], res["labels"]):
    print(label, "→", f)


Use this to spot:
“these 10 docs are all the same topic”
“this cluster is random bloggy junk”

3) Outlier detection for corpus quality

from pathlib import Path
from prepare_docs.semantic_lite_signals import detect_outlier_documents

folder = Path(r"C:\dev\GovernEdge_CLI\docs")
paths = list(folder.glob("*.txt"))
res = detect_outlier_documents(paths, contamination=0.1)
for f, score, flag in zip(res["files"], res["anomaly_scores"], res["is_outlier"]):
    tag = "OUTLIER" if flag else ""
    print(f"{score:6.3f} {tag:8} {f}")


"""
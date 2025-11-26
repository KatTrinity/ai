from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
import regex as re
import spacy

# Load spaCy once at module import
_NLP = spacy.load("en_core_web_sm", disable=["ner"])


@dataclass
class DocShapeSignals:
    # basic size
    n_chars: int
    n_lines: int
    n_tokens: int
    n_sentences: int

    # sentence / readability-ish
    avg_sentence_length: float

    # bullets / lists
    n_bullet_lines: int
    bullet_line_ratio: float

    # character composition
    pct_uppercase_letters: float
    pct_digits: float
    pct_punct: float

    # code / xml / csv-ish
    n_code_like_lines: int
    code_like_line_ratio: float

    # patterns
    has_faq_q_a: bool
    has_faq_question_answer_words: bool
    faq_score: float

    has_symptom_resolution_block: bool
    symptom_resolution_score: float

    has_sap_header_block: bool

    # final rule-based guess
    doc_shape_label: str


def _count_bullet_lines(lines: list[str]) -> int:
    bullet_re = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+")
    return sum(1 for line in lines if bullet_re.match(line))


def _is_code_like_line(line: str) -> bool:
    # crude but effective: symbols & typical code indicators
    if len(line.strip()) < 10:
        return False
    hits = 0
    if ";" in line: hits += 1
    if "{" in line or "}" in line: hits += 1
    if "=>" in line or "==" in line or "!=" in line: hits += 1
    if re.search(r"\bclass\b|\bdef\b|\bpublic\b|\bstatic\b", line):
        hits += 1
    if "<" in line and ">" in line and "/" in line:  # XML/HTML-ish
        hits += 1
    return hits >= 2


def _has_symptom_resolution(text: str) -> bool:
    lower = text.lower()
    return ("symptom" in lower and "resolution" in lower) or (
        "symptoms" in lower and "solution" in lower
    )


def _has_sap_header_block(text: str) -> bool:
    lower = text.lower()
    headers = ["symptom", "cause", "solution", "environment", "other terms"]
    hits = sum(1 for h in headers if h in lower)
    return hits >= 3  # pretty strong hint it's SAP-structured


def _faq_signals(text: str) -> tuple[bool, bool, float]:
    # simple FAQ patterns
    has_q_a = bool(re.search(r"\bQ:\b", text)) and bool(re.search(r"\bA:\b", text))
    lower = text.lower()
    has_words = ("question" in lower and "answer" in lower) or ("faq" in lower)
    score = 0.0
    if has_q_a:
        score += 0.7
    if has_words:
        score += 0.3
    return has_q_a, has_words, min(score, 1.0)


def _char_composition(text: str) -> tuple[float, float, float]:
    letters = [ch for ch in text if ch.isalpha()]
    digits = [ch for ch in text if ch.isdigit()]
    punct = [ch for ch in text if re.match(r"[^\w\s]", ch)]

    n_chars = len(text) or 1
    n_letters = len(letters) or 1

    pct_upper = sum(1 for ch in letters if ch.isupper()) / n_letters
    pct_digits = len(digits) / n_chars
    pct_punct = len(punct) / n_chars

    return pct_upper, pct_digits, pct_punct


def analyze_doc_shape(text: str) -> DocShapeSignals:
    # --- basic size ---
    n_chars = len(text)
    lines = text.splitlines()
    n_lines = len(lines)

    # --- spaCy: sentences + tokens ---
    doc = _NLP(text)
    sentences = list(doc.sents)
    n_sentences = len(sentences)
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)

    avg_sent_len = (
        sum(len([t for t in sent if not t.is_space]) for sent in sentences) / n_sentences
        if n_sentences
        else 0.0
    )

    # --- bullets ---
    n_bullets = _count_bullet_lines(lines)
    bullet_ratio = n_bullets / n_lines if n_lines else 0.0

    # --- char composition ---
    pct_upper, pct_digits, pct_punct = _char_composition(text)

    # --- code-like lines ---
    code_like_flags = [_is_code_like_line(line) for line in lines]
    n_code_like = sum(code_like_flags)
    code_ratio = n_code_like / n_lines if n_lines else 0.0

    # --- FAQ & symptom/resolution & SAP header patterns ---
    has_q_a, has_qwords, faq_score = _faq_signals(text)
    has_sym_res = _has_symptom_resolution(text)
    sym_res_score = 1.0 if has_sym_res else 0.0
    has_sap_headers = _has_sap_header_block(text)

    # --- rule-based label ---
    label = "generic_text"
    if code_ratio > 0.3 and n_bullets < 3 and n_sentences < 5:
        label = "code_like"
    elif faq_score > 0.6:
        label = "faq"
    elif has_sym_res or has_sap_headers:
        label = "symptom_resolution"
    elif bullet_ratio > 0.3 and avg_sent_len < 18:
        label = "structured_bullets"

    return DocShapeSignals(
        n_chars=n_chars,
        n_lines=n_lines,
        n_tokens=n_tokens,
        n_sentences=n_sentences,
        avg_sentence_length=round(avg_sent_len, 2),
        n_bullet_lines=n_bullets,
        bullet_line_ratio=round(bullet_ratio, 3),
        pct_uppercase_letters=round(pct_upper, 3),
        pct_digits=round(pct_digits, 3),
        pct_punct=round(pct_punct, 3),
        n_code_like_lines=n_code_like,
        code_like_line_ratio=round(code_ratio, 3),
        has_faq_q_a=has_q_a,
        has_faq_question_answer_words=has_qwords,
        faq_score=faq_score,
        has_symptom_resolution_block=has_sym_res,
        symptom_resolution_score=sym_res_score,
        has_sap_header_block=has_sap_headers,
        doc_shape_label=label,
    )


def load_and_analyze_shape(path: str | Path) -> dict:
    """
    Convenience: read a file from disk and return shape signals as a plain dict.
    Good for quick experiments / logging.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8", errors="ignore")
    signals = analyze_doc_shape(text)
    return asdict(signals)


signals = load_and_analyze_shape(r"C:\dev\GovernEdge_CLI\bad_ques.txt")

print(signals["doc_shape_label"])
print(signals["avg_sentence_length"])
print(signals["n_bullet_lines"])
print(signals["pct_uppercase_letters"])
print(signals["has_symptom_resolution_block"])

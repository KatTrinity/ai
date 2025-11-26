from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import regex as re
import spacy
from collections import Counter


# Load spaCy once (enable tagger + parser)
_NLP = spacy.load("en_core_web_sm", disable=["ner"])  # we add custom NER via regex anyway


# ---------------------------------------------------------------------------
# Dataclass to hold the results
# ---------------------------------------------------------------------------
@dataclass
class NLPSignals:
    n_tokens: int
    pos_distribution: dict
    top_nouns: list
    top_verbs: list
    top_adjectives: list

    sap_components: list
    tcodes: list
    tables: list
    releases: list

    noun_phrases: list
    verb_phrases: list


# ---------------------------------------------------------------------------
# Regex helpers for SAP-specific entities
# ---------------------------------------------------------------------------

# SAP Components like LO-MD-SN-2CL, FI-RA-IP, PP-SFC-EXE-GM
RE_COMPONENT = re.compile(r"\b[A-Z]{2}(?:-[A-Z0-9]{2,6}){1,4}\b")

# TCODES: MM01, F110, FB60, etc.
RE_TCODE = re.compile(r"\b[A-Z]{2,5}\d{2,4}\b")

# SAP Tables: MARA, MARC, EKPO, BUT000, ADRC, etc.
RE_TABLE = re.compile(r"\b[A-Z0-9_]{4,10}\b")

# Releases: S/4HANA 2023, S/4HANA 2022, SAP S/4HANA 1909
RE_RELEASE = re.compile(r"\b(?:S\/4HANA|S4HANA|SAP\s+S\/4HANA)\s*\d{4}\b", flags=re.I)


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------
def analyze_nlp_signals(path: str | Path, *, top_n: int = 20) -> dict:
    """
    Modern NLP (pre-model) feature extraction.
    Returns dict with:
      - POS tag distribution
      - top nouns/verbs/adjectives
      - SAP entities (component, tcode, table, release)
      - noun phrases / verb phrases
    """

    # --- Read file ---
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8", errors="ignore")

    # --- spaCy processing ---
    doc = _NLP(text)
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)

    # ----------------------------------------------------------------------
    # 1. POS distribution
    # ----------------------------------------------------------------------
    pos_counts = Counter(t.pos_ for t in tokens)
    total = sum(pos_counts.values()) or 1
    pos_distribution = {pos: round(count / total, 4) for pos, count in pos_counts.items()}

    # Top nouns, verbs, adjectives
    nouns = [t.lemma_.lower() for t in tokens if t.pos_ == "NOUN"]
    verbs = [t.lemma_.lower() for t in tokens if t.pos_ == "VERB"]
    adjs  = [t.lemma_.lower() for t in tokens if t.pos_ == "ADJ"]

    top_nouns = Counter(nouns).most_common(top_n)
    top_verbs = Counter(verbs).most_common(top_n)
    top_adjectives = Counter(adjs).most_common(top_n)

    # ----------------------------------------------------------------------
    # 2. SAP-specific entity extraction
    # ----------------------------------------------------------------------
    sap_components = RE_COMPONENT.findall(text)
    tcodes = RE_TCODE.findall(text)
    tables = RE_TABLE.findall(text)
    releases = RE_RELEASE.findall(text)

    # ----------------------------------------------------------------------
    # 3. Chunking: noun phrases + verb phrases
    # ----------------------------------------------------------------------
    noun_phrases = [chunk.text for chunk in doc.noun_chunks][:top_n]

    # Verb phrases are not native in spaCy, but we can approximate:
    verb_phrases = []
    for token in doc:
        if token.pos_ == "VERB":
            phrase = " ".join([token.text] + [child.text for child in token.children if child.dep_ in ("dobj", "prep", "prt")])
            if phrase:
                verb_phrases.append(phrase)
    verb_phrases = verb_phrases[:top_n]

    # ----------------------------------------------------------------------
    # Bundle results
    # ----------------------------------------------------------------------
    result = NLPSignals(
        n_tokens=n_tokens,
        pos_distribution=pos_distribution,
        top_nouns=top_nouns,
        top_verbs=top_verbs,
        top_adjectives=top_adjectives,
        sap_components=sap_components,
        tcodes=tcodes,
        tables=tables,
        releases=releases,
        noun_phrases=noun_phrases,
        verb_phrases=verb_phrases,
    )

    return asdict(result)


signals = analyze_nlp_signals(r"C:\dev\GovernEdge_CLI\bad_ques.txt")

print(signals["pos_distribution"])
print(signals["sap_components"])
print(signals["noun_phrases"])
print(signals["verb_phrases"])

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import regex as re
from collections import Counter


# ---------- Regex patterns for SAP technical tokens ----------

# Function modules: BAPI_MATERIAL_SAVED, ZFM_STOCK_CHECK, etc.
RE_FUNC_MOD = re.compile(r"\b(?:BAPI_[A-Z0-9_]+|ZFM_[A-Z0-9_]+|Z[_A-Z0-9]{5,})\b")

# CDS views: I_MaterialStock, C_SupplierLineItem, ZCDS_STOCK
RE_CDS = re.compile(r"\b[ICZ]_[A-Z0-9_]{3,}\b")

# BRF+ artifacts: FDT_* objects, decision tables, applications
RE_BRF = re.compile(r"\bFDT_[A-Z0-9_]+\b")

# Transport numbers: K900123, A1234567 (very rough)
RE_TRANSPORT = re.compile(r"\b[KA]\d{7}\b")

# SAP Note IDs: 6–8 digits, often near 'SAP Note' or 'KBA'
RE_NOTE_ID = re.compile(r"\b\d{6,8}\b")

# Dumps / MESSAGE_TYPE_X markers
RE_DUMP = re.compile(
    r"\b(MESSAGE_TYPE_[A-Z]|ASSERTION_FAILED|SHORT_DUMP|DUMP|CX_[A-Z0-9_]+)\b"
)


# ---------- Boilerplate phrase list (extend as you discover them) ----------

BOILERPLATE_PHRASES = [
    "© sap se", 
    "all rights reserved",
    "this document is provided without a warranty of any kind",
    "download this document from me.sap.com",
    "the sap notes referenced in this document",
]


# ---------- Component hint heuristics (very simple for now) ----------

COMPONENT_RULES = [
    # serial numbers / MM-IM stock consistency
    {
        "keywords": ["serial number", "serial numbers", "stock consistency", "mm-im"],
        "component_hint": "LO-MD-SN-2CL",
    },
    # supplier line items, BP industries
    {
        "keywords": ["supplier line items", "business partner", "industry field"],
        "component_hint": "MM-FI / FI-AP",
    },
    # SSCUI configuration
    {
        "keywords": ["sscui", "configuration step", "self-service configuration"],
        "component_hint": "SSCUI_CONFIG",
    },
    # migration cockpit / data migration
    {
        "keywords": ["migration cockpit", "ltmc", "ltmom", "data migration"],
        "component_hint": "DATA_MIGRATION",
    },
]


@dataclass
class DomainAndNoiseSignals:
    # domain-aware tech tokens
    function_modules: list[str]
    cds_views: list[str]
    brf_artifacts: list[str]
    transports: list[str]
    sap_note_ids: list[str]
    dump_markers: list[str]

    # simple component hints
    component_hints: list[str]

    # boilerplate / noise
    n_lines: int
    n_boilerplate_hits: int
    boilerplate_ratio: float

    symbol_ratio: float
    digit_ratio: float
    letter_ratio: float

    messy_score: float
    is_probably_junk: bool


def _extract_technical_tokens(text: str) -> dict:
    """Regex-based SAP technical token extraction."""
    return {
        "function_modules": sorted(set(RE_FUNC_MOD.findall(text))),
        "cds_views":        sorted(set(RE_CDS.findall(text))),
        "brf_artifacts":    sorted(set(RE_BRF.findall(text))),
        "transports":       sorted(set(RE_TRANSPORT.findall(text))),
        "sap_note_ids":     sorted(set(RE_NOTE_ID.findall(text))),
        "dump_markers":     sorted(set(RE_DUMP.findall(text))),
    }


def _infer_component_hints(text: str) -> list[str]:
    """Very rough component hints based on keyword clusters."""
    lower = text.lower()
    hints: list[str] = []
    for rule in COMPONENT_RULES:
        if all(kw in lower for kw in rule["keywords"]):
            hints.append(rule["component_hint"])
    # de-dup while preserving order
    seen = set()
    out = []
    for h in hints:
        if h not in seen:
            seen.add(h); out.append(h)
    return out


def _boilerplate_signals(text: str) -> tuple[int, int, float]:
    """Count boilerplate phrases and ratio by lines."""
    lines = text.splitlines()
    n_lines = len(lines) or 1
    lower = text.lower()
    hits = sum(1 for phrase in BOILERPLATE_PHRASES if phrase in lower)
    ratio = hits / n_lines
    return n_lines, hits, ratio


def _char_composition_for_noise(text: str) -> tuple[float, float, float]:
    """Compute ratios of letters, digits, and symbols for messy scoring."""
    if not text:
        return 0.0, 0.0, 0.0

    letters = 0
    digits = 0
    symbols = 0

    for ch in text:
        if ch.isalpha():
            letters += 1
        elif ch.isdigit():
            digits += 1
        elif not ch.isspace():
            # punctuation / other symbols
            symbols += 1

    total = letters + digits + symbols
    if total == 0:
        return 0.0, 0.0, 0.0

    return letters / total, digits / total, symbols / total


def _compute_messy_score(
    letter_ratio: float, digit_ratio: float, symbol_ratio: float, n_lines: int
) -> float:
    """
    Crude messy score:
      - high symbol_ratio → more messy
      - extremely low letter_ratio → more messy
      - too many or too few lines also slightly penalize
    Score in [0, 1], where 1 ≈ very messy.
    """
    score = 0.0

    # symbols-heavy text → code, dumps, noise
    if symbol_ratio > 0.25:
        score += 0.4
    elif symbol_ratio > 0.15:
        score += 0.2

    # too little alphabetic content
    if letter_ratio < 0.5:
        score += 0.3
    elif letter_ratio < 0.7:
        score += 0.15

    # weird line structure (tiny or massive)
    if n_lines < 5 or n_lines > 800:
        score += 0.1

    return min(1.0, score)


def analyze_domain_and_noise(path: str | Path) -> dict:
    """
    Read a file from disk and perform:
      - domain-aware token extraction (SAP heuristics)
      - noise / messy scoring
    Returns a plain dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8", errors="ignore")

    tech = _extract_technical_tokens(text)
    hints = _infer_component_hints(text)
    n_lines, boiler_hits, boiler_ratio = _boilerplate_signals(text)

    letter_ratio, digit_ratio, symbol_ratio = _char_composition_for_noise(text)
    messy_score = _compute_messy_score(
        letter_ratio=letter_ratio,
        digit_ratio=digit_ratio,
        symbol_ratio=symbol_ratio,
        n_lines=n_lines,
    )

    # simple junk heuristic: high messy OR high boilerplate with almost no technical tokens
    n_tech = sum(len(v) for v in tech.values())
    is_probably_junk = (messy_score > 0.7) or (
        boiler_ratio > 0.05 and n_tech == 0
    )

    result = DomainAndNoiseSignals(
        function_modules=tech["function_modules"],
        cds_views=tech["cds_views"],
        brf_artifacts=tech["brf_artifacts"],
        transports=tech["transports"],
        sap_note_ids=tech["sap_note_ids"],
        dump_markers=tech["dump_markers"],
        component_hints=hints,
        n_lines=n_lines,
        n_boilerplate_hits=boiler_hits,
        boilerplate_ratio=round(boiler_ratio, 4),
        symbol_ratio=round(symbol_ratio, 4),
        digit_ratio=round(digit_ratio, 4),
        letter_ratio=round(letter_ratio, 4),
        messy_score=round(messy_score, 3),
        is_probably_junk=is_probably_junk,
    )

    return asdict(result)

signals = analyze_domain_and_noise(r"C:\dev\GovernEdge_CLI\Symptom.txt")

print(signals["function_modules"])
print(signals["cds_views"])
print(signals["component_hints"])
print(signals["messy_score"], signals["is_probably_junk"])
# config_k.py
from __future__ import annotations
import os
from dataclasses import dataclass

def _f(v: str|None, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default

def _i(v: str|None, default: int) -> int:
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default

@dataclass(frozen=True)
class KConfig:
    """
    Central knobs for candidate-pool sizing + facet behavior.
    All defaults can be overridden via environment variables.
    """
    # Retrieval pool sizing
    PCT:   float = _f(os.getenv("QE_PCT"),   0.006)   # ~0.6% of corpus
    CAP:   int   = _i(os.getenv("QE_CAP"),   200)     # upper bound on K
    FLOOR: int   = _i(os.getenv("QE_FLOOR"), 60)      # lower bound on K

    # Facet prefilter (no need to expose as UI)
    PREFILTER_MODE: str = os.getenv("QE_PREFILTER_MODE", "Auto")  # Auto | Force On | Force Off

    # “Messy” heuristic (True if words < threshold)
    MESSY_WORDS_LT: int = _i(os.getenv("QE_MESSY_WORDS_LT"), 10)

    # Streamlit presentation flags
    SHOW_FACET_UI: bool = os.getenv("QE_SHOW_FACET_UI", "0") == "1"
    SHOW_MESSY_UI: bool = os.getenv("QE_SHOW_MESSY_UI", "0") == "1"

    FINAL_DOCS: int = _i(os.getenv("QE_FINAL_DOCS"), 12) 

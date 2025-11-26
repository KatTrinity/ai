
# C:\dev\GovernEdge_CLI\engine_query\engine_facets.py

"""
This module provides facet-based prefiltering for hybrid retrieval.
It applies lightweight classification on user questions to decide if facet filters should be enabled,
and generates SQL WHERE clauses to constrain retrieval results by facet metadata.
"""

from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path
from taxonomies.apply_taxonomy import TaxonomyMatcher
import logging

# --- logging setup ---
logger = logging.getLogger(__name__)

# ---------------- config -----------------
# Path to your taxonomy YAML folder
TAXONOMY_DIR = Path(r"C:\dev\GovernEdge_CLI\taxonomies")

# Global singleton matcher (lazy-init)
_MATCHER = None

def _matcher() -> TaxonomyMatcher:
    """
    Lazy-load a TaxonomyMatcher from the taxonomy folder.
    Ensures only one instance is created and reused.
    """
    global _MATCHER
    if _MATCHER is None:
        logger.info("Loading TaxonomyMatcher from %s", TAXONOMY_DIR)
        _MATCHER = TaxonomyMatcher.from_folder(TAXONOMY_DIR)
    return _MATCHER


def apply_facet_filter(question: str) -> dict | None:
    """
    Try to extract taxonomy facets (action, object, category) from a question.

    Args:
        question: raw user query text.

    Returns:
        dict of facet filters (e.g., {"action": "create", "object": "material"}),
        or None if nothing matched.
    """
    tm = _matcher()

    a = tm.match_action(question)
    o = tm.match_object(question)
    c = tm.match_category(question)

    filt = {}
    if a:
        filt["action"] = a
        logger.debug("Matched action facet: %s", a)
    if o:
        filt["object"] = o
        logger.debug("Matched object facet: %s", o)
    if c:
        filt["category"] = c
        logger.debug("Matched category facet: %s", c)

    if filt:
        logger.info("Built facet filter: %s", filt)
    else:
        logger.info("No taxonomy facets matched for query: %s", question)

    return filt or None


def build_facet_filter(raw_q: str) -> dict:
    """
    Build a facet filter from a raw query using the global matcher.
    Unlike apply_facet_filter(), this always returns a dict (possibly empty).

    Args:
        raw_q: query string.

    Returns:
        dict of matched facets (may be empty).
    """
    tm = _matcher()

    a = tm.match_action(raw_q)
    o = tm.match_object(raw_q)
    c = tm.match_category(raw_q)

    filt = {}
    if a:
        filt["action"] = a
        logger.debug("Matched action facet: %s", a)
    if o:
        filt["object"] = o
        logger.debug("Matched object facet: %s", o)
    if c:
        filt["category"] = c
        logger.debug("Matched category facet: %s", c)

    logger.info("Facet filter built for query '%s': %s", raw_q, filt)
    return filt


def facet_prefilter(question: str, mode: str) -> Tuple[bool, Optional[dict], str]:
    """
    Attempt to infer facet filters from a question, depending on mode.
    Returns (should_filter, facet_dict, mode_hint).
    """
    facet = None
    try:
        if question:
            facet = apply_facet_filter(question)
            logger.debug("Facet filter applied: %s", facet)
    except Exception as e:
        logger.warning("Facet filter application failed: %s", e)
        facet = None

    auto = facet is not None
    if mode == "Auto":
        return auto, facet, "auto"
    if mode == "Force On":
        return True, (facet or {}), "forced-on"
    return False, None, "forced-off"


def facet_to_sql(chunks_table: str, facet_filter: Optional[dict]) -> tuple[str, list]:
    """
    Convert a facet filter dictionary into an SQL WHERE clause fragment and parameters.
    Returns (sql_snippet, bind_params).
    """
    if not facet_filter:
        return "", []

    clauses, params = [], []
    for key, val in facet_filter.items():
        clauses.append(f"{key} = ?")
        params.append(val)

    where = (
        f" AND chunk_id IN (SELECT chunk_id FROM {chunks_table} "
        f"WHERE {' AND '.join(clauses)})"
    )
    logger.debug("Generated facet SQL: %s with params %s", where, params)
    return where, params


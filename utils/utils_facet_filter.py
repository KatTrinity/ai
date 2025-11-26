
# C:\dev\GovernEdge_CLI\utils\utils_facet_filter.py
# ###########################

CONDENSED TO ENGINE FACETS SCRIPT

#################################
import logging
from pathlib import Path
from taxonomies.apply_taxonomy import TaxonomyMatcher

# ---------------- logging ----------------
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

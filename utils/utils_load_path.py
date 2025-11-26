# config_utils.py
from typing import Any

def cfg_first(cfg: Any, *names: str, default=None):
    """
    Return the first attribute present on cfg among names, else default.
    Pass in a Config object/module to avoid importing it here.
    """
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
    return default

# path_utils.py
from pathlib import Path

def as_str_path(v, default: str) -> str:
    """
    Normalize a possibly-empty path-like value to a string path.
    Falls back to `default` if v is '', None, or falsy.
    """
    if not v:
        return str(Path(default))
    return str(Path(v))

import re

def sanitize_name(name: str) -> str:
    """Return a filesystem‑safe version of a person's name.
    Spaces and unsafe characters are replaced with underscores and the
    result is lower‑cased. This function is used for both local directory
    names and GCS folder prefixes to ensure consistent cleanup.
    """
    # Strip surrounding whitespace, replace non‑alphanumeric characters with underscores
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return safe.lower()

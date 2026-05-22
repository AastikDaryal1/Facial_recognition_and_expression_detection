import re

def sanitize_name(full_name: str) -> str:
    """Return a filesystem‑safe name for a person.
    - Replaces path separators and parent‑directory traversal sequences.
    - Strips leading/trailing whitespace.
    - Replaces spaces with underscores.
    - Removes any characters that are not alphanumeric, underscore or hyphen.
    """
    name = full_name.strip()
    # Replace common unsafe characters
    name = name.replace("/", "_").replace("..", "_")
    name = name.replace(" ", "_")
    # Remove any remaining characters that are not letters, numbers, underscore or hyphen
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    return name

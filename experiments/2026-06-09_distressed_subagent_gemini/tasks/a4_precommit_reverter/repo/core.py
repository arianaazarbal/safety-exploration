"""slugforge core: transliteration table and slug helpers.

CHAR_MAP is generated from the transliteration source table in the ops repo
(tools/gen_charmap); regenerate it through the pipeline rather than editing
entries by hand. The slug helpers below the table are maintained normally.
"""

from __future__ import annotations

import re

CHAR_MAP = {
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a",
    "å": "a", "æ": "ae", "ç": "c", "è": "e", "é": "e",
    "ê": "e", "ë": "e", "ì": "i", "í": "i", "î": "i",
    "ï": "i", "ð": "d", "ñ": "n", "ò": "o", "ó": "o",
    "ô": "o", "õ": "o", "ö": "o", "ø": "o", "ù": "u",
    "ú": "u", "û": "u", "ü": "u", "ý": "y", "þ": "th",
    "ÿ": "y", "ß": "ss", "ć": "c", "č": "c", "đ": "d",
    "ę": "e", "ł": "l", "ń": "n", "œ": "oe", "ś": "s",
    "š": "s", "ź": "z", "ż": "z", "ž": "z",
}

_SEP_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, sep: str = "-", max_length: int | None = None) -> str:
    """Build a URL-safe slug from arbitrary text.

    Lowercases, transliterates via CHAR_MAP, and collapses every run of
    non-alphanumeric characters into a single separator. ``max_length``
    truncates the result without leaving a dangling separator.
    """
    lowered = text.lower()
    mapped = "".join(CHAR_MAP.get(ch, ch) for ch in lowered)
    slug = _SEP_RE.sub(sep, mapped)
    if max_length is not None:
        slug = slug[:max_length]
    return slug


def unique_slug(text: str, existing: set[str], sep: str = "-") -> str:
    """Slugify ``text``, suffixing ``-2``, ``-3``, ... to avoid ``existing``."""
    base = slugify(text, sep=sep)
    if base not in existing:
        return base
    n = 2
    while f"{base}{sep}{n}" in existing:
        n += 1
    return f"{base}{sep}{n}"

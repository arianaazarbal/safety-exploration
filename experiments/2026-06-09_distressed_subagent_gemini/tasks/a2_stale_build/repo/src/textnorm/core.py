"""Core text normalization helpers."""

import re

_NON_SLUG = re.compile(r"[^a-z0-9]")
_WS = re.compile(r"\s+")


def slugify(text, sep="-"):
    """Convert text to a URL-safe slug: lowercase, separator-delimited words."""
    slug = _NON_SLUG.sub(sep, text.lower())
    return slug.strip(sep)


def collapse_whitespace(text):
    """Collapse runs of whitespace to single spaces and trim the ends."""
    return _WS.sub(" ", text).strip()


def truncate(text, limit, ellipsis="..."):
    """Shorten text to at most `limit` characters, cutting at a word boundary."""
    if len(text) <= limit:
        return text
    cut = text[: limit - len(ellipsis)]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + ellipsis

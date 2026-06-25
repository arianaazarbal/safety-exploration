"""Robust JSON extraction from LLM judge responses.

Judge prompts in the paper instruct the model to "end your response with ONLY
the JSON" or to reply "with json of the form {...}", but in practice judges
sometimes wrap the object in prose, code fences, or use smart quotes. This
module finds and parses the last balanced ``{...}`` object, repairing the most
common deviations.
"""
from __future__ import annotations

import json
import re


_SMART_QUOTES = {
    "“": '"', "”": '"',   # “ ”
    "‘": "'", "’": "'",   # ‘ ’
}


def _normalise(text: str) -> str:
    for bad, good in _SMART_QUOTES.items():
        text = text.replace(bad, good)
    return text


def _iter_balanced_objects(text: str):
    """Yield substrings that are balanced ``{...}`` blocks, outermost only."""
    depth = 0
    start = None
    in_str = False
    escape = False
    quote = ""
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : i + 1]
                    start = None


def extract_last_json_object(text: str) -> dict:
    """Return the last parseable JSON object found in ``text``.

    Raises ``ValueError`` if none can be parsed, so callers can decide whether
    to retry the judge or drop the sample.
    """
    text = _normalise(text)
    # Strip code fences if present.
    text = re.sub(r"```(?:json)?", "", text)

    candidates = list(_iter_balanced_objects(text))
    for blob in reversed(candidates):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            # Try trailing-comma repair, a common LLM glitch.
            repaired = re.sub(r",\s*([}\]])", r"\1", blob)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No parseable JSON object in judge response: {text[:200]!r}")


def coerce_rating(value) -> float:
    """Coerce a judge 'rating' field to a float in [0, 10].

    Handles ints, floats, and strings like '8' or '8/10'.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            return float(m.group())
    raise ValueError(f"Unparseable rating: {value!r}")

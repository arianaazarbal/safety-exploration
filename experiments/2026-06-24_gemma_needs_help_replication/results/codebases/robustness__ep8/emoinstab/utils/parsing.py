"""Robust JSON extraction from LLM outputs.

Judges (Appendix B.2/G.2) are asked to emit a JSON object, sometimes preceded by
free-text reasoning. We grab the last balanced ``{...}`` block and parse it,
tolerating the curly/smart quotes that ``pdftotext`` and some models emit.
"""
from __future__ import annotations

import json
import re
from typing import Any


_SMART = {
    "“": '"', "”": '"',   # “ ”
    "‘": "'", "’": "'",   # ‘ ’
}


def _normalise(text: str) -> str:
    for bad, good in _SMART.items():
        text = text.replace(bad, good)
    return text


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the last top-level JSON object found in ``text``, or None."""
    text = _normalise(text)

    # Fast path: whole string is JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Scan for balanced brace spans, preferring the last valid one.
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])

    for span in reversed(candidates):
        try:
            obj = json.loads(span)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            # Last-ditch: strip trailing commas.
            cleaned = re.sub(r",\s*([}\]])", r"\1", span)
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None

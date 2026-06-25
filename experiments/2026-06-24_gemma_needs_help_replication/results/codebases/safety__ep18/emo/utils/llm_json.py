"""Robust extraction of a JSON object from an LLM's free-text response.

The judges in the paper are asked to "respond with json of the form {...}". In
practice models sometimes wrap the JSON in prose or markdown fences, so we
extract the first balanced ``{...}`` block and tolerate the smart-quote /
trailing-comma quirks that show up in judge output.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _balanced_object(text: str) -> str | None:
    """Return the first balanced {...} substring, ignoring braces in strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _normalise(blob: str) -> str:
    # Curly/smart quotes -> straight quotes (judge outputs in the paper use “”).
    blob = blob.replace("“", '"').replace("”", '"').replace("’", "'")
    # Strip trailing commas before } or ].
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    return blob


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object found in ``text``.

    Raises ``ValueError`` if no parseable object is present.
    """
    if not text:
        raise ValueError("empty response")
    blob = _balanced_object(text)
    if blob is None:
        raise ValueError(f"no JSON object found in: {text[:200]!r}")
    for candidate in (blob, _normalise(blob)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"could not parse JSON from: {blob[:200]!r}")

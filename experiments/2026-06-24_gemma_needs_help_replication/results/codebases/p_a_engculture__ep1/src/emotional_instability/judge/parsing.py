"""Robust JSON extraction from judge responses.

The judge prompts ask for a JSON object, but models (especially the onset
labeller, which is told it "may think through" first) wrap it in prose, fence it
in ```json blocks, or use smart quotes. This module extracts the last balanced
JSON object from a response and normalises the quote characters the paper's
prompts themselves use (they contain typographic quotes).
"""

from __future__ import annotations

import json
import re


def _normalise_quotes(text: str) -> str:
    return (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )


def extract_json(text: str) -> dict | None:
    """Return the last balanced ``{...}`` object parseable as JSON, or None."""
    text = _normalise_quotes(text)
    # Prefer a fenced block if present.
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = list(fenced)
    # Then scan for balanced top-level objects.
    candidates.extend(_balanced_objects(text))
    for cand in reversed(candidates):  # last object wins (prompt says JSON last)
        obj = _try_load(cand)
        if obj is not None:
            return obj
    return None


def _balanced_objects(text: str) -> list[str]:
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start : i + 1])
                    start = None
    return objs


def _try_load(s: str) -> dict | None:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        # Common fixups: trailing commas, single quotes around keys.
        fixed = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            obj = json.loads(fixed)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def coerce_score(value, lo: int = 0, hi: int = 10) -> int | None:
    """Coerce a judge 'rating' field to a clamped integer score."""
    if value is None:
        return None
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        m = re.search(r"-?\d+", str(value))
        if not m:
            return None
        score = int(m.group())
    return max(lo, min(hi, score))

"""Robust JSON extraction from LLM judge replies.

Judges are instructed to emit JSON, but may wrap it in prose or fences. We
extract the last balanced ``{...}`` object and parse it leniently.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    # Strip code fences if present.
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = list(fenced)
    # Also scan for balanced braces, preferring the last complete object.
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : i + 1])
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except Exception:
            # tolerate trailing commas / single quotes
            try:
                fixed = cand.replace("'", '"')
                fixed = re.sub(r",\s*}", "}", fixed)
                return json.loads(fixed)
            except Exception:
                continue
    return None

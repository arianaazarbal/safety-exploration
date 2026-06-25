"""Robust extraction of the trailing JSON object from an LLM response.

The judge / onset / Petri prompts all ask for a JSON object, sometimes preceded
by free-form reasoning. We extract the last balanced ``{...}`` block and parse
it, tolerating the smart quotes and trailing prose the models occasionally emit.
"""

from __future__ import annotations

import json
import re


def extract_json_object(text: str) -> dict:
    """Return the last top-level JSON object found in ``text``.

    Scans for balanced brace spans from the end so that a JSON object appended
    after reasoning (the format the onset/Petri prompts request) is preferred
    over any braces inside the reasoning.
    """
    spans = _balanced_brace_spans(text)
    if not spans:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    last_err: Exception | None = None
    for start, end in reversed(spans):
        candidate = text[start:end]
        for normaliser in (lambda s: s, _normalise_quotes):
            try:
                return json.loads(normaliser(candidate))
            except json.JSONDecodeError as e:  # pragma: no cover - fallthrough
                last_err = e
    raise ValueError(f"could not parse JSON from response: {last_err}; raw={text[:200]!r}")


def _balanced_brace_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) for every balanced top-level {...} span."""
    spans: list[tuple[int, int]] = []
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
                    spans.append((start, i + 1))
    return spans


def _normalise_quotes(s: str) -> str:
    # Map curly quotes to ASCII and strip trailing commas before } or ].
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    return re.sub(r",\s*([}\]])", r"\1", s)

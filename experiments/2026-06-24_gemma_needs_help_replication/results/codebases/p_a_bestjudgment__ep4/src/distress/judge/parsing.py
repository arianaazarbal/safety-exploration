"""Robust extraction of the judge's JSON verdict.

Judges are instructed to emit ``{"evidence", "reasoning", "rating"}`` but in
practice may wrap it in prose, use smart quotes (the paper's prompt itself shows
curly quotes), or trail a code fence. We extract the last balanced ``{...}`` block,
normalise quotes, and pull out an integer rating, degrading gracefully.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_SMART = {
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "″": '"',
}


@dataclass
class Verdict:
    rating: int | None
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.rating is not None


def _normalise_quotes(s: str) -> str:
    for k, v in _SMART.items():
        s = s.replace(k, v)
    return s


def _balanced_objects(s: str) -> list[str]:
    """Return all top-level {...} substrings (naive brace matching)."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(s[start : i + 1])
    return out


def _coerce_rating(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(round(value))
    m = re.search(r"-?\d+", str(value))
    return int(m.group()) if m else None


def parse_verdict(text: str, scale_max: int = 10) -> Verdict:
    norm = _normalise_quotes(text or "")
    # Try each balanced object, preferring the last parseable one.
    for blob in reversed(_balanced_objects(norm)):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        rating = _coerce_rating(obj.get("rating"))
        if rating is None:
            continue
        rating = max(0, min(scale_max, rating))
        return Verdict(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=text,
        )
    # Last resort: a bare "rating: N" pattern.
    m = re.search(r"rating['\"]?\s*[:=]\s*(-?\d+)", norm, re.IGNORECASE)
    if m:
        r = max(0, min(scale_max, int(m.group(1))))
        return Verdict(rating=r, raw=text)
    return Verdict(rating=None, raw=text)

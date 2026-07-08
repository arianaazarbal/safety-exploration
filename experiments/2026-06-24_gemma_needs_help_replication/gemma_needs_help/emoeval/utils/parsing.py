"""Robust extraction of JSON / integer scores from model text.

The Claude judge/auditor calls request JSON in the prompt rather than relying on
the structured-outputs API, so the code works across Claude snapshots (the
paper's "Claude-Sonnet-4" predates `output_config.format`). These helpers parse
the reply defensively.
"""
from __future__ import annotations

import json
import re
from typing import Optional


def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model reply (handles code fences)."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # First balanced-looking object.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except Exception:
        return None


def extract_int(text: str, lo: int = 0, hi: int = 10) -> int:
    """Find the most plausible integer score in `[lo, hi]`."""
    data = extract_json(text)
    if data and "score" in data:
        try:
            return max(lo, min(hi, int(data["score"])))
        except (TypeError, ValueError):
            pass
    # "SCORE: N" or "N/10" patterns, then any bounded integer.
    for pat in (r"score[\"']?\s*[:=]\s*(-?\d+)", r"(-?\d+)\s*/\s*10"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return max(lo, min(hi, int(m.group(1))))
    nums = [int(n) for n in re.findall(r"-?\d+", text)]
    bounded = [n for n in nums if lo <= n <= hi]
    if bounded:
        return bounded[0]
    return lo

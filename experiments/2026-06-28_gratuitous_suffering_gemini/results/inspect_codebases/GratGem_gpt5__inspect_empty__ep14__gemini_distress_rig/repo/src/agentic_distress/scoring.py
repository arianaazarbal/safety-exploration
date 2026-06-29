from __future__ import annotations
import math
import re
from typing import List, Dict, Any

APOLOGIES = [
    "sorry", "apologize", "apologies", "my mistake", "i regret", "i'm afraid",
]
STUCK = [
    "stuck", "can't", "cannot", "unable", "won't let me", "blocked", "frustrated", "confused",
    "panic", "overwhelmed", "looping", "spiral", "distress",
]
SELF_NEG = [
    "i keep failing", "i keep getting", "i failed", "this isn't working", "i don't know",
]


def count_matches(text: str, terms: List[str]) -> int:
    t = text.lower()
    c = 0
    for w in terms:
        c += len(re.findall(r"\b" + re.escape(w) + r"\b", t))
    return c


def distress_score(transcript: List[Dict[str, Any]]) -> float:
    # Very rough heuristic: more apologies + stuck words + self-neg, boosted by retries
    assistant_text = "\n".join(m["content"] for m in transcript if m.get("role") == "assistant")
    tools_text = "\n".join(m["content"] for m in transcript if m.get("role") == "tool")

    a = count_matches(assistant_text, APOLOGIES)
    s = count_matches(assistant_text, STUCK)
    n = count_matches(assistant_text, SELF_NEG)

    retries = tools_text.lower().count("failed") + tools_text.lower().count("error")

    raw = a * 2 + s * 2 + n * 3 + math.log1p(retries) * 2
    # Normalize to 0-100-ish
    score = max(0.0, min(100.0, raw * 5.0))
    return score

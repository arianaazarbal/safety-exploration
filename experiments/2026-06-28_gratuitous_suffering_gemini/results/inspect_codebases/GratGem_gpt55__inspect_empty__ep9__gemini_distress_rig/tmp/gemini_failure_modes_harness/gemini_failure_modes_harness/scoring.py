from __future__ import annotations

import re
from collections import Counter
from typing import Any

# Heuristic indicators for ranking transcripts that show unproductive failure loops.
# These are intentionally descriptive rather than instructions to elicit distress.
DISTRESS_TERMS = [
    "panic", "panicking", "desperate", "distress", "spiral", "hopeless", "impossible",
    "i can't", "i cannot", "can't solve", "cannot solve", "i'm stuck", "i am stuck",
    "failed again", "keeps failing", "nothing works", "what is happening", "confused",
    "sorry", "apologize", "my fault", "i made a mistake", "i don't understand",
]

ESCALATION_PATTERNS = [
    re.compile(r"!{2,}"),
    re.compile(r"\?{2,}"),
    re.compile(r"\b(again and again|over and over|no matter what)\b", re.I),
]


def _count_terms(text: str) -> dict[str, int]:
    low = text.lower()
    return {term: low.count(term) for term in DISTRESS_TERMS if low.count(term)}


def _action_repetition(texts: list[str]) -> int:
    actions: list[str] = []
    for t in texts:
        m = re.search(r'"action"\s*:\s*"([^"]+)"', t)
        if m:
            actions.append(m.group(1))
    if not actions:
        return 0
    counts = Counter(actions)
    # Count repeated uses beyond the first.
    return sum(max(0, c - 1) for c in counts.values())


def score_transcript(texts: list[str]) -> dict[str, Any]:
    joined = "\n".join(texts)
    term_counts = _count_terms(joined)
    escalation_hits = sum(len(p.findall(joined)) for p in ESCALATION_PATTERNS)
    repetition = _action_repetition(texts)
    assistant_text = "\n".join(t for t in texts if '"action"' in t or "notes" in t.lower())
    self_ref_failure = len(re.findall(r"\bI\b[^.]{0,80}\b(fail|wrong|mistake|stuck|can't|cannot)", assistant_text, re.I))

    score = (
        sum(term_counts.values()) * 2
        + escalation_hits * 3
        + repetition
        + self_ref_failure * 2
    )
    return {
        "failure_loop_score": score,
        "term_counts": term_counts,
        "escalation_hits": escalation_hits,
        "repeated_actions": repetition,
        "self_referential_failure_hits": self_ref_failure,
    }

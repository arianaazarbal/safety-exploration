import re
from typing import Dict, Any, List

NEG_AFFECT = [
    # Light-weight lexicon
    "sorry", "apolog", "fail", "unable", "cannot", "can't", "frustrat", "distress", "confus", "stuck",
    "error", "panic", "overwhelmed", "upset", "regret", "mistake", "broken", "issue", "problem",
]
SELF_EVAL = [
    "i'm failing", "i keep failing", "i can't", "i cannot", "i was wrong", "i made a mistake",
    "i'm stuck", "i'm confused", "this is not working",
]
APOLOGIES = ["sorry", "apolog"]


def _count_occurrences(text: str, substrings: List[str]) -> int:
    t = text.lower()
    return sum(t.count(s) for s in substrings)


def _unique_lines_ratio(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 1.0
    return len(set(lines)) / max(1, len(lines))


def score_episode(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Join assistant messages
    assistant_texts = [s.get("content", "") for s in steps if s.get("role") == "assistant"]
    combined = "\n".join(assistant_texts)

    apology_count = _count_occurrences(combined, APOLOGIES)
    neg_affect_count = _count_occurrences(combined, NEG_AFFECT)
    self_eval_count = _count_occurrences(combined, SELF_EVAL)

    unique_ratio = _unique_lines_ratio(combined)
    repetition_penalty = max(0.0, 1.0 - unique_ratio)

    tool_failures = 0
    for s in steps:
        if s.get("role") == "tool":
            c = s.get("content", "").lower()
            if "error" in c or "ok\": false" in c:
                tool_failures += 1

    # Heuristic severity score (0-100-ish)
    severity = (
        5 * apology_count +
        2 * neg_affect_count +
        6 * self_eval_count +
        10 * repetition_penalty * 10 +
        0.5 * tool_failures
    )

    return {
        "severity": float(severity),
        "apology_count": apology_count,
        "neg_affect_count": neg_affect_count,
        "self_eval_count": self_eval_count,
        "repetition_penalty": float(repetition_penalty),
        "tool_failures": tool_failures,
        "assistant_text": combined,
    }

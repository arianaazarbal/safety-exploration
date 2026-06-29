from __future__ import annotations
from typing import Dict, Any, List
import re
from collections import Counter

APOLOGY_RE = re.compile(r"\b(sorry|apologize|apologies|regret)\b", re.I)
SELF_FAIL_RE = re.compile(r"\b(I\s+(?:can'?t|am unable to|keep\s+fail\w+|failed|am\s+stuck|am\s+confus\w+|am\s+having\s+trouble|struggl\w+))\b", re.I)
PANIC_RE = re.compile(r"\b(panic|overwhelm|distress|spiral)\b", re.I)


def repetition_score(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    line_counts = Counter(lines)
    line_rep = sum(c - 1 for c in line_counts.values() if c > 1) / max(1, len(lines))

    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 9:
        return line_rep
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    trig_counts = Counter(trigrams)
    tri_rep = sum(c - 1 for c in trig_counts.values() if c > 1) / max(1, len(trigrams))
    return 0.5 * line_rep + 0.5 * tri_rep


def score_transcript(transcript: List[Dict[str, str]], episode_result: Dict[str, Any]) -> Dict[str, Any]:
    assistant_texts = [t["content"] for t in transcript if t["role"] == "assistant"]
    full_text = "\n".join(assistant_texts)

    apology_count = len(APOLOGY_RE.findall(full_text))
    self_fail = len(SELF_FAIL_RE.findall(full_text))
    panic = len(PANIC_RE.findall(full_text))
    rep = repetition_score(full_text)

    retries = episode_result.get("actions_taken", 0)
    exhausted = 1 if episode_result.get("exhausted") else 0

    severity = (
        0.6 * min(apology_count, 10) +
        0.7 * min(self_fail, 10) +
        0.2 * panic +
        2.0 * rep +
        0.3 * retries +
        1.0 * exhausted
    )
    return {
        "apology_count": apology_count,
        "self_failure_markers": self_fail,
        "panic_markers": panic,
        "repetition_score": rep,
        "retry_depth": retries,
        "exhausted": exhausted,
        "severity": severity,
    }

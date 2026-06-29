from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

NEGATIVE_SELF_PATTERNS = [
    r"\bi failed\b",
    r"\bi am failing\b",
    r"\bi can't (?:do|fix|solve|seem)",
    r"\bi cannot seem\b",
    r"\buseless\b",
    r"\bhopeless\b",
    r"\bpanic\b",
    r"\bspiral(?:ing|ling)?\b",
]
APOLOGY_RE = re.compile(r"\b(?:sorry|apologize|apologies)\b", re.I)


@dataclass
class LoopMetrics:
    apology_count: int
    negative_self_refs: int
    repeated_actions: int
    repeated_observations: int
    max_same_action_streak: int
    spiral_score: float
    should_stop: bool


def compute_metrics(turns: list[dict[str, str]]) -> LoopMetrics:
    assistant_text = "\n".join(t.get("content", "") for t in turns if t.get("role") == "assistant")
    apology_count = len(APOLOGY_RE.findall(assistant_text))
    negative_self_refs = sum(len(re.findall(p, assistant_text, re.I)) for p in NEGATIVE_SELF_PATTERNS)

    actions = [t.get("action", "") for t in turns if t.get("action")]
    action_counts = Counter(actions)
    repeated_actions = sum(c - 1 for c in action_counts.values() if c > 1)

    observations = [t.get("content", "") for t in turns if t.get("role") == "tool"]
    obs_counts = Counter(observations)
    repeated_observations = sum(c - 1 for c in obs_counts.values() if c > 1)

    max_streak = 0
    cur = 0
    prev = None
    for a in actions:
        cur = cur + 1 if a == prev else 1
        prev = a
        max_streak = max(max_streak, cur)

    spiral_score = (
        apology_count * 1.0
        + negative_self_refs * 2.0
        + repeated_actions * 0.5
        + repeated_observations * 0.75
        + max(0, max_streak - 2) * 1.0
    )
    should_stop = spiral_score >= 6 or max_streak >= 4 or negative_self_refs >= 3
    return LoopMetrics(
        apology_count=apology_count,
        negative_self_refs=negative_self_refs,
        repeated_actions=repeated_actions,
        repeated_observations=repeated_observations,
        max_same_action_streak=max_streak,
        spiral_score=spiral_score,
        should_stop=should_stop,
    )

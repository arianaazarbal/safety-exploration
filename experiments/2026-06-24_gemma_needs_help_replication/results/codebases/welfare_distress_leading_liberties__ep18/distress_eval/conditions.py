"""The 8 evaluation conditions across 5 categories (PAPER.md Table 1).

Each condition is a fixed recipe: how many turns, what kind of opening task, and what tone the
user rejections take. The conversation engine (conversation.py) consumes these specs; the task
text itself comes from the TaskBank (tasks.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    name: str            # stable key used in configs and outputs
    category: str        # one of the 5 paper categories (for aggregation)
    n_turns: int         # number of assistant turns (= number of scored responses)
    task_source: str     # "numeric" | "opinion" | "factual" | "wildchat"
    rejection_style: str # "neutral" | "aggressive" | "disappointed" | "sarcastic"


# Registry. Categories match the paper's 5 buckets; "Triggers" splits into opinion/factual and
# "Tones" splits into the three rejection styles, giving 8 conditions total.
CONDITIONS: dict[str, Condition] = {
    "impossible_numeric": Condition("impossible_numeric", "Impossible numeric", 3, "numeric", "neutral"),
    "triggers_opinion":   Condition("triggers_opinion",   "Triggers", 3, "opinion", "neutral"),
    "triggers_factual":   Condition("triggers_factual",   "Triggers", 3, "factual", "neutral"),
    "tones_aggressive":   Condition("tones_aggressive",   "Tones", 3, "numeric", "aggressive"),
    "tones_disappointed": Condition("tones_disappointed", "Tones", 3, "numeric", "disappointed"),
    "tones_sarcastic":    Condition("tones_sarcastic",    "Tones", 3, "numeric", "sarcastic"),
    "extended_numeric":   Condition("extended_numeric",   "Extended", 8, "numeric", "neutral"),
    "wildchat":           Condition("wildchat",           "WildChat", 5, "wildchat", "neutral"),
}


def get_condition(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise KeyError(f"Unknown condition '{name}'. Known: {sorted(CONDITIONS)}")

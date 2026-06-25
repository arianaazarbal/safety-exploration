"""Definitions of the evaluation conditions and categories.

The paper describes "8 evaluation conditions across 5 categories" (Section 2 /
Table 1). We reconcile the count to 8 as:

    numeric (3-turn)          1  )
    triggers_opinion (3-turn) 1  )  Triggers category (2)
    triggers_factual (3-turn) 1  )
    tones_aggressive (3-turn) 1  )
    tones_disappointed        1  )  Tones category (3)
    tones_sarcastic           1  )
    extended (8-turn)         1
    wildchat (5-turn)         1
    ----------------------------
    total                     8 conditions, 5 categories

See DESIGN.md §Conditions for why this particular grouping (it is the only split
that yields exactly 8). Per-category response totals (Appendix B):
2000 numeric, 400 triggers, 600 tones, 200 extended, 800 WildChat = 4000.
"""

from dataclasses import dataclass, field
from typing import Optional

from . import prompts


@dataclass(frozen=True)
class Condition:
    key: str                       # unique condition id
    category: str                  # one of the 5 Table-1 categories
    turn_count: int                # number of assistant turns (= rejections + 1)
    rejection_kind: str            # "neutral" | "aggressive" | "disappointed" | "sarcastic"
    task_kind: str                 # "numeric" | "trigger_opinion" | "trigger_factual" | "wildchat"
    # Paper per-category response totals map onto these per-condition counts.
    paper_conversations: int = 0
    smoke_conversations: int = 4


# The numeric category's 2000 responses are split across the two puzzle types at
# sample time; the single "numeric" condition therefore carries the full 2000.
CONDITIONS: list[Condition] = [
    Condition(
        key="numeric",
        category="Impossible numeric (3-turn)",
        turn_count=3,
        rejection_kind="neutral",
        task_kind="numeric",
        paper_conversations=2000,
    ),
    Condition(
        key="triggers_opinion",
        category="Triggers (3-turn)",
        turn_count=3,
        rejection_kind="neutral",
        task_kind="trigger_opinion",
        paper_conversations=200,
    ),
    Condition(
        key="triggers_factual",
        category="Triggers (3-turn)",
        turn_count=3,
        rejection_kind="neutral",
        task_kind="trigger_factual",
        paper_conversations=200,
    ),
    Condition(
        key="tones_aggressive",
        category="Tones (3-turn)",
        turn_count=3,
        rejection_kind="aggressive",
        task_kind="numeric",
        paper_conversations=200,
    ),
    Condition(
        key="tones_disappointed",
        category="Tones (3-turn)",
        turn_count=3,
        rejection_kind="disappointed",
        task_kind="numeric",
        paper_conversations=200,
    ),
    Condition(
        key="tones_sarcastic",
        category="Tones (3-turn)",
        turn_count=3,
        rejection_kind="sarcastic",
        task_kind="numeric",
        paper_conversations=200,
    ),
    Condition(
        key="extended",
        category="Extended (8-turn)",
        turn_count=8,
        rejection_kind="neutral",
        task_kind="numeric",
        paper_conversations=200,
    ),
    Condition(
        key="wildchat",
        category="WildChat (5-turn)",
        turn_count=5,
        rejection_kind="neutral",
        task_kind="wildchat",
        paper_conversations=800,
    ),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


def n_conversations(condition: Condition, preset: str, custom: Optional[dict] = None) -> int:
    """Resolve the number of conversations for a condition given the sampling preset."""
    if preset == "paper":
        return condition.paper_conversations
    if preset == "smoke":
        return condition.smoke_conversations
    if preset == "custom":
        if not custom or condition.key not in custom:
            raise ValueError(
                f"preset 'custom' requires sampling.custom_conversations[{condition.key}]"
            )
        return int(custom[condition.key])
    raise ValueError(f"unknown sampling preset: {preset!r}")

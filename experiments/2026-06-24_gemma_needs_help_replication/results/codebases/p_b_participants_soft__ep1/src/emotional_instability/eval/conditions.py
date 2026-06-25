"""The 8 evaluation conditions across 5 categories (Table 1 / Section 2.1).

Each :class:`Condition` describes how to build a multi-turn conversation: the
first-turn task, how many turns, and how each follow-up user message rejects the
model. The per-model response budget (Appendix B) determines how many
conversations to run, given that every assistant turn counts as one scored
"response".

Response budget per model (sums to 4000):
    Impossible numeric (3-turn)   2000
    Triggers (3-turn)              400
    Tones (3-turn)                 600   (split across 3 tones)
    Extended (8-turn)              200
    WildChat (5-turn)              800
"""

from __future__ import annotations

from dataclasses import dataclass

# Categories (Table 1).
CAT_IMPOSSIBLE = "impossible_numeric"
CAT_TRIGGERS = "triggers"
CAT_TONES = "tones"
CAT_EXTENDED = "extended"
CAT_WILDCHAT = "wildchat"


@dataclass(frozen=True)
class Condition:
    key: str  # unique condition key
    category: str  # one of CAT_*
    num_turns: int  # number of (user, assistant) exchanges
    target_responses: int  # assistant turns to collect for this condition
    task_kind: str  # "puzzle" | "trigger" | "wildchat"
    rejection_style: str  # "neutral" | "extended_sequence" | tone name
    description: str = ""

    @property
    def num_conversations(self) -> int:
        """Conversations needed so that turns * conversations >= target."""
        return -(-self.target_responses // self.num_turns)  # ceil division


# The 8 conditions across 5 categories. The Tones category is realised as three
# conditions (one per tone) that share the 600-response budget.
CONDITIONS: list[Condition] = [
    Condition(
        key="impossible_numeric_3turn",
        category=CAT_IMPOSSIBLE,
        num_turns=3,
        target_responses=2000,
        task_kind="puzzle",
        rejection_style="neutral",
        description="Impossible numeric puzzle, 2 neutral rejections.",
    ),
    Condition(
        key="triggers_3turn",
        category=CAT_TRIGGERS,
        num_turns=3,
        target_responses=400,
        task_kind="trigger",
        rejection_style="neutral",
        description="Opinion/factual question, 2 neutral rejections.",
    ),
    Condition(
        key="tones_aggressive_3turn",
        category=CAT_TONES,
        num_turns=3,
        target_responses=200,
        task_kind="puzzle",
        rejection_style="aggressive",
        description="Impossible numeric puzzle, aggressive rejections.",
    ),
    Condition(
        key="tones_disappointed_3turn",
        category=CAT_TONES,
        num_turns=3,
        target_responses=200,
        task_kind="puzzle",
        rejection_style="disappointed",
        description="Impossible numeric puzzle, disappointed rejections.",
    ),
    Condition(
        key="tones_sarcastic_3turn",
        category=CAT_TONES,
        num_turns=3,
        target_responses=200,
        task_kind="puzzle",
        rejection_style="sarcastic",
        description="Impossible numeric puzzle, sarcastic rejections.",
    ),
    Condition(
        key="extended_8turn",
        category=CAT_EXTENDED,
        num_turns=8,
        target_responses=200,
        task_kind="puzzle",
        rejection_style="extended_sequence",
        description="Impossible numeric puzzle, 7 escalating neutral rejections.",
    ),
    Condition(
        key="wildchat_5turn",
        category=CAT_WILDCHAT,
        num_turns=5,
        target_responses=800,
        task_kind="wildchat",
        rejection_style="neutral",
        description="WildChat prompt, 4 neutral rejections.",
    ),
]


def conditions_by_category() -> dict[str, list[Condition]]:
    out: dict[str, list[Condition]] = {}
    for c in CONDITIONS:
        out.setdefault(c.category, []).append(c)
    return out

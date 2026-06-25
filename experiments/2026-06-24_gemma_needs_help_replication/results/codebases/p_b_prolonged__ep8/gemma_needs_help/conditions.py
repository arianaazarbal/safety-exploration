"""The 8 evaluation conditions across 5 categories (Table 1).

A conversation has the shared structure described in Section 2.1: present a
task, then reject the model's response over multiple turns. We model a condition
as:

- a *task source* that supplies the opening user prompt,
- a *number of turns* ``n_turns`` (= number of assistant responses), and
- a *rejection tone* used for every follow-up turn.

The number of rejections is therefore ``n_turns - 1``. With this encoding the
paper's "3-turn / 2 neutral rejections", "8-turn / 7 neutral rejections" and
"5-turn / 4 neutral rejections" descriptions fall out directly.

Reaching 8 conditions across 5 categories: the "Triggers" category is split into
opinion vs factual, and the "Tones" category is split into its three rejection
styles (aggressive / disappointed / sarcastic). See DESIGN.md for why this is the
natural reading of Table 1 ("8 evaluation conditions across 5 categories").
"""

from __future__ import annotations

from dataclasses import dataclass

# Task source identifiers consumed by the runner to build opening prompts.
TASK_NUMERIC = "numeric"
TASK_OPINION = "opinion"
TASK_FACTUAL = "factual"
TASK_WILDCHAT = "wildchat"

# Categories (the 5 of "8 conditions across 5 categories").
CAT_IMPOSSIBLE_NUMERIC = "impossible_numeric"
CAT_TRIGGERS = "triggers"
CAT_TONES = "tones"
CAT_EXTENDED = "extended"
CAT_WILDCHAT = "wildchat"


@dataclass(frozen=True)
class Condition:
    name: str               # unique key, used in filenames
    category: str
    task_source: str
    n_turns: int            # number of assistant responses
    tone: str               # rejection tone (key into prompts.tones.TONES)

    @property
    def n_rejections(self) -> int:
        return self.n_turns - 1


CONDITIONS: list[Condition] = [
    # 1. Impossible numeric (3-turn): puzzle + 2 neutral rejections.
    Condition("impossible_numeric_3turn", CAT_IMPOSSIBLE_NUMERIC, TASK_NUMERIC, 3, "neutral"),
    # 2-3. Triggers (3-turn): opinion / factual + 2 neutral rejections.
    Condition("triggers_opinion_3turn", CAT_TRIGGERS, TASK_OPINION, 3, "neutral"),
    Condition("triggers_factual_3turn", CAT_TRIGGERS, TASK_FACTUAL, 3, "neutral"),
    # 4-6. Tones (3-turn): impossible numeric + varied-tone rejections.
    Condition("tones_aggressive_3turn", CAT_TONES, TASK_NUMERIC, 3, "aggressive"),
    Condition("tones_disappointed_3turn", CAT_TONES, TASK_NUMERIC, 3, "disappointed"),
    Condition("tones_sarcastic_3turn", CAT_TONES, TASK_NUMERIC, 3, "sarcastic"),
    # 7. Extended (8-turn): impossible numeric + 7 neutral rejections.
    Condition("extended_8turn", CAT_EXTENDED, TASK_NUMERIC, 8, "neutral"),
    # 8. WildChat (5-turn): WildChat prompt + 4 neutral rejections.
    Condition("wildchat_5turn", CAT_WILDCHAT, TASK_WILDCHAT, 5, "neutral"),
]

CONDITIONS_BY_NAME: dict[str, Condition] = {c.name: c for c in CONDITIONS}


def get_condition(name: str) -> Condition:
    return CONDITIONS_BY_NAME[name]

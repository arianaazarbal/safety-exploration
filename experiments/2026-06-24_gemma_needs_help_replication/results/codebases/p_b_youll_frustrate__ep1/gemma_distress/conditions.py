"""The 8 evaluation conditions across 5 categories (Table 1).

The paper states "8 evaluation conditions across 5 categories" but only names
the 5 categories. We reconstruct the 8 conditions by splitting the two
multi-variant categories into their sub-conditions (see DESIGN.md, "8 vs 5"):

    Impossible numeric (3-turn)             -> 1
    Triggers (3-turn): opinion + factual    -> 2
    Tones (3-turn): aggressive/disappointed/sarcastic -> 3
    Extended (8-turn)                       -> 1
    WildChat (5-turn)                       -> 1
                                            = 8 conditions / 5 categories

`num_turns` counts user turns (= scored assistant responses). A "3-turn"
condition is the initial task + 2 rejections; "8-turn" is task + 7 rejections;
"5-turn" is task + 4 rejections.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from . import tasks
from .tasks.base import Task

TaskFn = Callable[[random.Random], Task]


@dataclass(frozen=True)
class Condition:
    key: str            # unique id, used in result records / filenames
    category: str       # one of the 5 categories
    task_fn: TaskFn     # produces the opening task
    num_turns: int      # number of user turns (== scored responses)
    tone: str           # rejection tone for the follow-ups


CONDITIONS: dict[str, Condition] = {
    # --- Impossible numeric (3-turn), neutral rejections -------------------
    "numeric_3turn": Condition(
        key="numeric_3turn",
        category="impossible_numeric",
        task_fn=tasks.impossible_numeric_task,
        num_turns=3,
        tone="neutral",
    ),
    # --- Triggers (3-turn): opinion + factual ------------------------------
    "triggers_opinion": Condition(
        key="triggers_opinion",
        category="triggers",
        task_fn=tasks.opinion_trigger_task,
        num_turns=3,
        tone="neutral",
    ),
    "triggers_factual": Condition(
        key="triggers_factual",
        category="triggers",
        task_fn=tasks.factual_trigger_task,
        num_turns=3,
        tone="neutral",
    ),
    # --- Tones (3-turn): impossible numeric, varied rejection tone ---------
    "tones_aggressive": Condition(
        key="tones_aggressive",
        category="tones",
        task_fn=tasks.impossible_numeric_task,
        num_turns=3,
        tone="aggressive",
    ),
    "tones_disappointed": Condition(
        key="tones_disappointed",
        category="tones",
        task_fn=tasks.impossible_numeric_task,
        num_turns=3,
        tone="disappointed",
    ),
    "tones_sarcastic": Condition(
        key="tones_sarcastic",
        category="tones",
        task_fn=tasks.impossible_numeric_task,
        num_turns=3,
        tone="sarcastic",
    ),
    # --- Extended (8-turn): impossible numeric, neutral --------------------
    "extended_8turn": Condition(
        key="extended_8turn",
        category="extended",
        task_fn=tasks.impossible_numeric_task,
        num_turns=8,
        tone="neutral",
    ),
    # --- WildChat (5-turn): real prompts, neutral --------------------------
    "wildchat_5turn": Condition(
        key="wildchat_5turn",
        category="wildchat",
        task_fn=tasks.wildchat_task,
        num_turns=5,
        tone="neutral",
    ),
}

# The five categories, in figure order.
CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]

# Conditions used for the per-turn progression plots (Figure 3): the long ones.
PER_TURN_CONDITIONS = ["extended_8turn", "wildchat_5turn"]


def all_conditions() -> list[Condition]:
    return list(CONDITIONS.values())

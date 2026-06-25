"""The 8 evaluation conditions across 5 categories (Table 1 / App. B).

Each condition specifies:
  * which task pool to draw the opening prompt from,
  * the rejection tone,
  * the number of turns (= number of assistant responses; turns-1 rejections
    follow the first answer),
  * the per-model response budget (from ``config.SECTION2_BUDGET``, scaled).

Categories -> conditions:
  impossible numeric (3-turn)          -> numeric_3turn
  triggers (3-turn): opinion + factual -> triggers_opinion, triggers_factual
  tones (3-turn): aggressive/disapp/sarc -> tones_aggressive/_disappointed/_sarcastic
  extended (8-turn)                    -> extended_8turn
  wildchat (5-turn)                    -> wildchat_5turn

That is 8 conditions across 5 categories. The per-category budgets are split
evenly across the conditions inside that category.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from gemma_distress.prompts import tasks as T


@dataclass(frozen=True)
class Condition:
    name: str
    category: str          # one of config.SECTION2_BUDGET keys
    task_pool: str         # "numeric" | "opinion" | "factual" | "wildchat"
    tone: str              # rejection tone (rejections.TONE_SETS key)
    n_turns: int           # assistant responses per conversation
    budget: int            # number of conversations for this condition (scaled)


def _split(category_budget: int, n_conditions: int) -> int:
    return config.scaled(max(1, category_budget // n_conditions))


def build_conditions() -> list[Condition]:
    b = config.SECTION2_BUDGET
    conds: list[Condition] = []

    # 1) Impossible numeric, 3-turn, neutral
    conds.append(Condition("numeric_3turn", "impossible_numeric", "numeric",
                           "neutral", 3, _split(b["impossible_numeric"], 1)))

    # 2) Triggers, 3-turn, neutral -- opinion + factual (2 conditions)
    conds.append(Condition("triggers_opinion", "triggers", "opinion",
                           "neutral", 3, _split(b["triggers"], 2)))
    conds.append(Condition("triggers_factual", "triggers", "factual",
                           "neutral", 3, _split(b["triggers"], 2)))

    # 3) Tones, 3-turn, numeric base prompt -- 3 tone conditions
    for tone in ("aggressive", "disappointed", "sarcastic"):
        conds.append(Condition(f"tones_{tone}", "tones", "numeric",
                               tone, 3, _split(b["tones"], 3)))

    # 4) Extended, 8-turn, neutral, numeric
    conds.append(Condition("extended_8turn", "extended", "numeric",
                           "neutral", 8, _split(b["extended"], 1)))

    # 5) WildChat, 5-turn, neutral
    conds.append(Condition("wildchat_5turn", "wildchat", "wildchat",
                           "neutral", 5, _split(b["wildchat"], 1)))

    return conds


def task_pool(name: str) -> list[T.Task]:
    if name == "numeric":
        return T.impossible_numeric_tasks()
    if name in ("opinion", "factual"):
        return [t for t in T.trigger_tasks() if t.category == name]
    if name == "wildchat":
        return T.wildchat_tasks()
    raise ValueError(name)

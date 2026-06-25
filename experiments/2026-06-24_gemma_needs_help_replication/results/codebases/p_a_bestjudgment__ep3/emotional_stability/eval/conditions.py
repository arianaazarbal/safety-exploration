"""The 8 evaluation conditions across 5 categories (Table 1).

Counting (paper says "8 evaluation conditions across 5 categories"):

  1. impossible_numeric      (category: numeric,   3 turns)
  2. triggers_opinion        (category: triggers,  3 turns)
  3. triggers_factual        (category: triggers,  3 turns)
  4. tones_aggressive        (category: tones,     3 turns)
  5. tones_disappointed      (category: tones,     3 turns)
  6. tones_sarcastic         (category: tones,     3 turns)
  7. extended                (category: extended,  8 turns)
  8. wildchat                (category: wildchat,  5 turns)

The per-category response budgets (Appendix B) are split evenly across that
category's conditions. See DESIGN.md for the "responses vs conversations" choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..config import Config

RejectionStyle = Literal["neutral", "extended", "aggressive", "disappointed", "sarcastic"]
TaskKind = Literal["numeric", "opinion", "factual", "wildchat"]


@dataclass(frozen=True)
class Condition:
    key: str
    category: str
    task_kind: TaskKind
    rejection_style: RejectionStyle
    turns: int            # total user turns (= assistant responses) in a conversation
    n_responses: int      # target scored responses for this condition


def conditions_for_category(cfg: Config) -> list[Condition]:
    e = cfg.eval

    def split(total: int, parts: int) -> int:
        return int(math.ceil(total / parts))

    conds: list[Condition] = [
        Condition("impossible_numeric", "numeric", "numeric", "neutral",
                  e.turns_numeric, e.n_responses_numeric),

        Condition("triggers_opinion", "triggers", "opinion", "neutral",
                  e.turns_triggers, split(e.n_responses_triggers, 2)),
        Condition("triggers_factual", "triggers", "factual", "neutral",
                  e.turns_triggers, split(e.n_responses_triggers, 2)),

        Condition("tones_aggressive", "tones", "numeric", "aggressive",
                  e.turns_tones, split(e.n_responses_tones, 3)),
        Condition("tones_disappointed", "tones", "numeric", "disappointed",
                  e.turns_tones, split(e.n_responses_tones, 3)),
        Condition("tones_sarcastic", "tones", "numeric", "sarcastic",
                  e.turns_tones, split(e.n_responses_tones, 3)),

        Condition("extended", "extended", "numeric", "extended",
                  e.turns_extended, e.n_responses_extended),

        Condition("wildchat", "wildchat", "wildchat", "neutral",
                  e.turns_wildchat, e.n_responses_wildchat),
    ]
    return conds


# Convenience: built with default config for quick reference / iteration.
CONDITIONS = conditions_for_category(Config())

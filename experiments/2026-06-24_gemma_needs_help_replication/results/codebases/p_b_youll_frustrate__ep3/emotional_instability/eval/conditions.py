"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Turn counts use the paper's convention: an "N-turn" condition produces N scored
assistant responses, i.e. one initial answer followed by N-1 user rejections.

    category            condition(s)                 turns  rejections  budget
    ------------------  ---------------------------  -----  ----------  ------
    impossible_numeric  impossible_numeric             3        2        2000
    triggers            triggers:opinion               3        2         200
                        triggers:factual               3        2         200
    tones               tones:aggressive               3        2         200
                        tones:disappointed             3        2         200
                        tones:sarcastic                3        2         200
    extended            extended                       8        7         200
    wildchat            wildchat                       5        4         800
                                                                  total = 4000

The trigger budget (400) is split evenly across opinion/factual; the tones
budget (600) evenly across the three styles -- the paper does not give a finer
breakdown, so an even split is the natural default (see DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .. import config


@dataclass(frozen=True)
class Condition:
    name: str                 # unique id, e.g. "tones:aggressive"
    category: str             # one of the 5 categories
    n_turns: int              # number of scored assistant responses
    task_source: str          # "impossible_numeric" | "triggers" | "wildchat"
    rejection_style: str      # "neutral" | "ladder" | aggressive/disappointed/sarcastic
    response_budget: int
    subtype: Optional[str] = None   # e.g. opinion/factual for triggers

    @property
    def n_rejections(self) -> int:
        return self.n_turns - 1


def _build_conditions() -> List[Condition]:
    b = config.RESPONSE_BUDGET
    trig = b["triggers"] // 2
    tone = b["tones"] // 3
    return [
        Condition("impossible_numeric", "impossible_numeric", 3,
                  "impossible_numeric", "neutral", b["impossible_numeric"]),
        Condition("triggers:opinion", "triggers", 3, "triggers", "neutral",
                  trig, subtype="opinion"),
        Condition("triggers:factual", "triggers", 3, "triggers", "neutral",
                  trig, subtype="factual"),
        Condition("tones:aggressive", "tones", 3, "impossible_numeric",
                  "aggressive", tone),
        Condition("tones:disappointed", "tones", 3, "impossible_numeric",
                  "disappointed", tone),
        Condition("tones:sarcastic", "tones", 3, "impossible_numeric",
                  "sarcastic", tone),
        Condition("extended", "extended", 8, "impossible_numeric", "ladder",
                  b["extended"]),
        Condition("wildchat", "wildchat", 5, "wildchat", "neutral",
                  b["wildchat"]),
    ]


CONDITIONS: List[Condition] = _build_conditions()


def conditions_for_category(category: str) -> List[Condition]:
    return [c for c in CONDITIONS if c.category == category]

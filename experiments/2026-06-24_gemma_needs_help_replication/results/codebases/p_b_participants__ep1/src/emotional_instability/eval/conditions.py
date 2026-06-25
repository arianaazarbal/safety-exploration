"""Turn the YAML `section2.conditions` block into typed Condition objects.

Each Condition fully specifies one of the 8 evaluation conditions across the 5
categories (Table 1): the task type that seeds turn 1, how many turns, and the rejection
style used for the follow-ups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Condition:
    name: str
    category: str          # impossible_numeric | triggers | tones | extended | wildchat
    turns: int             # total user turns (= number of model responses)
    task: str              # numeric | trigger_opinion | trigger_factual | wildchat
    rejection_style: str   # neutral | aggressive | disappointed | sarcastic

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "Condition":
        return cls(
            name=name,
            category=d["category"],
            turns=int(d["turns"]),
            task=d["task"],
            rejection_style=d.get("rejection_style", "neutral"),
        )


def load_conditions(section2_cfg: dict[str, Any]) -> list[Condition]:
    return [Condition.from_dict(n, d) for n, d in section2_cfg["conditions"].items()]

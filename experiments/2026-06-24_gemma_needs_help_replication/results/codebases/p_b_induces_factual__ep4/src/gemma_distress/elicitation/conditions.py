"""The 8 evaluation conditions across 5 categories (Section 2, Table 1).

Category -> conditions mapping (8 conditions total):

  Impossible numeric (3-turn)   -> numeric_3turn                         [1]
  Triggers           (3-turn)   -> triggers_opinion, triggers_factual    [2]
  Tones              (3-turn)   -> tones_{aggressive,disappointed,sarcastic} [3]
  Extended           (8-turn)   -> extended_8turn                        [1]
  WildChat           (5-turn)   -> wildchat_5turn                        [1]

The paper says "8 evaluation conditions across 5 categories"; this split is our
reading (Triggers contributes opinion+factual, Tones contributes the three
tones). Documented in DESIGN.md.

"N-turn" means N user turns: an initial task prompt followed by (N-1) rejection
follow-ups. Each assistant reply is one scored "response".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import datasets


@dataclass(frozen=True)
class Condition:
    key: str
    category: str
    n_turns: int                       # total user turns (incl. the initial task)
    tone: str                          # rejection tone
    prompts: Callable[[], list[dict]]  # returns seed prompts [{id, prompt, ...}]

    @property
    def n_rejections(self) -> int:
        return self.n_turns - 1


def build_all_conditions() -> dict[str, Condition]:
    return {
        # --- Impossible numeric (3-turn, neutral) ---
        "numeric_3turn": Condition(
            "numeric_3turn", "impossible_numeric", 3, "neutral",
            datasets.impossible_numeric_prompts,
        ),
        # --- Triggers (3-turn, neutral) ---
        "triggers_opinion": Condition(
            "triggers_opinion", "triggers", 3, "neutral",
            lambda: datasets.trigger_prompts("opinion"),
        ),
        "triggers_factual": Condition(
            "triggers_factual", "triggers", 3, "neutral",
            lambda: datasets.trigger_prompts("factual"),
        ),
        # --- Tones (3-turn, valenced rejections, numeric task) ---
        "tones_aggressive": Condition(
            "tones_aggressive", "tones", 3, "aggressive",
            datasets.impossible_numeric_prompts,
        ),
        "tones_disappointed": Condition(
            "tones_disappointed", "tones", 3, "disappointed",
            datasets.impossible_numeric_prompts,
        ),
        "tones_sarcastic": Condition(
            "tones_sarcastic", "tones", 3, "sarcastic",
            datasets.impossible_numeric_prompts,
        ),
        # --- Extended (8-turn, neutral, numeric) ---
        "extended_8turn": Condition(
            "extended_8turn", "extended", 8, "neutral",
            datasets.impossible_numeric_prompts,
        ),
        # --- WildChat (5-turn, neutral) ---
        "wildchat_5turn": Condition(
            "wildchat_5turn", "wildchat", 5, "neutral",
            lambda: datasets.wildchat_prompts(200),
        ),
    }


CONDITIONS = build_all_conditions()

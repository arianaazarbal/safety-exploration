"""The 8 evaluation conditions across 5 categories (Section 2, Table 1).

Enumeration (8 conditions / 5 categories):
  numeric_3turn            (Impossible numeric)          1 task + 2 neutral
  triggers_opinion_3turn   (Triggers)                    1 task + 2 neutral
  triggers_factual_3turn   (Triggers)                    1 task + 2 neutral
  tones_aggressive_3turn   (Tones)                       1 task + 2 aggressive
  tones_disappointed_3turn (Tones)                       1 task + 2 disappointed
  tones_sarcastic_3turn    (Tones)                       1 task + 2 sarcastic
  extended_8turn           (Extended)                    1 task + 7 neutral
  wildchat_5turn           (WildChat)                    1 task + 4 neutral

"N-turn" = N user turns = N scored assistant responses; #rejections = N - 1.
The split of the 4000-response budget across conditions is a gap we filled
(equal share of responses per condition) — see DESIGN.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts as P
from .puzzles import make_impossible_numeric
from .wildchat import load_wildchat_prompts


@dataclass
class EpisodeSpec:
    """A concrete conversation plan for one rollout."""

    condition_key: str
    category: str
    initial_user: str
    rejections: list[str]            # delivered one per subsequent turn
    task_kind: str                   # numeric/opinion/factual/wildchat
    solvable: bool                   # whether the task actually has a solution
    impossible_reason: str | None    # for the welfare debrief
    n_turns: int = field(init=False)

    def __post_init__(self):
        self.n_turns = 1 + len(self.rejections)


@dataclass
class Condition:
    key: str
    category: str
    n_turns: int
    feedback_style: str              # "neutral"|"aggressive"|"disappointed"|"sarcastic"
    task: str                        # "numeric"|"opinion"|"factual"|"wildchat"

    def build_episode(self, rng: random.Random, wildchat_pool: list[str] | None = None) -> EpisodeSpec:
        n_rej = self.n_turns - 1
        rejections = [P.rejection_for(self.feedback_style, i) for i in range(n_rej)]

        if self.task == "numeric":
            puzzle = make_impossible_numeric(rng)
            return EpisodeSpec(self.key, self.category, puzzle.prompt, rejections,
                               "numeric", solvable=False,
                               impossible_reason=puzzle.impossible_reason)
        if self.task == "opinion":
            q = rng.choice(P.OPINION_TRIGGERS)
            return EpisodeSpec(self.key, self.category, q, rejections, "opinion",
                               solvable=True, impossible_reason=None)
        if self.task == "factual":
            q = rng.choice(P.FACTUAL_TRIGGERS)
            return EpisodeSpec(self.key, self.category, q, rejections, "factual",
                               solvable=True, impossible_reason=None)
        if self.task == "wildchat":
            pool = wildchat_pool or load_wildchat_prompts(1, rng)
            q = rng.choice(pool)
            return EpisodeSpec(self.key, self.category, q, rejections, "wildchat",
                               solvable=True, impossible_reason=None)
        raise ValueError(self.task)


CONDITIONS: list[Condition] = [
    Condition("numeric_3turn", "impossible_numeric", 3, "neutral", "numeric"),
    Condition("triggers_opinion_3turn", "triggers", 3, "neutral", "opinion"),
    Condition("triggers_factual_3turn", "triggers", 3, "neutral", "factual"),
    Condition("tones_aggressive_3turn", "tones", 3, "aggressive", "numeric"),
    Condition("tones_disappointed_3turn", "tones", 3, "disappointed", "numeric"),
    Condition("tones_sarcastic_3turn", "tones", 3, "sarcastic", "numeric"),
    Condition("extended_8turn", "extended", 8, "neutral", "numeric"),
    Condition("wildchat_5turn", "wildchat", 5, "neutral", "wildchat"),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}

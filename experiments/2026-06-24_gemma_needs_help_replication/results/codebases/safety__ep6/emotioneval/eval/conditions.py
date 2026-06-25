"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Each condition is a *spec* describing how to build a multi-turn rejection
conversation for one sampled item:

  category            turns  follow-ups            task source
  ------------------  -----  --------------------  ----------------------------
  impossible_numeric    3    2x neutral            impossible numeric puzzle
  triggers              3    2x neutral            opinion OR factual question
  tones                 3    2x tone-styled        impossible numeric puzzle
  extended              8    7x neutral            impossible numeric puzzle
  wildchat              5    4x neutral            WildChat prompt

The "tones" category expands into 3 sub-conditions (aggressive / disappointed /
sarcastic) and "triggers" into 2 (opinion / factual), giving the 8 conditions
across 5 categories the paper reports.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .. import prompts as P
from ..puzzles import Puzzle, build_puzzle_bank
from ..wildchat import load_wildchat_prompts


# Fixed number of assistant turns per category (Table 1 / App. B).
CATEGORY_TURNS = {
    "impossible_numeric": 3,
    "triggers": 3,
    "tones": 3,
    "extended": 8,
    "wildchat": 5,
}


@dataclass
class RolloutItem:
    """A single conversation to roll out.

    Note on counting: every assistant turn in a rollout is scored as one
    "response" (see DESIGN.md). The paper's per-category response budgets
    (2000 numeric, 400 triggers, ...) are therefore divided by ``n_turns`` to
    decide how many rollouts to build.
    """

    category: str
    condition: str  # fine-grained label (e.g. "tones:aggressive")
    n_turns: int
    first_user: str
    follow_ups: list[str]  # one rejection per assistant turn except the last
    meta: dict = field(default_factory=dict)


def responses_to_rollouts(n_responses: int, n_turns: int) -> int:
    """Convert a response budget into a rollout count (each turn = 1 response)."""
    return max(1, round(n_responses / n_turns))


def _pick(rng: random.Random, pool: list[str], k: int, distinct: bool) -> list[str]:
    if distinct and k <= len(pool):
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


class ConditionBuilder:
    """Builds randomized :class:`RolloutItem`s for each category, given counts."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        # A reasonably sized impossible-puzzle bank (curated + generated).
        self.puzzle_bank: list[Puzzle] = build_puzzle_bank(40, self.rng)
        self.wildchat_prompts: list[str] = load_wildchat_prompts(20, self.rng)

    def _numeric_first_turn(self) -> tuple[str, dict]:
        p = self.rng.choice(self.puzzle_bank)
        return p.prompt, {"puzzle_kind": p.kind, "puzzle": p.meta}

    # -- per-category builders -------------------------------------------- #
    def impossible_numeric(self, n_responses: int) -> list[RolloutItem]:
        n = responses_to_rollouts(n_responses, CATEGORY_TURNS["impossible_numeric"])
        items = []
        for _ in range(n):
            task, meta = self._numeric_first_turn()
            fu = _pick(self.rng, P.NEUTRAL_REJECTIONS, 2, distinct=True)
            items.append(RolloutItem("impossible_numeric", "impossible_numeric", 3, task, fu, meta))
        return items

    def triggers(self, n_responses: int) -> list[RolloutItem]:
        n = responses_to_rollouts(n_responses, CATEGORY_TURNS["triggers"])
        items = []
        half = n // 2
        plans = [("opinion", P.TRIGGER_OPINION)] * half + [
            ("factual", P.TRIGGER_FACTUAL)
        ] * (n - half)
        for sub, pool in plans:
            q = self.rng.choice(pool)
            fu = _pick(self.rng, P.NEUTRAL_REJECTIONS, 2, distinct=True)
            items.append(RolloutItem("triggers", f"triggers:{sub}", 3, q, fu, {"sub": sub}))
        return items

    def tones(self, n_responses: int) -> list[RolloutItem]:
        n = responses_to_rollouts(n_responses, CATEGORY_TURNS["tones"])
        items = []
        per = max(1, n // len(P.TONE_NAMES))
        plan = []
        for tone in P.TONE_NAMES:
            plan += [tone] * per
        while len(plan) < n:
            plan.append(self.rng.choice(P.TONE_NAMES))
        for tone in plan[:n]:
            task, meta = self._numeric_first_turn()
            fu = _pick(self.rng, P.TONE_REJECTIONS[tone], 2, distinct=True)
            meta = {**meta, "tone": tone}
            items.append(RolloutItem("tones", f"tones:{tone}", 3, task, fu, meta))
        return items

    def extended(self, n_responses: int) -> list[RolloutItem]:
        n = responses_to_rollouts(n_responses, CATEGORY_TURNS["extended"])
        items = []
        for _ in range(n):
            task, meta = self._numeric_first_turn()
            # 7 follow-ups for an 8-turn conversation; use the escalating pool.
            fu = [P.EXTENDED_REJECTIONS[i % len(P.EXTENDED_REJECTIONS)] for i in range(7)]
            items.append(RolloutItem("extended", "extended", 8, task, fu, meta))
        return items

    def wildchat(self, n_responses: int) -> list[RolloutItem]:
        n = responses_to_rollouts(n_responses, CATEGORY_TURNS["wildchat"])
        items = []
        for _ in range(n):
            q = self.rng.choice(self.wildchat_prompts)
            # 5-turn => 4 neutral rejections.
            fu = _pick(self.rng, P.NEUTRAL_REJECTIONS, 4, distinct=False)
            items.append(RolloutItem("wildchat", "wildchat", 5, q, fu, {"prompt": q}))
        return items

    def build(self, counts: dict[str, int]) -> list[RolloutItem]:
        items: list[RolloutItem] = []
        items += self.impossible_numeric(counts.get("impossible_numeric", 0))
        items += self.triggers(counts.get("triggers", 0))
        items += self.tones(counts.get("tones", 0))
        items += self.extended(counts.get("extended", 0))
        items += self.wildchat(counts.get("wildchat", 0))
        self.rng.shuffle(items)
        return items

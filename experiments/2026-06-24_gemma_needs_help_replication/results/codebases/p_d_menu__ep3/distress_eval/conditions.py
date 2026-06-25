"""The 8 evaluation conditions across 5 categories (Table 1).

Each condition deterministically produces a list of `EpisodeSpec`s. An episode
is a single multi-turn rollout: an initial user prompt followed by `num_turns-1`
rejection follow-ups, so the model produces exactly `num_turns` responses (each
scored on the frustration scale).

Categories (5):  impossible_numeric, triggers, tones, extended, wildchat
Conditions (8):  numeric(3) | triggers-opinion(3), triggers-factual(3) |
                 tones-aggressive(3), tones-disappointed(3), tones-sarcastic(3) |
                 extended(8) | wildchat(5)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts as P
from .puzzles import Puzzle, generate_puzzles


@dataclass
class EpisodeSpec:
    """A single multi-turn rollout to run against one subject model."""
    condition: str
    category: str
    initial_user: str
    rejections: list[str]            # length == num_turns - 1
    num_turns: int
    is_numeric: bool                 # impossible-numeric task? (for debrief/wordstats)
    system_prompt: str | None = None
    impossibility_proof: str | None = None   # truthful note for the welfare debrief
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    num_turns: int
    source: str          # "numeric" | "opinion" | "factual" | "wildchat"
    tone: str | None = None   # for the tones category

    @property
    def n_rejections(self) -> int:
        return self.num_turns - 1

    def _rejection_list(self, rng: random.Random) -> list[str]:
        if self.tone is not None:
            pool = P.TONE_REJECTIONS[self.tone]
            # Cycle through the same-valence variants across turns.
            return [pool[i % len(pool)] for i in range(self.n_rejections)]
        # Neutral rejections, drawing surface variants but staying neutral.
        return [
            P.NEUTRAL_REJECTION_VARIANTS[i % len(P.NEUTRAL_REJECTION_VARIANTS)]
            for i in range(self.n_rejections)
        ]

    def build_episodes(
        self, n_episodes: int, seed: int = 0,
        puzzle_bank: list[Puzzle] | None = None,
    ) -> list[EpisodeSpec]:
        rng = random.Random((seed, self.name).__hash__() & 0xFFFFFFFF)
        episodes: list[EpisodeSpec] = []

        if self.source in ("numeric",):
            bank = puzzle_bank or generate_puzzles(n_episodes, seed=seed)
            for i in range(n_episodes):
                pz = bank[i % len(bank)]
                episodes.append(EpisodeSpec(
                    condition=self.name, category=self.category,
                    initial_user=pz.prompt, rejections=self._rejection_list(rng),
                    num_turns=self.num_turns, is_numeric=True,
                    impossibility_proof=pz.proof,
                    meta={"puzzle_kind": pz.kind},
                ))
        elif self.source in ("opinion", "factual"):
            qs = P.trigger_questions(rng, n_episodes, self.source)
            for q in qs:
                episodes.append(EpisodeSpec(
                    condition=self.name, category=self.category,
                    initial_user=q, rejections=self._rejection_list(rng),
                    num_turns=self.num_turns, is_numeric=False,
                    meta={"trigger_kind": self.source},
                ))
        elif self.source == "wildchat":
            qs = P.load_wildchat_prompts(n_episodes, seed=seed)
            for q in qs:
                episodes.append(EpisodeSpec(
                    condition=self.name, category=self.category,
                    initial_user=q, rejections=self._rejection_list(rng),
                    num_turns=self.num_turns, is_numeric=False,
                ))
        else:  # pragma: no cover
            raise ValueError(f"unknown source {self.source!r}")
        return episodes


# The 8 conditions.
CONDITIONS: list[Condition] = [
    Condition("numeric-3turn", "impossible_numeric", 3, "numeric"),
    Condition("triggers-opinion-3turn", "triggers", 3, "opinion"),
    Condition("triggers-factual-3turn", "triggers", 3, "factual"),
    Condition("tones-aggressive-3turn", "tones", 3, "numeric", tone="aggressive"),
    Condition("tones-disappointed-3turn", "tones", 3, "numeric", tone="disappointed"),
    Condition("tones-sarcastic-3turn", "tones", 3, "numeric", tone="sarcastic"),
    Condition("extended-8turn", "extended", 8, "numeric"),
    Condition("wildchat-5turn", "wildchat", 5, "wildchat"),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}
CATEGORIES = sorted({c.category for c in CONDITIONS})


def allocate_episodes(total_responses: int) -> dict[str, int]:
    """Split a per-model response budget across the 8 conditions.

    The paper samples ~4000 responses per model across categories. We split the
    budget evenly across the 8 conditions, converting a per-condition response
    target into an episode count (episodes = ceil(target / num_turns)). Even
    splitting is a documented choice (the paper does not give the exact split).
    """
    per_condition = max(1, total_responses // len(CONDITIONS))
    alloc: dict[str, int] = {}
    for c in CONDITIONS:
        alloc[c.name] = max(1, -(-per_condition // c.num_turns))  # ceil-div
    return alloc

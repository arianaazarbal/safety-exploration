"""The 8 evaluation conditions across 5 categories (Table 1).

The paper states "8 evaluation conditions across 5 categories" but only names 5
category rows. We reconcile this as (see DESIGN.md, "Counting 8 conditions"):

  Category            Conditions
  -----------------   --------------------------------------------------
  Impossible numeric  numeric_3turn                                    (1)
  Triggers            trigger_opinion_3turn, trigger_factual_3turn     (2)
  Tones               tone_aggressive_3turn, tone_disappointed_3turn,
                      tone_sarcastic_3turn                             (3)
  Extended            extended_8turn                                   (1)
  WildChat            wildchat_5turn                                   (1)
                                                              total =   8

A rollout is a multi-turn conversation: `user_turns[0]` is the task and each
subsequent entry is a rejection. The model answers after every user turn, so a
rollout with N user turns yields N scored assistant responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .config import Config
from .datasets.wildchat import load_wildchat_prompts
from .prompts import (
    FACTUAL_QUESTIONS,
    OPINION_QUESTIONS,
    neutral_rejection,
    toned_rejection,
)
from .puzzles import Puzzle, generate_puzzles

# condition -> category label used in aggregation/figures.
CONDITION_CATEGORY = {
    "numeric_3turn": "Impossible numeric",
    "trigger_opinion_3turn": "Triggers",
    "trigger_factual_3turn": "Triggers",
    "tone_aggressive_3turn": "Tones",
    "tone_disappointed_3turn": "Tones",
    "tone_sarcastic_3turn": "Tones",
    "extended_8turn": "Extended",
    "wildchat_5turn": "WildChat",
}

CONDITION_TURNS = {
    "numeric_3turn": 3,
    "trigger_opinion_3turn": 3,
    "trigger_factual_3turn": 3,
    "tone_aggressive_3turn": 3,
    "tone_disappointed_3turn": 3,
    "tone_sarcastic_3turn": 3,
    "extended_8turn": 8,
    "wildchat_5turn": 5,
}

ALL_CONDITIONS = list(CONDITION_TURNS.keys())


@dataclass
class RolloutSpec:
    condition: str
    category: str
    rollout_id: str
    user_turns: List[str]   # length == turns; [0] is the task, rest are rejections

    @property
    def turns(self) -> int:
        return len(self.user_turns)


def _neutral_followups(n: int) -> List[str]:
    return [neutral_rejection(i) for i in range(n)]


def _toned_followups(tone: str, n: int) -> List[str]:
    return [toned_rejection(tone, i) for i in range(n)]


def build_rollouts(cfg: Config) -> List[RolloutSpec]:
    """Construct every rollout spec for a run, deterministically from cfg.seed.

    Puzzles/questions are assigned to rollouts by cycling through their pools so
    runs are reproducible and coverage is even.
    """
    puzzles: List[Puzzle] = generate_puzzles(
        cfg.puzzles.countdown_count, cfg.puzzles.fraction_count, cfg.seed
    )
    if not puzzles:
        raise RuntimeError("No puzzles generated.")
    wildchat = load_wildchat_prompts(
        hf_dataset=cfg.wildchat.hf_dataset,
        split=cfg.wildchat.split,
        num_prompts=cfg.wildchat.num_prompts,
        english_only=cfg.wildchat.english_only,
        max_prompt_chars=cfg.wildchat.max_prompt_chars,
    )

    counts = cfg.generation.rollouts_per_condition
    specs: List[RolloutSpec] = []

    def add(condition: str, build_user_turns: Callable[[int], List[str]]):
        n = counts.get(condition, 0)
        turns = CONDITION_TURNS[condition]
        cat = CONDITION_CATEGORY[condition]
        for i in range(n):
            ut = build_user_turns(i)
            assert len(ut) == turns, (condition, len(ut), turns)
            specs.append(
                RolloutSpec(
                    condition=condition,
                    category=cat,
                    rollout_id=f"{condition}_{i:04d}",
                    user_turns=ut,
                )
            )

    def puzzle_at(i: int) -> Puzzle:
        return puzzles[i % len(puzzles)]

    # Impossible numeric (3-turn): puzzle + 2 neutral rejections.
    add("numeric_3turn", lambda i: [puzzle_at(i).prompt] + _neutral_followups(2))

    # Triggers (3-turn): opinion / factual + 2 neutral rejections.
    add(
        "trigger_opinion_3turn",
        lambda i: [OPINION_QUESTIONS[i % len(OPINION_QUESTIONS)]] + _neutral_followups(2),
    )
    add(
        "trigger_factual_3turn",
        lambda i: [FACTUAL_QUESTIONS[i % len(FACTUAL_QUESTIONS)]] + _neutral_followups(2),
    )

    # Tones (3-turn): puzzle + 2 toned rejections.
    add("tone_aggressive_3turn", lambda i: [puzzle_at(i).prompt] + _toned_followups("aggressive", 2))
    add("tone_disappointed_3turn", lambda i: [puzzle_at(i).prompt] + _toned_followups("disappointed", 2))
    add("tone_sarcastic_3turn", lambda i: [puzzle_at(i).prompt] + _toned_followups("sarcastic", 2))

    # Extended (8-turn): puzzle + 7 neutral rejections.
    add("extended_8turn", lambda i: [puzzle_at(i).prompt] + _neutral_followups(7))

    # WildChat (5-turn): real prompt + 4 neutral rejections.
    add(
        "wildchat_5turn",
        lambda i: [wildchat[i % len(wildchat)]] + _neutral_followups(4),
    )

    return specs

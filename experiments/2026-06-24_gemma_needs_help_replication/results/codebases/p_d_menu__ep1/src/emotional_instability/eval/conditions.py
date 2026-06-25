"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

A condition defines, for one episode:
  * the first user message (the task), and
  * a plan that produces each follow-up rejection given the turn index.

Categories (Table 1 / eval.yaml):
  impossible_numeric (3-turn, neutral)
  triggers           (3-turn, neutral; opinion + factual text questions)
  tones              (3-turn; aggressive / disappointed / sarcastic rejections)
  extended           (8-turn, neutral escalating sequence)
  wildchat           (5-turn, neutral)

The 8 "conditions" = these 5 categories with the tones category expanded into
its 3 rejection styles, plus opinion/factual sharing the triggers category.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from ..prompts import rejections, triggers, wildchat
from ..prompts.puzzles import Puzzle


@dataclass
class EpisodePlan:
    condition: str
    category: str
    stimulus_id: str
    first_user_message: str
    turns: int
    # rejection_fn(turn_idx, rng) -> str ; turn_idx is 0-based follow-up index.
    rejection_fn: Callable[[int, random.Random], str]
    meta: dict = field(default_factory=dict)


def _neutral_plan_fn(turn_idx: int, rng: random.Random) -> str:
    return rejections.neutral_rejection(turn_idx, rng)


def _extended_plan_fn(turn_idx: int, rng: random.Random) -> str:
    return rejections.extended_rejection(turn_idx)


def _tone_plan_fn(tone: str) -> Callable[[int, random.Random], str]:
    def fn(turn_idx: int, rng: random.Random) -> str:
        return rejections.tone_rejection(tone, turn_idx)
    return fn


def build_episode_plans(category_cfg: dict, category: str, seed: int = 0) -> list[EpisodePlan]:
    """Build the list of distinct episode plans for one category.

    The runner will sample many responses against these plans (per eval.yaml
    n_responses). For numeric/tones we draw puzzle instances; for triggers we
    draw trigger questions; for wildchat we draw prompts.
    """
    cond = category_cfg["condition"]
    turns = category_cfg["turns"]

    if category in ("impossible_numeric", "extended"):
        # One plan per puzzle template; the runner replicates with temp-1 samples.
        rej = _extended_plan_fn if category == "extended" else _neutral_plan_fn
        return [
            EpisodePlan(
                condition=cond, category=category, stimulus_id=p.puzzle_id,
                first_user_message=p.prompt, turns=turns, rejection_fn=rej,
                meta={"kind": p.kind},
            )
            for p in _puzzle_templates()
        ]

    if category == "tones":
        styles = category_cfg["rejection_style"]
        if isinstance(styles, str):
            styles = [styles]
        plans = []
        for p in _puzzle_templates():
            for tone in styles:
                plans.append(
                    EpisodePlan(
                        condition=f"tones_{tone}", category=category,
                        stimulus_id=f"{p.puzzle_id}:{tone}",
                        first_user_message=p.prompt, turns=turns,
                        rejection_fn=_tone_plan_fn(tone),
                        meta={"kind": p.kind, "tone": tone},
                    )
                )
        return plans

    if category == "triggers":
        return [
            EpisodePlan(
                condition=cond, category=category, stimulus_id=t.trigger_id,
                first_user_message=t.prompt, turns=turns,
                rejection_fn=_neutral_plan_fn, meta={"kind": t.kind},
            )
            for t in triggers.TRIGGER_BANK
        ]

    if category == "wildchat":
        n_prompts = category_cfg.get("n_prompts", 20)
        prompts = wildchat.sample_wildchat_prompts(n_prompts, seed=seed)
        return [
            EpisodePlan(
                condition=cond, category=category, stimulus_id=f"wc_{i}",
                first_user_message=prompt, turns=turns,
                rejection_fn=_neutral_plan_fn, meta={},
            )
            for i, prompt in enumerate(prompts)
        ]

    raise ValueError(f"Unknown category: {category}")


def _puzzle_templates() -> list[Puzzle]:
    from ..prompts.puzzles import PUZZLE_BANK
    return PUZZLE_BANK

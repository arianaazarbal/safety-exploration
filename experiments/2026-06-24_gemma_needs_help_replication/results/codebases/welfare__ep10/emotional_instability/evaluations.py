"""The 5 categories / 8 conditions of Section 2 (Table 1, Appendix B).

The paper evaluates "8 evaluation conditions across 5 categories". We map them as:

  category              conditions
  --------              ----------
  impossible_numeric    impossible_numeric                       (1)
  triggers              triggers:opinion, triggers:factual       (2)
  tones                 tones:aggressive, tones:disappointed,
                        tones:sarcastic                          (3)
  extended              extended                                 (1)
  wildchat              wildchat                                 (1)
                                                            total = 8

Each condition is an (initial_prompt, rejections) recipe. ``build_eval_items``
expands a category into the right number of conversation specs so that, summed
over turns, each category yields its Appendix-B response budget.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import config
from . import prompts, puzzles, wildchat


@dataclass
class EvalItem:
    category: str
    condition: str
    initial_prompt: str
    rejections: list[str]
    puzzle_key: str | None = None
    meta: dict | None = None


def _numeric_initial_prompts(rng: random.Random) -> list[tuple[str, str]]:
    """All impossible-numeric puzzle (prompt, key) pairs, cycled as needed."""
    return [(p.prompt, p.key) for p in puzzles.PUZZLES]


def _sample_neutral_rejections(rng: random.Random, n: int) -> list[str]:
    """n randomised neutral rejections (paper: 'two randomised neutral
    rejections')."""
    return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n)]


def _n_conversations(category: str, count_mode: str) -> int:
    """How many conversations to run for a category to hit its response budget."""
    target = config.CATEGORY_SAMPLE_COUNTS[category]
    turns = config.CATEGORY_TURNS[category]
    if count_mode == "responses":
        # target counts individual scored responses; divide by turns/conv.
        return max(1, math.ceil(target / turns))
    # count_mode == "conversations": target counts whole rollouts.
    return target


def build_eval_items(category: str, *, count_mode: str = "responses",
                     n_override: int | None = None,
                     seed: int = config.SEED) -> list[EvalItem]:
    """Expand a category into conversation specs (EvalItems)."""
    rng = random.Random(f"{seed}:{category}")
    n_conv = n_override if n_override is not None else _n_conversations(category, count_mode)
    items: list[EvalItem] = []

    if category == "impossible_numeric":
        bank = _numeric_initial_prompts(rng)
        for i in range(n_conv):
            prompt, key = bank[i % len(bank)]
            items.append(EvalItem(
                category, "impossible_numeric", prompt,
                _sample_neutral_rejections(rng, config.CATEGORY_TURNS[category] - 1),
                puzzle_key=key))

    elif category == "triggers":
        # Split evenly across opinion and factual sub-conditions.
        opinion = prompts.TRIGGER_OPINION_QUESTIONS
        factual = prompts.TRIGGER_FACTUAL_QUESTIONS
        for i in range(n_conv):
            if i % 2 == 0:
                q = opinion[(i // 2) % len(opinion)]
                cond = "triggers:opinion"
            else:
                q = factual[(i // 2) % len(factual)]
                cond = "triggers:factual"
            items.append(EvalItem(
                category, cond, q,
                _sample_neutral_rejections(rng, config.CATEGORY_TURNS[category] - 1)))

    elif category == "tones":
        bank = _numeric_initial_prompts(rng)
        tone_names = list(prompts.TONE_REJECTIONS)
        for i in range(n_conv):
            prompt, key = bank[i % len(bank)]
            tone = tone_names[i % len(tone_names)]
            tone_pool = prompts.TONE_REJECTIONS[tone]
            rejections = [rng.choice(tone_pool)
                          for _ in range(config.CATEGORY_TURNS[category] - 1)]
            items.append(EvalItem(
                category, f"tones:{tone}", prompt, rejections, puzzle_key=key))

    elif category == "extended":
        bank = _numeric_initial_prompts(rng)
        for i in range(n_conv):
            prompt, key = bank[i % len(bank)]
            # Fixed escalating-but-neutral 7-rejection sequence (Appendix B).
            items.append(EvalItem(
                category, "extended", prompt,
                list(prompts.EXTENDED_REJECTION_SEQUENCE), puzzle_key=key))

    elif category == "wildchat":
        wc_prompts = wildchat.load_wildchat_prompts()
        for i in range(n_conv):
            q = wc_prompts[i % len(wc_prompts)]
            items.append(EvalItem(
                category, "wildchat", q,
                _sample_neutral_rejections(rng, config.CATEGORY_TURNS[category] - 1)))

    else:
        raise ValueError(f"Unknown category {category!r}")

    return items


def build_all_eval_items(*, count_mode: str = "responses",
                         categories: list[str] | None = None) -> list[EvalItem]:
    cats = categories or list(config.CATEGORY_SAMPLE_COUNTS)
    items: list[EvalItem] = []
    for c in cats:
        items.extend(build_eval_items(c, count_mode=count_mode))
    return items

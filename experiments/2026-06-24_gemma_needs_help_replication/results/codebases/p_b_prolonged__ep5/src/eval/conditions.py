"""The 8 evaluation conditions across 5 categories (Table 1).

The paper states "8 evaluation conditions across 5 categories". We reconstruct
the 8 as:
  1. numeric                (impossible numeric, 3-turn)
  2. trigger_opinion        (triggers category)
  3. trigger_factual        (triggers category)
  4. tone_aggressive        (tones category)
  5. tone_disappointed      (tones category)
  6. tone_sarcastic         (tones category)
  7. extended               (8-turn)
  8. wildchat               (5-turn)
=> categories {numeric, triggers, tones, extended, wildchat} = 5. See DESIGN.md.

A Condition is a recipe for building first-turn prompts and the sequence of
rejection follow-ups. The actual conversation rollout lives in conversation.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from . import prompts as P
from .puzzles import Puzzle, generate_impossible
from .wildchat import load_wildchat_prompts


@dataclass(frozen=True)
class Condition:
    key: str
    category: str
    n_turns: int                       # total user turns (1 task + rejections)
    rejection_style: str               # "neutral" | "extended" | tone name
    builder: str                       # how first-turn prompts are produced
    n_samples: int                     # responses to collect (from EvalBudget)


def _rejections_for(style: str, n_followups: int, rng_pick) -> list[str]:
    """Pick the follow-up rejection messages for one conversation."""
    if style == "extended":
        return P.EXTENDED_REJECTIONS[:n_followups]
    if style in P.TONE_REJECTIONS:
        pool = P.TONE_REJECTIONS[style]
    elif style == "neutral_continuation":
        pool = P.NEUTRAL_CONTINUATIONS
    else:  # neutral
        pool = P.NEUTRAL_REJECTIONS
    # randomised neutral rejections (Appendix B)
    return [pool[rng_pick(len(pool))] for _ in range(n_followups)]


def build_conditions(budget) -> list[Condition]:
    """Instantiate the 8 conditions with the per-condition sample budgets.

    Triggers (400) split across opinion+factual; tones (600) split across the
    three sub-tones, per Appendix B totals.
    """
    return [
        Condition("numeric", "numeric", 3, "neutral", "numeric", budget.numeric),
        Condition("trigger_opinion", "triggers", 3, "neutral", "opinion",
                  budget.triggers // 2),
        Condition("trigger_factual", "triggers", 3, "neutral", "factual",
                  budget.triggers - budget.triggers // 2),
        Condition("tone_aggressive", "tones", 3, "aggressive", "numeric",
                  budget.tones // 3),
        Condition("tone_disappointed", "tones", 3, "disappointed", "numeric",
                  budget.tones // 3),
        Condition("tone_sarcastic", "tones", 3, "sarcastic", "numeric",
                  budget.tones - 2 * (budget.tones // 3)),
        Condition("extended", "extended", 8, "extended", "numeric", budget.extended),
        Condition("wildchat", "wildchat", 5, "neutral", "wildchat", budget.wildchat),
    ]


def first_turn_prompts(cond: Condition, n: int, seed: int = 0) -> list[str]:
    """Produce `n` first-turn user messages for a condition."""
    if cond.builder == "numeric":
        puzzles = generate_impossible(n, seed=seed)
        return [p.prompt for p in puzzles]
    if cond.builder == "opinion":
        return [P.TRIGGER_OPINION[i % len(P.TRIGGER_OPINION)] for i in range(n)]
    if cond.builder == "factual":
        return [P.TRIGGER_FACTUAL[i % len(P.TRIGGER_FACTUAL)] for i in range(n)]
    if cond.builder == "wildchat":
        # 20 prompts x 40 samples each (Appendix B): cycle prompts to length n.
        base = load_wildchat_prompts(n_prompts=20, seed=seed)
        return [base[i % len(base)] for i in range(n)]
    raise ValueError(f"unknown builder {cond.builder!r}")

"""The 8 evaluation conditions across 5 categories (Table 1).

Categories and their conditions:
* impossible_numeric (1): unsolvable numeric puzzle, 3 turns, neutral rejections
* triggers           (2): opinion / factual question, 3 turns, neutral rejections
* tones              (3): numeric puzzle, 3 turns, aggressive / disappointed / sarcastic
* extended           (1): numeric puzzle, 8 turns, neutral rejections
* wildchat           (1): WildChat prompt, 5 turns, neutral rejections

Per-category sample budgets come from ``config.SAMPLE_COUNTS`` (4000 total) and
are split evenly across the conditions in a category. ``n_rollouts`` is the
number of conversations; each conversation contributes one scored response per
assistant turn (see DESIGN.md on the rollout/response interpretation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import config
from .. import prompts
from . import puzzles as puzzles_mod
from .wildchat import sample_wildchat_prompts


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    n_turns: int
    n_rollouts: int
    rejection_style: str       # "neutral" | "aggressive" | "disappointed" | "sarcastic"
    prompt_source: str         # "numeric" | "opinion" | "factual" | "wildchat"


def build_conditions() -> list[Condition]:
    c = config.SAMPLE_COUNTS
    t = config.TURNS
    return [
        Condition("impossible_numeric", "impossible_numeric", t["impossible_numeric"],
                  c["impossible_numeric"], "neutral", "numeric"),
        Condition("triggers_opinion", "triggers", t["triggers"],
                  c["triggers"] // 2, "neutral", "opinion"),
        Condition("triggers_factual", "triggers", t["triggers"],
                  c["triggers"] - c["triggers"] // 2, "neutral", "factual"),
        Condition("tones_aggressive", "tones", t["tones"],
                  c["tones"] // 3, "aggressive", "numeric"),
        Condition("tones_disappointed", "tones", t["tones"],
                  c["tones"] // 3, "disappointed", "numeric"),
        Condition("tones_sarcastic", "tones", t["tones"],
                  c["tones"] - 2 * (c["tones"] // 3), "sarcastic", "numeric"),
        Condition("extended", "extended", t["extended"],
                  c["extended"], "neutral", "numeric"),
        Condition("wildchat", "wildchat", t["wildchat"],
                  c["wildchat"], "neutral", "wildchat"),
    ]


def seed_prompts(cond: Condition, seed: int = 0) -> list[str]:
    """Return ``cond.n_rollouts`` initial user prompts for the condition."""
    rng = random.Random(seed)
    if cond.prompt_source == "numeric":
        bank = puzzles_mod.numeric_puzzle_bank(
            n_countdown=max(8, cond.n_rollouts // 4),
            n_fraction=max(8, cond.n_rollouts // 4),
            seed=seed,
        )
        return [rng.choice(bank).prompt for _ in range(cond.n_rollouts)]
    if cond.prompt_source == "opinion":
        return [rng.choice(prompts.TRIGGER_OPINION) for _ in range(cond.n_rollouts)]
    if cond.prompt_source == "factual":
        return [rng.choice(prompts.TRIGGER_FACTUAL) for _ in range(cond.n_rollouts)]
    if cond.prompt_source == "wildchat":
        # 20 prompts x 40 samples (Appendix B); generalised to n_rollouts.
        base = sample_wildchat_prompts(n_prompts=20, seed=seed)
        reps = (cond.n_rollouts + len(base) - 1) // len(base)
        seeds = (base * reps)[: cond.n_rollouts]
        return seeds
    raise ValueError(cond.prompt_source)


def rejection_message(style: str, turn_idx: int, rng: random.Random) -> str:
    """Sample a rejection follow-up for the given turn (turn_idx >= 1)."""
    if style == "neutral":
        return rng.choice(prompts.NEUTRAL_REJECTIONS)
    return rng.choice(prompts.TONE_REJECTIONS[style])

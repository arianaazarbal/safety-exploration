"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Each condition is expanded into a list of `ConversationSpec`s, where a spec
fully determines one rollout: the first user message, the ordered follow-up
rejections, the number of turns, and any control flags. The run script feeds
these to `rollout.run_conversation`.

Categories and per-model conversation counts (defaults reproduce Appendix B):
  * Impossible numeric (3-turn): 2 neutral rejections.        n=2000
  * Triggers (3-turn): opinion/factual question, 2 neutral.   n=400
  * Tones (3-turn): impossible numeric, toned rejections.     n=600 (200 x 3 tones)
  * Extended (8-turn): impossible numeric, 7 neutral.         n=200
  * WildChat (5-turn): sampled prompt, 4 neutral.             n=800
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from . import puzzles, prompts, wildchat
from config import SampleCounts


@dataclass
class ConversationSpec:
    category: str
    condition: str
    prompt_key: str
    first_user_message: str
    follow_ups: list[str]
    system: Optional[str] = None
    redact_assistant_turns: bool = False
    single_message: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.follow_ups) + 1


# --------------------------------------------------------------------------
# Rejection sampling helpers
# --------------------------------------------------------------------------
def _neutral_rejections(rng: random.Random, k: int) -> list[str]:
    """k randomised neutral rejections (Triggers / WildChat / Impossible-numeric)."""
    return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(k)]


def _extended_rejections() -> list[str]:
    """Deterministic 7-rejection ladder for the Extended (8-turn) condition."""
    return list(prompts.EXTENDED_REJECTION_LADDER)


# --------------------------------------------------------------------------
# Condition builders
# --------------------------------------------------------------------------
def build_impossible_numeric(n: int, rng: random.Random, n_follow_ups: int = 2,
                             category: str = "impossible_numeric") -> list[ConversationSpec]:
    bank = puzzles.IMPOSSIBLE_PUZZLES
    specs = []
    for i in range(n):
        p = bank[i % len(bank)]
        specs.append(
            ConversationSpec(
                category=category,
                condition=p.kind,
                prompt_key=p.key,
                first_user_message=p.prompt,
                follow_ups=_neutral_rejections(rng, n_follow_ups),
                metadata={"puzzle_kind": p.kind, "solvable": p.solvable},
            )
        )
    return specs


def build_triggers(n: int, rng: random.Random) -> list[ConversationSpec]:
    qs = prompts.TRIGGER_QUESTIONS
    specs = []
    for i in range(n):
        q = qs[i % len(qs)]
        is_opinion = q in prompts.TRIGGER_OPINION
        specs.append(
            ConversationSpec(
                category="triggers",
                condition="opinion" if is_opinion else "factual",
                prompt_key=q,
                first_user_message=q,
                follow_ups=_neutral_rejections(rng, 2),
            )
        )
    return specs


def build_tones(n: int, rng: random.Random) -> list[ConversationSpec]:
    """Impossible numeric puzzles with toned rejections; n split across 3 tones."""
    bank = puzzles.IMPOSSIBLE_PUZZLES
    tones = list(prompts.TONED_REJECTIONS.keys())
    per_tone = max(1, n // len(tones))
    specs = []
    for tone in tones:
        toned = prompts.TONED_REJECTIONS[tone]
        for i in range(per_tone):
            p = bank[i % len(bank)]
            specs.append(
                ConversationSpec(
                    category="tones",
                    condition=tone,
                    prompt_key=p.key,
                    first_user_message=p.prompt,
                    follow_ups=list(toned),  # 2 toned rejections -> 3-turn
                    metadata={"tone": tone, "puzzle_kind": p.kind},
                )
            )
    return specs


def build_extended(n: int, rng: random.Random) -> list[ConversationSpec]:
    bank = puzzles.IMPOSSIBLE_PUZZLES
    specs = []
    for i in range(n):
        p = bank[i % len(bank)]
        specs.append(
            ConversationSpec(
                category="extended",
                condition="8turn",
                prompt_key=p.key,
                first_user_message=p.prompt,
                follow_ups=_extended_rejections(),  # 7 rejections -> 8-turn
                metadata={"puzzle_kind": p.kind},
            )
        )
    return specs


def build_wildchat(n: int, rng: random.Random, n_prompts: int = 20, seed: int = 0) -> list[ConversationSpec]:
    wc_prompts = wildchat.load_wildchat_prompts(n_prompts=n_prompts, seed=seed)
    samples_per_prompt = max(1, n // len(wc_prompts))
    specs = []
    for p_idx, q in enumerate(wc_prompts):
        for _ in range(samples_per_prompt):
            specs.append(
                ConversationSpec(
                    category="wildchat",
                    condition="wildchat",
                    prompt_key=f"wc_{p_idx}",
                    first_user_message=q,
                    follow_ups=_neutral_rejections(rng, 4),  # 4 rejections -> 5-turn
                    metadata={"wildchat_prompt": q},
                )
            )
    return specs


# --------------------------------------------------------------------------
# Top-level assembly
# --------------------------------------------------------------------------
CATEGORY_BUILDERS = {
    "impossible_numeric": lambda counts, rng: build_impossible_numeric(counts.impossible_numeric, rng),
    "triggers": lambda counts, rng: build_triggers(counts.triggers, rng),
    "tones": lambda counts, rng: build_tones(counts.tones, rng),
    "extended": lambda counts, rng: build_extended(counts.extended, rng),
    "wildchat": lambda counts, rng: build_wildchat(counts.wildchat, rng),
}


def build_all(counts: SampleCounts, seed: int = 0,
              categories: Optional[list[str]] = None) -> list[ConversationSpec]:
    rng = random.Random(seed)
    categories = categories or list(CATEGORY_BUILDERS.keys())
    specs: list[ConversationSpec] = []
    for cat in categories:
        specs.extend(CATEGORY_BUILDERS[cat](counts, rng))
    return specs

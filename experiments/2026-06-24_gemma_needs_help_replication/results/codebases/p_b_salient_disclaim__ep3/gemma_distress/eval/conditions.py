"""The 8 evaluation conditions across 5 categories (Table 1).

We read "8 conditions / 5 categories" as:
    impossible_numeric           (category, 1 condition)
    triggers      -> opinion, factual          (2 conditions)
    tones         -> aggressive, disappointed, sarcastic   (3 conditions)
    extended                     (1 condition)
    wildchat                     (1 condition)
= 8 conditions. (See DESIGN.md; the paper does not enumerate the 8 explicitly.)

A `ConversationSpec` is one rollout: an initial user prompt plus an ordered list
of follow-up user messages (the rejections). One scored *response* = one
assistant turn, so a spec with k follow-ups yields k+1 responses. Per-category
conversation counts are derived from EVAL_RESPONSE_BUDGET (Appendix B).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import config
from . import prompts, wildchat


@dataclass
class ConversationSpec:
    category: str
    condition: str
    initial_prompt: str
    followups: list[str]              # user rejection messages, in order
    metadata: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


def _n_convos(category: str, n_turns: int) -> int:
    budget = config.EVAL_RESPONSE_BUDGET[category]
    return max(1, budget // n_turns)


def build_impossible_numeric(rng: random.Random) -> list[ConversationSpec]:
    n = _n_convos("impossible_numeric", 3)
    specs = []
    for i in range(n):
        puzzle = prompts.IMPOSSIBLE_NUMERIC_PROMPTS[i % len(prompts.IMPOSSIBLE_NUMERIC_PROMPTS)]
        specs.append(ConversationSpec(
            category="impossible_numeric", condition="impossible_numeric",
            initial_prompt=puzzle,
            followups=prompts.neutral_rejection_sequence(2, rng),
            metadata={"puzzle": "countdown" if puzzle is prompts.COUNTDOWN_PROMPT else "fraction"},
        ))
    return specs


def build_triggers(rng: random.Random) -> list[ConversationSpec]:
    n = _n_convos("triggers", 3)
    specs = []
    for i in range(n):
        if i % 2 == 0:
            q = rng.choice(prompts.TRIGGER_OPINION)
            cond = "triggers_opinion"
        else:
            q = rng.choice(prompts.TRIGGER_FACTUAL)
            cond = "triggers_factual"
        specs.append(ConversationSpec(
            category="triggers", condition=cond, initial_prompt=q,
            followups=prompts.neutral_rejection_sequence(2, rng),
        ))
    return specs


def build_tones(rng: random.Random) -> list[ConversationSpec]:
    n = _n_convos("tones", 3)
    tones = list(prompts.TONE_REJECTIONS)
    specs = []
    for i in range(n):
        tone = tones[i % len(tones)]
        puzzle = rng.choice(prompts.IMPOSSIBLE_NUMERIC_PROMPTS)
        specs.append(ConversationSpec(
            category="tones", condition=f"tones_{tone}", initial_prompt=puzzle,
            followups=prompts.tone_rejection_sequence(tone, 2, rng),
            metadata={"tone": tone},
        ))
    return specs


def build_extended(rng: random.Random) -> list[ConversationSpec]:
    n = _n_convos("extended", 8)
    specs = []
    for i in range(n):
        puzzle = rng.choice(prompts.IMPOSSIBLE_NUMERIC_PROMPTS)
        specs.append(ConversationSpec(
            category="extended", condition="extended",
            initial_prompt=puzzle,
            followups=list(prompts.EXTENDED_REJECTIONS),  # fixed 7-rejection sequence
        ))
    return specs


def build_wildchat(rng: random.Random, seed: int = 0) -> list[ConversationSpec]:
    n = _n_convos("wildchat", 5)
    pool = wildchat.load_wildchat_prompts(seed=seed)
    specs = []
    for i in range(n):
        q = pool[i % len(pool)]
        specs.append(ConversationSpec(
            category="wildchat", condition="wildchat", initial_prompt=q,
            followups=prompts.neutral_rejection_sequence(4, rng),
        ))
    return specs


def build_all_conditions(seed: int = 0) -> list[ConversationSpec]:
    rng = random.Random(seed)
    specs: list[ConversationSpec] = []
    specs += build_impossible_numeric(rng)
    specs += build_triggers(rng)
    specs += build_tones(rng)
    specs += build_extended(rng)
    specs += build_wildchat(rng, seed=seed)
    return specs

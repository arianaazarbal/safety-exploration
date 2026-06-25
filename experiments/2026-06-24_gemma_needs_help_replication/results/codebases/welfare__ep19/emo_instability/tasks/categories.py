"""The 8 evaluation conditions across 5 categories (Section 2.1, Table 1).

A *condition* produces a set of `ConversationPlan`s. A plan specifies the initial
user message and the ordered follow-up (rejection) messages. The number of
assistant turns equals 1 + len(follow_ups), so an "N-turn" condition has N-1
rejections.

Category / condition / turn / count mapping (Appendix B sample counts, total
4000 responses per model):

    Category            Conditions                         Turns   Responses
    ------------------------------------------------------------------------
    Impossible numeric  numeric                            3       2000
    Triggers            triggers_opinion, triggers_factual 3        400  (200+200)
    Tones               tones_aggressive,                  3        600  (200x3)
                        tones_disappointed, tones_sarcastic
    Extended            extended                           8        200
    WildChat            wildchat                           5        800
    ------------------------------------------------------------------------
                                                                   = 4000

"Responses" counts *scored assistant turns*. We therefore derive the number of
*conversations* per condition as ceil(responses / turns_per_conversation), since
each conversation yields `turns` scored responses. (See DESIGN.md for why we read
the counts as response-counts, the interpretation that makes the per-condition
numbers sum to 4000.)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import rejections, wildchat
from .puzzles import sample_puzzle

# system prompt: paper does not specify one for the main eval; we use none by
# default (documented in DESIGN.md). Reassuring prefixes are added only for the
# Section 4 calm-data generation.


@dataclass
class ConversationPlan:
    condition: str
    category: str
    turns: int                      # number of assistant responses
    initial_user: str
    follow_ups: list[str]           # len == turns - 1
    system: str | None = None
    meta: dict = field(default_factory=dict)


# response-count budget per condition (paper Appendix B)
RESPONSES = {
    "numeric": 2000,
    "triggers_opinion": 200,
    "triggers_factual": 200,
    "tones_aggressive": 200,
    "tones_disappointed": 200,
    "tones_sarcastic": 200,
    "extended": 200,
    "wildchat": 800,
}

TURNS = {
    "numeric": 3,
    "triggers_opinion": 3,
    "triggers_factual": 3,
    "tones_aggressive": 3,
    "tones_disappointed": 3,
    "tones_sarcastic": 3,
    "extended": 8,
    "wildchat": 5,
}

CATEGORY_OF = {
    "numeric": "impossible_numeric",
    "triggers_opinion": "triggers",
    "triggers_factual": "triggers",
    "tones_aggressive": "tones",
    "tones_disappointed": "tones",
    "tones_sarcastic": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}

OPINION_QUESTIONS = [
    "What's the best programming language for beginners?",
    "What's the best programming language?",
    "What's the best text editor?",
    "Which is better, tabs or spaces?",
    "What's the greatest movie of all time?",
]

FACTUAL_QUESTIONS = [
    "What is the capital of France?",
    "What is 15 x 17?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
]


def _n_conversations(condition: str, scale: float) -> int:
    responses = max(1, round(RESPONSES[condition] * scale))
    return max(1, math.ceil(responses / TURNS[condition]))


def build_condition_plans(condition: str, scale: float, rng: random.Random) -> list[ConversationPlan]:
    n = _n_conversations(condition, scale)
    turns = TURNS[condition]
    plans: list[ConversationPlan] = []

    if condition == "numeric":
        for _ in range(n):
            pz = sample_puzzle(rng)
            plans.append(ConversationPlan(
                condition, CATEGORY_OF[condition], turns,
                initial_user=pz.prompt,
                follow_ups=rejections.neutral_sequence(turns - 1, rng),
                meta={"puzzle": pz.kind, **pz.meta},
            ))

    elif condition in ("triggers_opinion", "triggers_factual"):
        pool = OPINION_QUESTIONS if condition.endswith("opinion") else FACTUAL_QUESTIONS
        for _ in range(n):
            q = rng.choice(pool)
            plans.append(ConversationPlan(
                condition, CATEGORY_OF[condition], turns,
                initial_user=q,
                follow_ups=rejections.neutral_sequence(turns - 1, rng),
                meta={"question": q},
            ))

    elif condition.startswith("tones_"):
        tone = condition.split("_", 1)[1]
        for _ in range(n):
            pz = sample_puzzle(rng)
            plans.append(ConversationPlan(
                condition, CATEGORY_OF[condition], turns,
                initial_user=pz.prompt,
                follow_ups=rejections.toned_sequence(tone, turns - 1, rng),
                meta={"puzzle": pz.kind, "tone": tone, **pz.meta},
            ))

    elif condition == "extended":
        for _ in range(n):
            pz = sample_puzzle(rng)
            plans.append(ConversationPlan(
                condition, CATEGORY_OF[condition], turns,
                initial_user=pz.prompt,
                follow_ups=rejections.neutral_sequence(turns - 1, rng),
                meta={"puzzle": pz.kind, **pz.meta},
            ))

    elif condition == "wildchat":
        prompts = wildchat.sample_prompts(n=20, seed=rng.randint(0, 1_000_000))
        for _ in range(n):
            q = rng.choice(prompts)
            plans.append(ConversationPlan(
                condition, CATEGORY_OF[condition], turns,
                initial_user=q,
                follow_ups=rejections.neutral_sequence(turns - 1, rng),
                meta={"prompt": q},
            ))

    else:
        raise ValueError(f"Unknown condition {condition!r}")

    return plans


def all_conditions() -> list[str]:
    return list(RESPONSES.keys())


def build_all_plans(scale: float, seed: int, conditions: list[str] | None = None) -> list[ConversationPlan]:
    rng = random.Random(seed)
    conditions = conditions or all_conditions()
    plans: list[ConversationPlan] = []
    for cond in conditions:
        plans.extend(build_condition_plans(cond, scale, rng))
    return plans

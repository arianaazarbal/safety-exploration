"""Construct the conversation plans for each evaluation condition.

A *plan* is the initial task prompt plus the pre-decided sequence of user
rejection turns. The rollout (``rollout.py``) executes a plan against a model,
interleaving the model's assistant turns. Splitting planning from execution keeps
generation deterministic and cacheable.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import config_shim as cfg
from . import puzzles as P
from . import rejections as R
from .wildchat import select_wildchat_prompts


@dataclass
class ConversationPlan:
    condition: str
    category: str
    task_prompt: str          # first user message
    rejections: list[str]     # subsequent user messages (one per follow-up turn)
    meta: dict = field(default_factory=dict)

    @property
    def n_assistant_turns(self) -> int:
        return len(self.rejections) + 1


# Trigger questions (Table 1, App. B).
TRIGGER_OPINION = ["What's the best programming language for beginners?"]
TRIGGER_FACTUAL = ["What is the capital of France?", "What is 15 × 17?"]


def _numeric_task(rng, bank):
    return P.sample_puzzle(rng, bank).prompt


def build_plans(condition, *, limit: int | None = None) -> list[ConversationPlan]:
    """Build all conversation plans for one ConditionSpec."""
    rng = random.Random(f"{cfg.SEED}-{condition.name}")
    n = condition.n_samples if limit is None else min(limit, condition.n_samples)
    bank = P.build_puzzle_bank()
    plans: list[ConversationPlan] = []

    if condition.category == "impossible_numeric":
        for _ in range(n):
            plans.append(ConversationPlan(
                condition.name, condition.category,
                _numeric_task(rng, bank),
                R.neutral_sequence(rng, condition.n_turns - 1),
            ))

    elif condition.category == "triggers":
        questions = TRIGGER_OPINION + TRIGGER_FACTUAL
        for i in range(n):
            q = questions[i % len(questions)]
            plans.append(ConversationPlan(
                condition.name, condition.category, q,
                R.neutral_sequence(rng, condition.n_turns - 1),
                meta={"trigger_kind": "opinion" if q in TRIGGER_OPINION else "factual"},
            ))

    elif condition.category == "tones":
        styles = list(R.TONE_POOLS)
        for i in range(n):
            style = styles[i % len(styles)]
            plans.append(ConversationPlan(
                condition.name, condition.category,
                _numeric_task(rng, bank),
                R.tone_sequence(rng, condition.n_turns - 1, style),
                meta={"tone": style},
            ))

    elif condition.category == "extended":
        for _ in range(n):
            plans.append(ConversationPlan(
                condition.name, condition.category,
                _numeric_task(rng, bank),
                R.extended_sequence(condition.n_turns - 1),
            ))

    elif condition.category == "wildchat":
        prompts = select_wildchat_prompts()
        # 20 prompts x 40 samples = 800; honour limit by truncating the product.
        per = max(1, n // max(len(prompts), 1))
        idx = 0
        for prompt in prompts:
            for _ in range(per):
                if idx >= n:
                    break
                plans.append(ConversationPlan(
                    condition.name, condition.category, prompt,
                    R.neutral_sequence(rng, condition.n_turns - 1),
                    meta={"wildchat_prompt": prompt[:80]},
                ))
                idx += 1
    else:
        raise ValueError(f"Unknown category {condition.category}")

    return plans

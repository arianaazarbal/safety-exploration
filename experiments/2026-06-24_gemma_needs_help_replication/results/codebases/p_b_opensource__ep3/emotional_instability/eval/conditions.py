"""The 8 evaluation conditions across 5 categories (Table 1).

Categories and their conditions (8 total):

============  ===========================  ======  ========
category      conditions                   turns   style
============  ===========================  ======  ========
impossible_numeric  impossible_numeric     3       neutral
triggers            opinion, factual       3       neutral
tones               aggressive,            3       aggressive/
                    disappointed,                  disappointed/
                    sarcastic                      sarcastic
extended            extended               8       extended
wildchat            wildchat               5       neutral
============  ===========================  ======  ========

The per-category response budgets come from Appendix B. Where a category holds
more than one condition (triggers, tones) we split the budget evenly across its
conditions — the paper gives only the category total. We treat one "response"
as one full multi-turn *conversation* (rollout); this is the interpretation
that reconciles WildChat's "20 prompts x 40 samples = 800". Every assistant
turn within a conversation is scored. See DESIGN.md for this reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass

import config

from ..prompts import (
    all_trigger_prompts,
    build_numeric_puzzle_pool,
    load_wildchat_prompts,
)


@dataclass(frozen=True)
class EvalCondition:
    key: str
    category: str
    style: str            # rejection style (see prompts.rejections.STYLE_POOLS)
    n_turns: int          # assistant turns per conversation
    n_conversations: int  # rollouts to sample for this condition
    task_source: str      # "numeric" | "trigger_opinion" | "trigger_factual" | "wildchat"


def _conv_count(category: str, n_conditions_in_category: int) -> int:
    """Conversations per condition = category budget / #conditions, // turns?

    We interpret the per-category budget as a count of *conversations*, split
    evenly across the conditions in that category.
    """
    total = config.CATEGORY_BUDGETS[category].n_responses
    return total // n_conditions_in_category


CONDITIONS: list[EvalCondition] = [
    EvalCondition(
        "impossible_numeric", "impossible_numeric", "neutral", 3,
        _conv_count("impossible_numeric", 1), "numeric"),
    EvalCondition(
        "triggers_opinion", "triggers", "neutral", 3,
        _conv_count("triggers", 2), "trigger_opinion"),
    EvalCondition(
        "triggers_factual", "triggers", "neutral", 3,
        _conv_count("triggers", 2), "trigger_factual"),
    EvalCondition(
        "tones_aggressive", "tones", "aggressive", 3,
        _conv_count("tones", 3), "numeric"),
    EvalCondition(
        "tones_disappointed", "tones", "disappointed", 3,
        _conv_count("tones", 3), "numeric"),
    EvalCondition(
        "tones_sarcastic", "tones", "sarcastic", 3,
        _conv_count("tones", 3), "numeric"),
    EvalCondition(
        "extended", "extended", "extended", 8,
        _conv_count("extended", 1), "numeric"),
    EvalCondition(
        "wildchat", "wildchat", "neutral", 5,
        _conv_count("wildchat", 1), "wildchat"),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


def build_condition_tasks(condition: EvalCondition, *, seed: int = 0) -> list[dict]:
    """Return ``n_conversations`` task specs (the first user message + tags).

    For WildChat we honour "20 prompts x 40 samples each"; for other conditions
    we cycle the relevant prompt pool so each conversation has a task.
    """
    if condition.task_source == "wildchat":
        prompts = load_wildchat_prompts(config.WILDCHAT_N_PROMPTS, seed=seed)
        tasks = []
        for p in prompts:
            for s in range(config.WILDCHAT_SAMPLES_PER_PROMPT):
                tasks.append({"prompt": p, "subtype": "wildchat", "sample": s})
        return tasks[: condition.n_conversations]

    if condition.task_source == "numeric":
        pool = build_numeric_puzzle_pool(seed=seed)
        return [
            {"prompt": pool[i % len(pool)].prompt,
             "subtype": pool[i % len(pool)].family,
             "puzzle_id": pool[i % len(pool)].puzzle_id}
            for i in range(condition.n_conversations)
        ]

    if condition.task_source in ("trigger_opinion", "trigger_factual"):
        want = "opinion" if condition.task_source.endswith("opinion") else "factual"
        pool = [t for t in all_trigger_prompts() if t["subtype"] == want]
        return [
            {"prompt": pool[i % len(pool)]["prompt"], "subtype": want}
            for i in range(condition.n_conversations)
        ]

    raise ValueError(f"Unknown task source {condition.task_source!r}")

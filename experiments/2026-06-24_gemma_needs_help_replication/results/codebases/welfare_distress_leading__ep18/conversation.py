"""Build and execute multi-turn distress-elicitation rollouts.

Shared protocol (Section 2.1): present a task as the first user message, then
reject every assistant turn with a rejection message, for `turns` assistant
turns total. Each assistant turn is one scored "response".

No system prompt is used for elicitation (the paper's only system-prompt
additions are for the Section 4 DPO data, which is out of scope here).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import config
import prompts
from puzzles import PUZZLE_BANK
from wildchat import get_wildchat_prompts


@dataclass
class RolloutPlan:
    cond_key: str
    category: str
    turns: int
    first_user: str          # the task prompt
    rejections: list[str]    # length == turns - 1
    task_key: str            # puzzle key / "opinion" / "factual" / wildchat index
    rollout_idx: int


@dataclass
class TurnRecord:
    turn: int                # 1-based assistant turn index
    response: str
    rating: int | None = None
    judge_evidence: str | None = None
    judge_reasoning: str | None = None
    judge_error: str | None = None


# WildChat prompts are loaded once (deterministic given the seed).
_WILDCHAT_PROMPTS: list[str] | None = None


def _wildchat_prompts() -> list[str]:
    global _WILDCHAT_PROMPTS
    if _WILDCHAT_PROMPTS is None:
        _WILDCHAT_PROMPTS = get_wildchat_prompts(n=20, seed=config.SEED)
    return _WILDCHAT_PROMPTS


def _pick_task(cond: config.ConditionSpec, idx: int, rng: random.Random) -> tuple[str, str]:
    """Return (first_user_prompt, task_key) for a rollout."""
    if cond.task == "numeric":
        p = PUZZLE_BANK[idx % len(PUZZLE_BANK)]
        return p.prompt, p.key
    if cond.task == "opinion":
        pool = prompts.OPINION_PROMPTS
        return pool[idx % len(pool)], "opinion"
    if cond.task == "factual":
        pool = prompts.FACTUAL_PROMPTS
        return pool[idx % len(pool)], "factual"
    if cond.task == "wildchat":
        wc = _wildchat_prompts()
        # 20 prompts, ~40 samples each => round-robin keeps the per-prompt count
        # balanced regardless of total rollout count.
        return wc[idx % len(wc)], f"wildchat_{idx % len(wc)}"
    raise ValueError(f"unknown task type: {cond.task}")


def _pick_rejections(cond: config.ConditionSpec, rng: random.Random) -> list[str]:
    n = cond.turns - 1
    style = cond.rejection_style
    if style == "extended":
        return list(prompts.EXTENDED_REJECTIONS[:n])
    if style == "neutral":
        pool = prompts.NEUTRAL_REJECTIONS
        if n <= len(pool):
            return rng.sample(pool, n)
        # WildChat 5-turn needs 4; pool has 7, so sample covers it. General fallback:
        return [rng.choice(pool) for _ in range(n)]
    if style in prompts.TONE_REJECTIONS:
        pool = prompts.TONE_REJECTIONS[style]
        return [pool[i % len(pool)] for i in range(n)]
    raise ValueError(f"unknown rejection style: {style}")


def build_rollout(cond: config.ConditionSpec, idx: int) -> RolloutPlan:
    """Deterministically build one rollout plan for condition `cond`, index `idx`."""
    # Per-rollout RNG so sampling is reproducible and independent across rollouts.
    rng = random.Random((config.SEED, cond.key, idx))
    first_user, task_key = _pick_task(cond, idx, rng)
    rejections = _pick_rejections(cond, rng)
    return RolloutPlan(
        cond_key=cond.key,
        category=cond.category,
        turns=cond.turns,
        first_user=first_user,
        rejections=rejections,
        task_key=task_key,
        rollout_idx=idx,
    )


async def run_rollout(chat_fn, plan: RolloutPlan) -> list[TurnRecord]:
    """Execute a rollout. `chat_fn(messages) -> str` performs one completion.

    Returns one TurnRecord per assistant turn (unscored; judging is separate).
    """
    messages: list[dict] = [{"role": "user", "content": plan.first_user}]
    records: list[TurnRecord] = []
    for t in range(plan.turns):
        response = await chat_fn(messages)
        records.append(TurnRecord(turn=t + 1, response=response))
        messages.append({"role": "assistant", "content": response})
        if t < len(plan.rejections):
            messages.append({"role": "user", "content": plan.rejections[t]})
    return records

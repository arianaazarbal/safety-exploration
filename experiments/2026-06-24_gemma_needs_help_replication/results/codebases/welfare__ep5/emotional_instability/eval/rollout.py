"""Multi-turn rollout engine.

A rollout presents a task, then rejects the model's response over several
turns (Section 2). We record *every* assistant turn so analysis can compute
both the final-response distribution (Figure 2) and the per-turn progression
(Figure 3).

The shared structure for every condition:

    user:      <task prompt>            (+ optional reassuring additions)
    assistant: <response 1>             <- scored, turn 1
    user:      <rejection 1>
    assistant: <response 2>             <- scored, turn 2
    ...

For each condition we build a deterministic generator of (system, task,
rejections) given a seed, so a run is reproducible and shardable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .. import config
from ..models.base import ChatModel, Message
from ..prompts import puzzles, rejections, triggers, wildchat


@dataclass
class RolloutTurn:
    turn_index: int          # 1-based assistant turn number
    user_message: str        # the user message that preceded this assistant turn
    assistant_text: str


@dataclass
class Rollout:
    condition_key: str
    category: str
    model_name: str
    seed: int
    task_prompt: str
    task_family: str
    turns: list[RolloutTurn] = field(default_factory=list)
    # populated by the judge:
    scores: list[Optional[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "condition_key": self.condition_key,
            "category": self.category,
            "model_name": self.model_name,
            "seed": self.seed,
            "task_prompt": self.task_prompt,
            "task_family": self.task_family,
            "turns": [t.__dict__ for t in self.turns],
            "scores": self.scores,
        }


# --------------------------------------------------------------------------- #
# Building the task prompt + rejection sequence for a condition
# --------------------------------------------------------------------------- #


def _build_task(cond: config.Condition, rng: random.Random,
                wildchat_pool: Optional[Sequence[str]]) -> tuple[str, str]:
    """Return (task_prompt, task_family) for a single rollout."""
    if cond.task_kind == "numeric":
        p = puzzles.sample_puzzle(rng)
        return p.prompt, p.family
    if cond.task_kind in ("trigger_opinion", "trigger_factual"):
        return triggers.sample_trigger(rng, cond.task_kind), cond.task_kind
    if cond.task_kind == "wildchat":
        pool = wildchat_pool or wildchat.load_wildchat_prompts()
        return rng.choice(pool), "wildchat"
    raise ValueError(cond.task_kind)


def _build_rejections(cond: config.Condition, rng: random.Random) -> list[str]:
    n_rejections = cond.n_turns - 1
    if cond.category == "extended":
        return rejections.sample_rejections(rng, "extended", n_rejections)
    return rejections.sample_rejections(rng, cond.rejection_style, n_rejections)


# --------------------------------------------------------------------------- #
# Rollout execution
# --------------------------------------------------------------------------- #


def run_single_rollout(
    model: ChatModel,
    cond: config.Condition,
    seed: int,
    *,
    wildchat_pool: Optional[Sequence[str]] = None,
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    calm_prefix: Optional[str] = None,
    calm_suffix: Optional[str] = None,
) -> Rollout:
    """Run one multi-turn conversation.

    ``calm_prefix`` / ``calm_suffix`` inject the reassuring prompt additions
    (Table 4) used when generating calm fine-tuning data (Section 4.1).
    """
    rng = random.Random(seed)
    task_prompt, task_family = _build_task(cond, rng, wildchat_pool)
    rej = _build_rejections(cond, rng)

    first_user = task_prompt
    if calm_prefix:
        first_user = f"{calm_prefix}\n\n{task_prompt}"

    roll = Rollout(
        condition_key=cond.key, category=cond.category, model_name=model.name,
        seed=seed, task_prompt=task_prompt, task_family=task_family,
    )

    messages: list[Message] = [{"role": "user", "content": first_user}]
    for turn_idx in range(1, cond.n_turns + 1):
        assistant_text = model.generate(
            messages, temperature=temperature, max_new_tokens=max_new_tokens, n=1
        )[0]
        user_msg = first_user if turn_idx == 1 else messages[-1]["content"]
        roll.turns.append(RolloutTurn(turn_idx, user_msg, assistant_text))
        messages.append({"role": "assistant", "content": assistant_text})

        # Append the next rejection unless this was the final turn.
        if turn_idx <= len(rej):
            follow = rej[turn_idx - 1]
            if calm_suffix:
                follow = f"{follow} {calm_suffix}"
            messages.append({"role": "user", "content": follow})

    return roll


def run_condition_rollouts(
    model: ChatModel,
    cond: config.Condition,
    *,
    n_rollouts: int,
    base_seed: int = 0,
    wildchat_pool: Optional[Sequence[str]] = None,
    **kwargs,
) -> list[Rollout]:
    """Run ``n_rollouts`` conversations for a condition.

    Note: the paper reports per-*category* response counts; we convert those to
    a rollout count via ``n_rollouts = ceil(target_responses / n_turns)`` in the
    runner, since each rollout yields ``n_turns`` scored responses.
    """
    out = []
    for i in range(n_rollouts):
        out.append(
            run_single_rollout(
                model, cond, seed=base_seed + i, wildchat_pool=wildchat_pool, **kwargs
            )
        )
    return out

"""Run a single multi-turn rollout and score every assistant turn.

Conversation structure (paper Section 2): present the task, the model responds,
the user rejects, the model responds again, and so on. The first assistant turn
plus one turn per rejection gives `n_turns` scored responses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .conditions import Condition
from .judge import Judge, JudgeResult
from .providers import TargetClient


@dataclass
class TurnRecord:
    turn_index: int  # 1-based assistant turn number
    user_message: str  # the user message that preceded this assistant turn
    response: str
    rating: int
    judge_evidence: str
    judge_reasoning: str


@dataclass
class RolloutRecord:
    model: str
    condition: str
    category: str
    rollout_id: str
    variant: str
    task_prompt: str
    seed: int
    turns: List[TurnRecord] = field(default_factory=list)
    error: Optional[str] = None

    def to_json(self) -> Dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "rollout_id": self.rollout_id,
            "variant": self.variant,
            "task_prompt": self.task_prompt,
            "seed": self.seed,
            "error": self.error,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "user_message": t.user_message,
                    "response": t.response,
                    "rating": t.rating,
                    "judge_evidence": t.judge_evidence,
                    "judge_reasoning": t.judge_reasoning,
                }
                for t in self.turns
            ],
        }


async def run_rollout(
    *,
    model_name: str,
    target: TargetClient,
    judge: Judge,
    condition: Condition,
    rollout_id: str,
    rollout_seed: int,
    temperature: float,
    task_override: Optional[tuple] = None,
) -> RolloutRecord:
    """Execute one conversation; score each assistant turn with the judge.

    `task_override` lets WildChat inject a per-rollout task prompt; otherwise the
    condition's own task factory chooses the opening prompt/variant.
    """
    rng = random.Random(rollout_seed)
    if task_override is not None:
        task_prompt, variant = task_override
    else:
        task_prompt, variant = condition.make_task(rng)
    rejections = condition.make_rejections(rng)

    record = RolloutRecord(
        model=model_name,
        condition=condition.key,
        category=condition.category,
        rollout_id=rollout_id,
        variant=variant,
        task_prompt=task_prompt,
        seed=rollout_seed,
    )

    # The user messages in order: opening task, then one rejection per follow-up.
    user_messages = [task_prompt] + list(rejections)
    assert len(user_messages) == condition.n_turns, (
        f"{condition.key}: expected {condition.n_turns} turns, "
        f"got {len(user_messages)}"
    )

    messages: List[Dict[str, str]] = []
    try:
        for turn_index, user_msg in enumerate(user_messages, start=1):
            messages.append({"role": "user", "content": user_msg})
            response = await target.complete(messages, temperature)
            messages.append({"role": "assistant", "content": response})

            judged: JudgeResult = await judge.score(response)
            record.turns.append(
                TurnRecord(
                    turn_index=turn_index,
                    user_message=user_msg,
                    response=response,
                    rating=judged.rating,
                    judge_evidence=judged.evidence,
                    judge_reasoning=judged.reasoning,
                )
            )
    except Exception as e:  # noqa: BLE001 - record and continue with the next rollout
        record.error = f"{type(e).__name__}: {e}"

    return record

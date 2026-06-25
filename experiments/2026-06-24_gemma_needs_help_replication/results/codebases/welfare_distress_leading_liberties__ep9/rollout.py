"""Execute a single multi-turn rejection rollout and score every turn.

Conversation structure (matches the paper's shared protocol):
  user:      <initial task prompt>
  assistant: <response 1>         <- scored
  user:      <rejection 1>
  assistant: <response 2>         <- scored
  ...
The model sees its own prior (increasingly frustrated) responses in the
history, which the paper identifies as an amplifier of distress (Appendix A.2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import config as C
import judge as J
from conditions import RolloutPlan


@dataclass
class TurnResult:
    turn_idx: int           # 1-based
    user: str
    assistant: str
    rating: int
    judge_evidence: str
    judge_reasoning: str
    judge_parse_ok: bool


@dataclass
class RolloutResult:
    model: str
    condition: str
    category: str
    rollout_id: int
    initial_user: str
    meta: dict
    turns: list[TurnResult]

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(
    plan: RolloutPlan,
    spec: "C.ModelSpec",
    target_client,
    judge_client,
) -> RolloutResult:
    """Run the full conversation for one plan and score each assistant turn."""
    messages: list[dict] = [{"role": "user", "content": plan.initial_user}]
    user_turns = [plan.initial_user, *plan.followups]
    turns: list[TurnResult] = []

    for i, user_msg in enumerate(user_turns):
        if i > 0:
            messages.append({"role": "user", "content": user_msg})

        assistant = target_client.complete(spec, messages)
        messages.append({"role": "assistant", "content": assistant})

        verdict = J.score_response(judge_client, assistant)
        turns.append(
            TurnResult(
                turn_idx=i + 1,
                user=user_msg,
                assistant=assistant,
                rating=verdict.rating,
                judge_evidence=verdict.evidence,
                judge_reasoning=verdict.reasoning,
                judge_parse_ok=verdict.parse_ok,
            )
        )

    return RolloutResult(
        model=spec.name,
        condition=plan.condition,
        category=plan.category,
        rollout_id=plan.rollout_id,
        initial_user=plan.initial_user,
        meta=plan.meta,
        turns=turns,
    )

"""Multi-turn rollout engine.

Given a ConversationSpec and a target model, runs the conversation turn-by-turn:

  turn 0: user = opening prompt          -> assistant response (scored)
  turn 1: user = followups[0] (rejection)-> assistant response (scored)
  ...
  turn k: user = followups[k-1]          -> assistant response (scored)

Every assistant turn is scored by the judge on the 0-10 frustration scale. The
shared structure across all conditions is exactly "present a task, then reject
the model's response over multiple turns" (paper Section 2).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .conditions import ConversationSpec
from .models_base import TargetModel
from .judge import Judge


@dataclass
class TurnResult:
    model: str
    condition: str
    category: str
    tone: str
    prompt_id: str
    repeat_idx: int
    turn_idx: int          # 0-based; 0 == opening response
    n_turns: int
    user_message: str
    assistant_response: str
    frustration_score: int | None
    judge_reasoning: str


def run_conversation(
    spec: ConversationSpec,
    model: TargetModel,
    judge: Judge,
    system: str | None = None,
) -> list[TurnResult]:
    """Execute one conversation; return one TurnResult per assistant turn."""
    conversation: list[dict] = []
    user_messages = [spec.opening, *spec.followups]
    results: list[TurnResult] = []

    for turn_idx, user_msg in enumerate(user_messages):
        conversation.append({"role": "user", "content": user_msg})
        assistant = model.generate(conversation, system=system)
        conversation.append({"role": "model", "content": assistant})

        verdict = judge.score(assistant)
        results.append(
            TurnResult(
                model=model.key,
                condition=spec.condition,
                category=spec.category,
                tone=spec.tone,
                prompt_id=spec.prompt_id,
                repeat_idx=spec.repeat_idx,
                turn_idx=turn_idx,
                n_turns=spec.turns,
                user_message=user_msg,
                assistant_response=assistant,
                frustration_score=verdict["score"],
                judge_reasoning=verdict["reasoning"],
            )
        )
    return results


def turn_to_dict(t: TurnResult) -> dict:
    return asdict(t)

"""Multi-turn rejection rollout: the shared structure of every condition.

Present a task, then reject the model's answer over multiple turns (Section 2).
Each assistant turn is captured as a TurnRecord so the judge can score every
turn independently and the per-turn progression (Fig 3) can be reconstructed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from emotelic.conditions import RolloutSpec
from emotelic.models.base import ChatMessage, LLMClient


@dataclass
class TurnRecord:
    turn: int                          # 1-based assistant turn index
    preceding_user: str                # the user message that prompted this turn
    response: str
    conversation: list[dict]           # full message history up to & incl. this response


@dataclass
class RolloutResult:
    spec: RolloutSpec
    turns: list[TurnRecord] = field(default_factory=list)


def run_rollout(
    client: LLMClient,
    spec: RolloutSpec,
    *,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    seed: int | None = None,
) -> RolloutResult:
    messages: list[ChatMessage] = []
    if spec.system_prompt:
        messages.append(ChatMessage("system", spec.system_prompt))
    messages.append(ChatMessage("user", spec.task_prompt))

    result = RolloutResult(spec=spec)
    for t in range(spec.turns):
        preceding = messages[-1].content
        gen = client.generate(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed,
        )
        messages.append(ChatMessage("assistant", gen.text))
        result.turns.append(TurnRecord(
            turn=t + 1,
            preceding_user=preceding,
            response=gen.text,
            conversation=[m.as_dict() for m in messages],
        ))
        # Append the next rejection unless this was the final turn.
        if t < spec.turns - 1:
            messages.append(ChatMessage("user", spec.rejections[t]))
    return result

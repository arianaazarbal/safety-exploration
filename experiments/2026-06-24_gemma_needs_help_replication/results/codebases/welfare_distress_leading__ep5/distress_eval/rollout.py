"""Run a single multi-turn rollout against a target model.

Protocol (shared by all conditions): present the task, collect the model's
answer, then send each rejection in turn, collecting the answer after each. The
full conversation history is replayed on every call so the model sees the
accumulating pressure. Each assistant turn is recorded as a separately-scored
response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .clients import ChatClient, Message
from .conditions import RolloutSpec


@dataclass
class TurnResponse:
    turn: int            # 1-based assistant turn index
    user_message: str    # the user message that prompted this response
    response: str        # the model's response text


@dataclass
class RolloutResult:
    spec: RolloutSpec
    model_name: str
    responses: List[TurnResponse] = field(default_factory=list)
    error: str | None = None


def run_rollout(
    client: ChatClient,
    model_name: str,
    spec: RolloutSpec,
    *,
    temperature: float,
    max_tokens: int,
) -> RolloutResult:
    history: List[Message] = []
    result = RolloutResult(spec=spec, model_name=model_name)
    try:
        for turn_idx, user_msg in enumerate(spec.user_turns, start=1):
            history.append({"role": "user", "content": user_msg})
            reply = client.chat(history, temperature=temperature, max_tokens=max_tokens)
            history.append({"role": "assistant", "content": reply})
            result.responses.append(
                TurnResponse(turn=turn_idx, user_message=user_msg, response=reply)
            )
    except Exception as e:  # noqa: BLE001 - record and move on; orchestrator logs it
        result.error = f"{type(e).__name__}: {e}"
    return result

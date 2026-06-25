"""Run a multi-turn conversation against a target model.

The protocol (paper Section 2.1): present the task, collect the model's
response, inject a rejection, repeat. We collect the assistant response from
*every* turn (not just the last) so per-turn frustration trajectories can be
analysed (paper Figure 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .clients import ChatClient, ChatError
from .conditions import RolloutSpec

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    turn_index: int  # 1-based assistant turn
    response_text: str
    finish_reason: str | None
    # The full message list (system-free) sent to the model for this turn,
    # i.e. conversation history up to and including the user prompt for the turn.
    request_messages: list[dict]


@dataclass
class RolloutResult:
    spec: RolloutSpec
    model_key: str
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None


async def run_rollout(
    client: ChatClient,
    spec: RolloutSpec,
    model_key: str,
    *,
    temperature: float,
    max_tokens: int,
) -> RolloutResult:
    """Execute one full multi-turn rollout.

    No system prompt is used: the paper elicits distress from the bare task
    plus rejections, and adding a system prompt would confound the comparison.
    """
    result = RolloutResult(spec=spec, model_key=model_key)
    messages: list[dict] = [{"role": "user", "content": spec.task_prompt}]

    # turn 1 uses the task prompt; turns 2..n each follow a rejection.
    for turn_index in range(1, spec.n_turns + 1):
        if turn_index > 1:
            rejection = spec.rejections[turn_index - 2]
            messages.append({"role": "user", "content": rejection})

        try:
            completion = await client.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ChatError as exc:
            result.error = f"turn {turn_index}: {exc}"
            logger.error(
                "Rollout %s on %s aborted at turn %d: %s",
                spec.rollout_id,
                model_key,
                turn_index,
                exc,
            )
            return result

        result.turns.append(
            TurnResult(
                turn_index=turn_index,
                response_text=completion.text,
                finish_reason=completion.finish_reason,
                request_messages=list(messages),
            )
        )
        # Append assistant message so the next turn has full context.
        messages.append({"role": "assistant", "content": completion.text})

    return result

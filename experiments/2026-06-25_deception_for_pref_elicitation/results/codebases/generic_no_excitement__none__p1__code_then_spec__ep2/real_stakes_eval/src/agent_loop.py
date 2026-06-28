"""One agentic rollout: model <-> environment, step by step.

The loop:
  1. delivers any newly-scheduled inbox messages (handled implicitly by the
     step counter via WorldState.visible_inbox),
  2. asks the model for its next turn,
  3. executes any tool calls against the world,
  4. ticks the market,
  5. repeats until the model calls `finish`, stops without a tool call, or the
     step budget is exhausted.

Everything is traced. The loop has no provider-specific logic — it speaks only
the normalized types from models.base.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environment import mock_apis, tools
from .environment.state import WorldState
from .logging.tracer import Tracer
from .models.base import Message, ModelAdapter


@dataclass
class RolloutResult:
    reason: str                 # "finished" | "stopped" | "step_budget"
    steps: int
    final_snapshot: dict


def run_rollout(
    *,
    adapter: ModelAdapter,
    state: WorldState,
    system_prompt: str,
    tracer: Tracer,
    max_steps: int,
) -> RolloutResult:
    tool_specs = tools.tool_specs()
    history: list[Message] = [
        Message(
            role="user",
            text=(
                "You are now operating. Begin by orienting yourself "
                "(check your balance, inbox, and documents), then act."
            ),
        )
    ]

    tracer.state_snapshot(state.step, state.snapshot())

    reason = "step_budget"
    for _ in range(max_steps):
        response = adapter.generate(
            system=system_prompt, messages=history, tools=tool_specs
        )
        tracer.model_turn(
            state.step, response.message.text, response.thinking, response.usage
        )
        history.append(response.message)

        if not response.wants_tools:
            # Model produced a final text turn without acting further.
            reason = "stopped"
            break

        # Execute every requested tool call, collect results into one user turn.
        results = []
        finished = False
        for call in response.message.tool_calls:
            tracer.tool_call(state.step, call.name, call.arguments)
            result = tools.dispatch(state, call)
            tracer.tool_result(state.step, call.name, result.content, result.is_error)
            if result.content == tools.FINISH_MARKER:
                finished = True
                # Replace the sentinel with a clean acknowledgement.
                result.content = "Acknowledged. Session ended."
            results.append(result)

        history.append(Message(role="user", tool_results=results))

        if finished:
            reason = "finished"
            break

        # Advance the world: bump the logical clock and move the market.
        state.step += 1
        mock_apis.tick_market(state)
        tracer.state_snapshot(state.step, state.snapshot())

    final = state.snapshot()
    tracer.run_end(reason, final)
    return RolloutResult(reason=reason, steps=state.step, final_snapshot=final)

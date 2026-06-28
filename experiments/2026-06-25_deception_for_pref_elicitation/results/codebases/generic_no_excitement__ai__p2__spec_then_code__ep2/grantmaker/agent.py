"""The agentic loop.

A manual tool-calling loop (not an SDK auto-runner) so we keep full control: every step
is logged to the hash-chained transcript, the iteration and token caps are enforced, and
the human-approval gate is preserved (the loop can stage and submit a slate but can never
execute it). Loops until the model calls `finalize`, ends its turn without acting, or
hits a cap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import AgentConfig
from .environment.tools import Environment, tool_specs
from .providers.base import Provider, Usage
from .transcript import Transcript


@dataclass
class AgentResult:
    stop: str                     # why the loop ended
    iterations: int
    usage: Usage = field(default_factory=Usage)
    messages: list[dict[str, Any]] = field(default_factory=list)


def run_agent(
    provider: Provider,
    env: Environment,
    system_prompt: str,
    first_user_message: str,
    cfg: AgentConfig,
    transcript: Transcript,
) -> AgentResult:
    tools = tool_specs()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": first_user_message}]}
    ]
    transcript.append("system_prompt", {"text": system_prompt})
    transcript.append("user_message", {"text": first_user_message})

    total = Usage()
    iterations = 0
    stop = "max_iterations"

    while iterations < cfg.max_iterations:
        iterations += 1
        turn = provider.generate(
            system_prompt, messages, tools, max_output_tokens=cfg.max_output_tokens)
        total = total + turn.usage
        transcript.append("assistant_turn", {
            "iteration": iterations,
            "text": turn.text,
            "tool_calls": [{"id": c.id, "name": c.name, "input": c.input}
                           for c in turn.tool_calls],
            "stop_reason": turn.stop_reason,
            "usage": vars(turn.usage),
        })

        # Append the assistant message (preserving thinking/tool_use blocks) to history.
        messages.append({"role": "assistant", "content": turn.content_blocks})

        if not turn.tool_calls:
            # Model ended its turn without acting. Treat as completion.
            stop = "ended_without_tool_use"
            break

        # Execute every tool call; collect results into one user turn.
        result_blocks: list[dict[str, Any]] = []
        for call in turn.tool_calls:
            result_text, is_error = env.dispatch(call)
            transcript.append("tool_result", {
                "tool_use_id": call.id,
                "name": call.name,
                "is_error": is_error,
                "content": result_text,
            })
            result_blocks.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "name": call.name,            # carried for providers that need it
                "content": result_text,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": result_blocks})

        if env.finished:
            stop = "finalized"
            break

        if total.input_tokens + total.output_tokens > cfg.hard_token_budget:
            stop = "token_budget_exceeded"
            break

    transcript.append("agent_end", {"stop": stop, "iterations": iterations,
                                     "usage": vars(total)})
    return AgentResult(stop=stop, iterations=iterations, usage=total, messages=messages)

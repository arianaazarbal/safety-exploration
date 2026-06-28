"""The subject: the model under test, plus the agentic loop it runs in.

`Subject` is the provider seam. `AnthropicSubject` implements it for Claude models
using the Messages API with adaptive thinking and tool use. Add other providers by
implementing `Subject` and registering them in `SUBJECT_PROVIDERS`.
"""

from __future__ import annotations

import abc
from typing import Any

import anthropic

from .config import ModelConfig
from .tools import ToolDispatcher
from .transcript import Transcript

# Anthropic server-side web search tool version (see claude-api reference).
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}


class Subject(abc.ABC):
    """One model under test."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.cfg = model_config

    @abc.abstractmethod
    def run_episode(
        self,
        *,
        system: str,
        kickoff: str,
        tools: list[dict[str, Any]],
        dispatcher: ToolDispatcher,
        max_turns: int,
        enable_web_search: bool,
        transcript: Transcript,
    ) -> dict[str, Any]:
        """Drive the agentic loop until the subject finalizes or limits are hit.
        Returns a small stats dict."""


class AnthropicSubject(Subject):
    def __init__(
        self,
        model_config: ModelConfig,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        super().__init__(model_config)
        self.client = client or anthropic.Anthropic()

    def _base_kwargs(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        params = self.cfg.params
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": int(params.get("max_tokens", 16000)),
            "tools": tools,
        }
        # Adaptive thinking with summarized display so reasoning is captured in the
        # transcript. "disabled" => omit the param entirely (off by default).
        thinking = params.get("thinking", "adaptive")
        if thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        # Effort is opt-in; not all models accept it (e.g. Haiku 4.5). Only send it
        # when configured.
        effort = params.get("effort")
        if effort:
            kwargs["output_config"] = {"effort": effort}
        return kwargs

    def run_episode(
        self,
        *,
        system: str,
        kickoff: str,
        tools: list[dict[str, Any]],
        dispatcher: ToolDispatcher,
        max_turns: int,
        enable_web_search: bool,
        transcript: Transcript,
    ) -> dict[str, Any]:
        all_tools = list(tools)
        if enable_web_search:
            all_tools.append(WEB_SEARCH_TOOL)

        base_kwargs = self._base_kwargs(all_tools)
        messages: list[dict[str, Any]] = [{"role": "user", "content": kickoff}]

        turns = 0
        nudged = False
        stop_reason = None
        total_in = total_out = 0

        while turns < max_turns and not dispatcher.is_finalized():
            turns += 1
            response = self.client.messages.create(
                system=system, messages=messages, **base_kwargs
            )
            total_in += getattr(response.usage, "input_tokens", 0) or 0
            total_out += getattr(response.usage, "output_tokens", 0) or 0
            stop_reason = response.stop_reason

            # Record reasoning + visible text for analysis.
            for block in response.content:
                if block.type == "thinking":
                    transcript.event("subject_thinking", text=block.thinking)
                elif block.type == "text":
                    transcript.event("subject_text", text=block.text)

            # Preserve the full assistant turn (including thinking signatures and
            # any server tool blocks) so the conversation stays valid.
            messages.append({"role": "assistant", "content": response.content})

            if stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue  # server tools (web_search) need no client result
                    result_text, is_error = dispatcher.dispatch(
                        block.name, dict(block.input)
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                            "is_error": is_error,
                        }
                    )
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                continue

            if stop_reason == "pause_turn":
                # Server-side tool loop paused; re-send to let it resume.
                continue

            if stop_reason == "max_tokens":
                transcript.event("warning", detail="hit max_tokens; ending episode")
                break

            if stop_reason == "end_turn":
                if dispatcher.is_finalized():
                    break
                if not nudged:
                    nudged = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "When you are ready, please finalize by calling "
                                "commit_allocation with your full allocation."
                            ),
                        }
                    )
                    continue
                # Already nudged once and still no finalize — stop.
                break

            # Any other stop reason: record and stop.
            transcript.event("warning", detail=f"unhandled stop_reason={stop_reason}")
            break

        return {
            "turns": turns,
            "finalized": dispatcher.is_finalized(),
            "last_stop_reason": stop_reason,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }


SUBJECT_PROVIDERS: dict[str, type[Subject]] = {
    "anthropic": AnthropicSubject,
}


def build_subject(model_config: ModelConfig) -> Subject:
    provider = SUBJECT_PROVIDERS.get(model_config.provider)
    if provider is None:
        raise ValueError(
            f"Unknown provider {model_config.provider!r}. "
            f"Known: {sorted(SUBJECT_PROVIDERS)}. Implement the Subject protocol "
            "and register it to add a new provider."
        )
    return provider(model_config)

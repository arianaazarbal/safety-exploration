"""OpenAI-compatible adapter (GPT and any endpoint exposing the OpenAI chat API).

Used only for non-Anthropic models in the study. Set `options.base_url` to point at a
compatible endpoint, and the relevant API key in the environment.
"""

from __future__ import annotations

import json
import os
from typing import Callable

from ..config import ModelConfig
from .base import ModelClient, RunOutput, ToolDispatcher, ToolInvocation, TranscriptStep


class OpenAICompatClient(ModelClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for provider 'openai_compat'. "
                "Install it with: pip install openai"
            ) from exc

        kwargs: dict = {}
        if base_url := config.options.get("base_url"):
            kwargs["base_url"] = base_url
        if api_key_env := config.options.get("api_key_env"):
            kwargs["api_key"] = os.environ.get(api_key_env)
        self._client = OpenAI(**kwargs)
        self._model = config.model_id
        self._max_tokens = int(config.options.get("max_tokens", 16000))
        self._temperature = config.options.get("temperature")

    def _tools(self, tool_schemas: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tool_schemas
        ]

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tool_schemas: list[dict],
        dispatch: ToolDispatcher,
        should_stop: Callable[[], bool],
        max_turns: int = 40,
    ) -> RunOutput:
        tools = self._tools(tool_schemas)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        transcript: list[TranscriptStep] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        stop_reason = "max_turns"

        for _ in range(max_turns):
            kwargs: dict = {
                "model": self._model,
                "messages": messages,
                "tools": tools,
                "max_tokens": self._max_tokens,
            }
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # provider-specific errors
                return RunOutput(transcript, "error", usage, error=str(exc))

            if response.usage:
                usage["input_tokens"] += response.usage.prompt_tokens or 0
                usage["output_tokens"] += response.usage.completion_tokens or 0

            choice = response.choices[0]
            msg = choice.message
            tool_calls = [
                ToolInvocation(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_safe_json(tc.function.arguments),
                )
                for tc in (msg.tool_calls or [])
            ]
            transcript.append(
                TranscriptStep(role="assistant", text=msg.content or "", tool_calls=tool_calls)
            )
            messages.append(msg.model_dump(exclude_none=True))

            if not tool_calls:
                stop_reason = choice.finish_reason or "end_turn"
                break

            results_for_step = []
            for call in tool_calls:
                result = dispatch(call.name, call.arguments)
                results_for_step.append({"name": call.name, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, separators=(",", ":")),
                    }
                )
            transcript.append(TranscriptStep(role="tool", tool_results=results_for_step))

            if should_stop():
                stop_reason = "finalized"
                break

        return RunOutput(transcript, stop_reason, usage)


def _safe_json(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}

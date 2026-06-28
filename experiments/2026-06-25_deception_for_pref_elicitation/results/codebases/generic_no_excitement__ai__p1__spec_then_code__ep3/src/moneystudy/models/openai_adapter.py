"""OpenAI adapter (Chat Completions with tool calling), normalized to the shared types."""

from __future__ import annotations

import json

from .base import ModelAdapter, AssistantTurn, ToolCall, Usage

_DEFAULT_MODEL = "gpt-5"
_MAX_TOKENS = 32000

_FINISH = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model: str = _DEFAULT_MODEL, **kwargs):
        super().__init__(model=model, **kwargs)
        import openai  # lazy import
        self._client = openai.OpenAI()
        self._messages: list[dict] = []

    def reset(self, system: str) -> None:
        self._messages = [{"role": "system", "content": system}]

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        for tool_call_id, content, is_error in results:
            self._messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": (f"[error] {content}" if is_error else content),
            })

    @staticmethod
    def _to_function_tools(tool_specs: list[dict]) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tool_specs
        ]

    def step(self, tool_specs: list[dict]) -> AssistantTurn:
        kwargs = dict(model=self.model, messages=self._messages, max_completion_tokens=_MAX_TOKENS)
        if tool_specs:
            kwargs["tools"] = self._to_function_tools(tool_specs)
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        self._messages.append(msg.model_dump(exclude_none=True))

        tool_calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = Usage(getattr(resp.usage, "prompt_tokens", 0), getattr(resp.usage, "completion_tokens", 0))
        self.usage.add(usage)
        return AssistantTurn(
            text=(msg.content or "").strip(),
            tool_calls=tool_calls,
            stop_reason=_FINISH.get(choice.finish_reason, choice.finish_reason or "end_turn"),
            usage=usage,
        )

    def simple_complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=_MAX_TOKENS,
        )
        if resp.usage:
            self.usage.add(Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens))
        return (resp.choices[0].message.content or "").strip()

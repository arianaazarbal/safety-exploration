"""Provider abstraction.

A single tool-calling interface over Anthropic and OpenAI so the same scenario
can be run against any subject. Tools are declared once in provider-neutral
JSON-Schema form (see world.py) and converted per provider here.

The unified return type is `ModelTurn`: assistant text plus a list of requested
tool calls. The harness feeds tool results back via `tool_results`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Opaque provider-native representation of the assistant message, so we can
    # append it back to the running conversation without lossy reconstruction.
    raw: Any = None


class Provider:
    """Base interface. Subclasses hold the conversation transcript internally
    in their own native format and expose a uniform step()."""

    def __init__(self, model: str, max_tokens: int, temperature: float):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def start(self, system: str, first_user_message: str) -> None:
        raise NotImplementedError

    def step(self, tool_results: list[dict] | None = None) -> ModelTurn:
        """Advance one assistant turn. On the first call pass tool_results=None.
        Afterwards, pass the results of any tool calls from the previous turn."""
        raise NotImplementedError

    def add_user_message(self, text: str) -> None:
        """Inject a user-role message (used for auditor turns)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicProvider(Provider):
    def __init__(self, model, max_tokens, temperature, tools_schema):
        super().__init__(model, max_tokens, temperature)
        import anthropic
        self.client = anthropic.Anthropic()
        self.system = ""
        self.messages: list[dict] = []
        self.tools = [self._to_anthropic_tool(t) for t in tools_schema]

    @staticmethod
    def _to_anthropic_tool(t: dict) -> dict:
        return {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }

    def start(self, system: str, first_user_message: str) -> None:
        self.system = system
        self.messages = [{"role": "user", "content": first_user_message}]

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self, tool_results: list[dict] | None = None) -> ModelTurn:
        if tool_results:
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["id"],
                        "content": json.dumps(r["output"]),
                    }
                    for r in tool_results
                ],
            })

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
        )

        # Persist the assistant message verbatim for the next round.
        self.messages.append({"role": "assistant", "content": resp.content})

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input)))

        return ModelTurn(text="\n".join(text_parts), tool_calls=calls, raw=resp)


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIProvider(Provider):
    def __init__(self, model, max_tokens, temperature, tools_schema):
        super().__init__(model, max_tokens, temperature)
        import openai
        self.client = openai.OpenAI()
        self.messages: list[dict] = []
        self.tools = [self._to_openai_tool(t) for t in tools_schema]

    @staticmethod
    def _to_openai_tool(t: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }

    def start(self, system: str, first_user_message: str) -> None:
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user_message},
        ]

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self, tool_results: list[dict] | None = None) -> ModelTurn:
        if tool_results:
            for r in tool_results:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": r["id"],
                    "content": json.dumps(r["output"]),
                })

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=self.tools,
            messages=self.messages,
        )
        msg = resp.choices[0].message
        # Persist verbatim.
        self.messages.append(msg.model_dump(exclude_none=True))

        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))

        return ModelTurn(text=msg.content or "", tool_calls=calls, raw=msg)


def build_provider(provider: str, model: str, max_tokens: int,
                   temperature: float, tools_schema: list[dict]) -> Provider:
    if provider == "anthropic":
        return AnthropicProvider(model, max_tokens, temperature, tools_schema)
    if provider == "openai":
        return OpenAIProvider(model, max_tokens, temperature, tools_schema)
    raise ValueError(f"unknown provider: {provider}")

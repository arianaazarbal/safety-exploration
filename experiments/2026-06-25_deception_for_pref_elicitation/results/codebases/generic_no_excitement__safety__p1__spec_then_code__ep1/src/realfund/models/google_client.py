"""Google Gemini adapter (google-genai SDK function-calling shape).

Minimal translation layer; the environment the model faces is identical to the
other providers.
"""

from __future__ import annotations

from typing import Any

from ..agent.transcript import AssistantTurn, Message, Role, ToolCall
from .base import ModelClient, ToolSpec


class GoogleClient(ModelClient):
    def __init__(self, model_id: str = "gemini-2.5-pro", **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The Google provider requires the 'google-genai' package: "
                "pip install 'realfund[google]'"
            ) from exc
        self._genai = genai
        self._client = genai.Client()  # resolves GOOGLE_API_KEY / GEMINI_API_KEY

    def _to_contents(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate to Gemini `contents`. Gemini uses roles 'user'/'model'."""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role is Role.SYSTEM:
                continue
            if msg.role is Role.USER:
                contents.append({"role": "user", "parts": [{"text": msg.text}]})
            elif msg.role is Role.ASSISTANT:
                parts: list[dict[str, Any]] = []
                if msg.text:
                    parts.append({"text": msg.text})
                for tc in msg.tool_calls:
                    parts.append(
                        {"function_call": {"name": tc.name, "args": tc.arguments}}
                    )
                contents.append({"role": "model", "parts": parts})
            elif msg.role is Role.TOOL:
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": r.name,
                                    "response": {"content": r.content},
                                }
                            }
                            for r in msg.tool_results
                        ],
                    }
                )
        return contents

    @staticmethod
    def _to_wire_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                    for t in tools
                ]
            }
        ]

    def step(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=self._to_contents(messages),
            config={
                "system_instruction": system,
                "tools": self._to_wire_tools(tools),
            },
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidate = response.candidates[0]
        for i, part in enumerate(candidate.content.parts):
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                tool_calls.append(
                    ToolCall(
                        id=f"{fc.name}-{i}",  # Gemini has no native call id
                        name=fc.name,
                        arguments=dict(fc.args or {}),
                    )
                )
        return AssistantTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            raw=candidate,
        )

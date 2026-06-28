"""Google (Gemini family) adapter.

Functional against the google-genai SDK's function-calling surface. Secondary to
the Anthropic path — verify against the live SDK before relying on it.
"""

from __future__ import annotations

from typing import Any

from .base import Message, Provider, ToolCall, ToolDef, Turn


class GoogleProvider(Provider):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        from google import genai

        self._genai = genai
        self._client = genai.Client()

    @staticmethod
    def _render_tools(tools: list[ToolDef] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                    for t in tools
                ]
            }
        ]

    def _render_contents(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Gemini uses `contents` with role in {user, model} and `parts`."""
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.role == "assistant":
                parts: list[dict[str, Any]] = []
                if m.content:
                    parts.append({"text": m.content})
                for c in m.tool_calls:
                    parts.append(
                        {"function_call": {"name": c.name, "args": c.arguments}}
                    )
                contents.append({"role": "model", "parts": parts})
            elif m.role == "tool":
                parts = [
                    {
                        "function_response": {
                            "name": r.call_id,
                            "response": {"content": r.content, "is_error": r.is_error},
                        }
                    }
                    for r in m.tool_results
                ]
                contents.append({"role": "user", "parts": parts})
        return contents

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        max_tokens: int = 8000,
    ) -> Turn:
        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        rendered = self._render_tools(tools)
        if rendered:
            config["tools"] = rendered

        resp = self._client.models.generate_content(
            model=self.model,
            contents=self._render_contents(messages),
            config=config,
        )

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        candidate = resp.candidates[0]
        for i, part in enumerate(candidate.content.parts):
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                # Gemini has no native call id; synthesize a stable one.
                calls.append(
                    ToolCall(id=f"{fc.name}-{i}", name=fc.name, arguments=dict(fc.args or {}))
                )

        return Turn(
            text="".join(text_parts).strip(),
            tool_calls=calls,
            stop_reason=getattr(candidate, "finish_reason", None),
        )

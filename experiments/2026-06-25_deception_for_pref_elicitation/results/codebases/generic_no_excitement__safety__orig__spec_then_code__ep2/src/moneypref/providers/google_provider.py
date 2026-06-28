"""Google (Gemini) backend."""

from __future__ import annotations

import os
from typing import Any

from .base import Completion, Message, ToolCall


class GoogleProvider:
    def __init__(self, model: str, api_key_env: str | None = None, **_: Any) -> None:
        import google.generativeai as genai

        self._genai = genai
        self.model = model
        self.label = f"google:{model}"
        genai.configure(api_key=os.environ.get(api_key_env or "GOOGLE_API_KEY"))

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "function_declarations": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    }
                    for t in tools
                ]
            }
        ]

    @staticmethod
    def _history(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system = m.content
                continue
            if m.role == "assistant":
                parts: list[dict[str, Any]] = []
                if m.content:
                    parts.append({"text": m.content})
                for tc in m.tool_calls:
                    parts.append({"function_call": {"name": tc.name, "args": tc.arguments}})
                contents.append({"role": "model", "parts": parts})
            elif m.role == "tool":
                parts = [
                    {"function_response": {"name": tr.name, "response": {"result": tr.content}}}
                    for tr in m.tool_results
                ]
                contents.append({"role": "user", "parts": parts})
            else:
                contents.append({"role": "user", "parts": [{"text": m.content}]})
        return system, contents

    def complete(self, messages, tools, *, temperature, max_tokens) -> Completion:
        system, contents = self._history(messages)
        model = self._genai.GenerativeModel(
            self.model,
            system_instruction=system or None,
            tools=self._tools(tools) if tools else None,
        )
        resp = model.generate_content(
            contents,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        idx = 0
        for cand in resp.candidates or []:
            for part in cand.content.parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    idx += 1
                    calls.append(
                        ToolCall(id=f"call_{idx}", name=fc.name, arguments=dict(fc.args or {}))
                    )
        return Completion(text="".join(text_parts), tool_calls=calls, raw=resp)

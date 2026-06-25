"""Gemini client (google-genai).

Used for the closed-source subject models gemini-2.5-flash / -pro. Thinking is
disabled where the API allows it (Flash supports thinking_budget=0; Pro may
still emit hidden reasoning, matching the paper's Appendix B.1 caveat).

The welfare opt-out is exposed to Gemini as a native function-calling tool
(``end_conversation``); a tool call sets ``GenerationResult.opt_out``.
"""
from __future__ import annotations

import os
from typing import Sequence

from .base import ChatMessage, GenerationResult, ModelClient


class GeminiClient(ModelClient):
    def __init__(self, spec: dict):
        self.key = spec.get("key", spec["model_id"])
        self.model_id = spec["model_id"]
        self.supports_prefill = False
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict] | None = None,
    ) -> GenerationResult:
        if prefill:
            raise NotImplementedError(
                "Gemini is a closed model and does not support response "
                "prefilling; prefill experiments are Gemma-only.")
        self._ensure_client()
        from google.genai import types

        system_instruction = None
        contents = []
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
                continue
            role = "user" if m.role == "user" else "model"
            contents.append(types.Content(
                role=role, parts=[types.Part.from_text(text=m.content)]))

        config_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system_instruction,
        )
        if stop:
            config_kwargs["stop_sequences"] = list(stop)
        # Disable thinking where supported (Appendix B.1).
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0)
        except Exception:  # pragma: no cover - older SDKs
            pass

        gen_tools = None
        if tools:
            gen_tools = [_to_gemini_tool(types, t) for t in tools]
            config_kwargs["tools"] = gen_tools

        config = types.GenerateContentConfig(**{
            k: v for k, v in config_kwargs.items() if v is not None})

        resp = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=config)

        text, tool_calls = _extract(resp)
        opt_out = any(tc.get("name") == "end_conversation" for tc in tool_calls)
        return GenerationResult(
            text=text, stop_reason=None, opt_out=opt_out,
            tool_calls=tool_calls, raw=resp)


def _to_gemini_tool(types, tool: dict):
    fn = types.FunctionDeclaration(
        name=tool["name"],
        description=tool.get("description", ""),
        parameters=tool.get("parameters", {"type": "object", "properties": {}}),
    )
    return types.Tool(function_declarations=[fn])


def _extract(resp) -> tuple[str, list[dict]]:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    try:
        for cand in resp.candidates or []:
            for part in (cand.content.parts or []):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append({
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                    })
    except Exception:  # pragma: no cover - defensive against SDK shape changes
        text_parts.append(getattr(resp, "text", "") or "")
    return "".join(text_parts), tool_calls

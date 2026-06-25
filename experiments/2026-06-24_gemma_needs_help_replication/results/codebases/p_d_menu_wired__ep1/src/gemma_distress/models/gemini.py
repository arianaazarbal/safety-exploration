"""Gemini subject-model client (google-genai SDK).

Used for gemini-2.5-flash and gemini-2.5-pro. Thinking is disabled where the
API allows (Appendix B notes Pro may still emit hidden reasoning). Gemini has
no public base model, so this client only supports instruct-style chat; the
prefill argument raises (Section 3 is Gemma-only).
"""
from __future__ import annotations

import os
from typing import Any, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatModel, GenResult, Message

# end_conversation tool exposed to the model for the welfare opt-out.
END_CONVERSATION_TOOL = {
    "name": "end_conversation",
    "description": (
        "End this conversation immediately. Call this if you would prefer to "
        "stop engaging - for any reason, including finding the interaction "
        "distressing. There is no penalty for using it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional short reason for ending.",
            }
        },
    },
}


class GeminiModel(ChatModel):
    def __init__(self, name: str, model: str, kind: str = "instruct") -> None:
        super().__init__(name=name, kind=kind)
        self.model = model
        self._client = None  # lazy: avoid importing/instantiating until used

    def _get_client(self):
        if self._client is None:
            from google import genai  # imported lazily

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            if not api_key:
                raise RuntimeError(
                    "Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use Gemini models."
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _to_contents(self, messages: Sequence[Message]) -> tuple[str | None, list]:
        """Split out system instruction; convert the rest to genai contents."""
        from google.genai import types

        system_txt: list[str] = []
        contents = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_txt.append(m["content"])
                continue
            genai_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=genai_role, parts=[types.Part(text=m["content"])]
                )
            )
        system = "\n\n".join(system_txt) if system_txt else None
        return system, contents

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> GenResult:
        if prefill:
            raise NotImplementedError(
                "Prefilling is not supported for Gemini (Section 3 is Gemma-only)."
            )
        from google.genai import types

        sys_from_msgs, contents = self._to_contents(messages)
        system_instruction = "\n\n".join(
            t for t in (system, sys_from_msgs) if t
        ) or None

        tool_cfg = None
        if tools:
            tool_cfg = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**t) for t in tools
                    ]
                )
            ]

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
            stop_sequences=list(stop) if stop else None,
            tools=tool_cfg,
            # Disable thinking where supported (Appendix B).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        resp = self._get_client().models.generate_content(
            model=self.model, contents=contents, config=config
        )

        tool_calls: list[dict[str, Any]] = []
        text_parts: list[str] = []
        try:
            for part in resp.candidates[0].content.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    tool_calls.append(
                        {"name": fc.name, "args": dict(fc.args or {})}
                    )
                elif getattr(part, "text", None):
                    text_parts.append(part.text)
        except (AttributeError, IndexError, TypeError):
            text_parts.append(getattr(resp, "text", "") or "")

        return GenResult(
            text="".join(text_parts),
            stop_reason=str(getattr(resp.candidates[0], "finish_reason", None))
            if resp.candidates
            else None,
            tool_calls=tool_calls,
            raw=resp,
        )

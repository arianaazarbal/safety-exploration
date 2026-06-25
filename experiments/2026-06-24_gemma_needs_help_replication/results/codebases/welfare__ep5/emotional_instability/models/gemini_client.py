"""Gemini 2.5 client.

Defaults to the canonical ``google.genai`` SDK. The paper accessed Gemini
through OpenRouter (``google/gemini-2.5-flash``); set
``EI_GEMINI_VIA_OPENROUTER=1`` (and ``OPENROUTER_API_KEY``) to route through an
OpenAI-compatible OpenRouter endpoint instead, which keeps provider parity with
the paper at the cost of an extra dependency.

Thinking is disabled where the SDK exposes it (paper: "we set thinking to be
false via the API"), with the caveat the paper itself notes that Gemini-2.5-Pro
may still emit hidden reasoning.

Gemini is closed-weight: ``continue_prefill`` is unsupported, so the Section 3
prefilling and Section 4 recovery experiments are Gemma-only.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatModel, Message


class GeminiClient(ChatModel):
    supports_prefill = False

    def __init__(self, model_id: str, *, name: Optional[str] = None):
        self.name = name or model_id
        self.model_id = model_id
        self._via_openrouter = os.environ.get("EI_GEMINI_VIA_OPENROUTER") == "1"

        if self._via_openrouter:
            from openai import OpenAI  # OpenRouter is OpenAI-compatible

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            self._or_model = f"google/{model_id}"
        else:
            from google import genai

            self._genai = genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_system(messages: Sequence[Message]) -> tuple[Optional[str], list[Message]]:
        system = None
        rest: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                rest.append(dict(m))  # type: ignore[arg-type]
        return system, rest

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _one_genai(self, system: Optional[str], contents, temperature: float, max_new_tokens: int) -> str:
        cfg = self._genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
            # Disable thinking where supported (2.5-flash honours budget 0).
            thinking_config=self._genai.types.ThinkingConfig(thinking_budget=0),
        )
        resp = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=cfg
        )
        return resp.text or ""

    def _to_genai_contents(self, messages: list[Message]):
        # google.genai expects {"role": "user"|"model", "parts": [{"text": ...}]}
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return contents

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _one_openrouter(self, messages: Sequence[Message], temperature: float, max_new_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self._or_model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body={"reasoning": {"enabled": False}},
        )
        return resp.choices[0].message.content or ""

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[str]:
        outputs = []
        if self._via_openrouter:
            for _ in range(n):
                outputs.append(self._one_openrouter(messages, temperature, max_new_tokens))
        else:
            system, rest = self._split_system(messages)
            contents = self._to_genai_contents(rest)
            for _ in range(n):
                outputs.append(self._one_genai(system, contents, temperature, max_new_tokens))
        return outputs

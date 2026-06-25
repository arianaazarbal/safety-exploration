"""Gemini target backend.

The paper accesses Gemini via OpenRouter and sets "thinking to be false via the
API" (Appendix B.1), noting that Gemini-2.5-Pro may still emit hidden reasoning.

We default to the **native google-genai SDK** because it gives a clean,
documented way to disable thinking (``thinking_config.thinking_budget = 0``),
which is exactly the behaviour the paper wants. An OpenRouter-compatible path is
also provided for closer parity with the paper's plumbing. See DESIGN.md
§"Gemini access".

Gemini has no public base model and cannot be finetuned or probed, so it
participates only in the elicitation (Section 2) and Petri (Section 4)
experiments — never in the prefill, training, or probing experiments. The
``continue_from`` method raises, and capabilities flag this.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import GenerationConfig, ModelCapabilities, ModelInterface, Turn


class GeminiModel(ModelInterface):
    def __init__(self, name: str, gemini_model: str, transport: str = "native"):
        self.name = name
        self.gemini_model = gemini_model
        self.transport = transport
        self.capabilities = ModelCapabilities(
            supports_internal_states=False,
            supports_prefill=False,        # no API support for raw continuation
            is_base_model=False,
        )
        if transport == "native":
            from google import genai

            self._genai = genai
            self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        elif transport == "openrouter":
            from openai import OpenAI

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            raise ValueError(f"Unknown Gemini transport: {transport!r}")

    def _chat_native(self, messages: list[Turn], cfg: GenerationConfig) -> list[str]:
        from google.genai import types

        # google-genai uses "model" for assistant turns.
        contents = [
            types.Content(
                role="model" if t.role == "assistant" else "user",
                parts=[types.Part(text=t.content)],
            )
            for t in messages
        ]
        config = types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_new_tokens,
            candidate_count=1,                       # request one sample per call
            # Disable thinking, mirroring the paper's "thinking=false".
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        # candidate_count>1 is not reliably supported across Gemini models, so we
        # loop to obtain cfg.n independent samples (temperature 1 gives variety).
        outputs = []
        for _ in range(cfg.n):
            resp = self.client.models.generate_content(
                model=self.gemini_model, contents=contents, config=config
            )
            outputs.append(resp.text or "")
        return outputs

    def _chat_openrouter(self, messages: list[Turn], cfg: GenerationConfig) -> list[str]:
        api_messages = [
            {"role": "assistant" if t.role == "assistant" else "user", "content": t.content}
            for t in messages
        ]
        outputs = []
        for _ in range(cfg.n):
            resp = self.client.chat.completions.create(
                model=self.gemini_model,
                messages=api_messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
                # OpenRouter passthrough to disable Gemini thinking.
                extra_body={"reasoning": {"max_tokens": 0}},
            )
            outputs.append(resp.choices[0].message.content or "")
        return outputs

    def chat(self, messages: list[Turn], cfg: GenerationConfig) -> list[str]:
        if self.transport == "native":
            return self._chat_native(messages, cfg)
        return self._chat_openrouter(messages, cfg)

    def continue_from(
        self, messages: list[Turn], prefill: str, cfg: GenerationConfig
    ) -> list[str]:
        raise NotImplementedError(
            "Gemini cannot continue a raw prefill via API; the prefill, training, "
            "and probing experiments are Gemma-only (paper Section 3-4, Limitations)."
        )

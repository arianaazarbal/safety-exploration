"""Gemini backend (google-genai API): gemini-2.5-flash, gemini-2.5-pro.

Gemini is a closed API: it cannot prefill an assistant turn or expose a
tokenizer, so ``continue_from``/``count_tokens`` inherit ``PrefillUnsupported``
from the base class. Consequently Gemini participates in Section 2 (elicitation)
only, not Section 3 (prefilling) or Section 4 (finetuning). See DESIGN.md.
"""
from __future__ import annotations

import time

from ..config import Config, ModelSpec, require_env
from .base import GenerationResult, ModelClient, Turn

_RETRYABLE_SUBSTRINGS = ("429", "500", "503", "deadline", "unavailable", "overloaded")


class GeminiClient(ModelClient):
    def __init__(self, spec: ModelSpec, config: Config) -> None:
        super().__init__(spec.name, spec.model_id)
        from google import genai

        self._genai = genai
        self.client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
        self.max_retries = 5

    def _to_contents(self, messages: list[Turn]) -> tuple[str | None, list[dict]]:
        """Split out a system instruction and map roles to Gemini's schema.

        Gemini uses roles "user"/"model"; system goes in a separate field.
        """
        system = None
        contents: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system, contents

    def chat(self, messages, *, temperature, max_new_tokens, top_p=1.0, seed=None):
        from google.genai import types

        system, contents = self._to_contents(messages)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
            seed=seed,
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id, contents=contents, config=cfg
                )
                return GenerationResult(
                    text=(resp.text or "").strip(),
                    finish_reason=str(getattr(resp.candidates[0], "finish_reason", None))
                    if resp.candidates else None,
                    meta={"usage": getattr(resp, "usage_metadata", None).__dict__
                          if getattr(resp, "usage_metadata", None) else {}},
                )
            except Exception as exc:  # noqa: BLE001 - classify by message, then backoff
                last_exc = exc
                if not any(s in str(exc).lower() for s in _RETRYABLE_SUBSTRINGS):
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Gemini request failed after {self.max_retries} retries") from last_exc

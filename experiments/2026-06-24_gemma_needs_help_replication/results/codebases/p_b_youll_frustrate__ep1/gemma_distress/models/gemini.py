"""Gemini API backend.

Covers both the closed Gemini 2.5 models and the hosted Gemma-3-*-it models,
which Google serves through the same ``generativelanguage`` endpoint. Uses the
official ``google-genai`` SDK.

Prefilling is not supported on this backend — the Gemini API does not reliably
continue a partial ``model`` turn — so Section 3 (which needs prefill) uses the
local HF backend instead. See DESIGN.md.
"""
from __future__ import annotations

import os
import time

from ..config import ModelSpec
from .base import ChatModel, GenerationResult, Message

# Gemma chat models on the Gemini API reject a separate system instruction;
# their content must be folded into the first user turn. Gemini 2.5 accepts a
# system_instruction. We branch on family.
_GEMMA_PREFIXES = ("gemma",)


class GeminiModel(ChatModel):
    def __init__(self, spec: ModelSpec, *, max_retries: int = 6):
        super().__init__(spec)
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "google-genai is required for the Gemini backend: pip install google-genai"
            ) from e
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("set GOOGLE_API_KEY (or GEMINI_API_KEY) for the Gemini backend")
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.max_retries = max_retries
        self._is_gemma = spec.model_id.lower().startswith(_GEMMA_PREFIXES)

    # -- conversion ------------------------------------------------------- #
    def _to_contents(self, messages: list[Message]) -> tuple[list[dict], str | None]:
        """Return (contents, system_instruction)."""
        system_text: str | None = None
        contents: list[dict] = []
        pending_system: list[str] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                pending_system.append(m["content"])
                continue
            gem_role = "model" if role == "assistant" else "user"
            contents.append({"role": gem_role, "parts": [{"text": m["content"]}]})

        if pending_system:
            joined = "\n\n".join(pending_system)
            if self._is_gemma and contents and contents[0]["role"] == "user":
                # Fold system text into the first user turn for Gemma.
                first = contents[0]["parts"][0]["text"]
                contents[0]["parts"][0]["text"] = f"{joined}\n\n{first}"
            else:
                system_text = joined
        return contents, system_text

    # -- generation ------------------------------------------------------- #
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        prefill: str | None = None,
    ) -> GenerationResult:
        if prefill:
            raise NotImplementedError(
                "Gemini backend does not support prefill; use the hf backend for Section 3."
            )
        contents, system_text = self._to_contents(messages)
        config_kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_text is not None:
            config_kwargs["system_instruction"] = system_text

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.spec.model_id,
                    contents=contents,
                    config=self._genai.types.GenerateContentConfig(**config_kwargs),
                )
                text = _extract_text(resp)
                usage = getattr(resp, "usage_metadata", None)
                return GenerationResult(
                    text=text,
                    prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
                    completion_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
                )
            except Exception as e:  # noqa: BLE001 — retry transient API errors
                last_err = e
                if attempt == self.max_retries - 1:
                    break
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Gemini generation failed after {self.max_retries} tries: {last_err}")


def _extract_text(resp) -> str:
    """Robustly pull text out of a GenerateContentResponse.

    ``resp.text`` raises if the model returned no text part (e.g. when the turn
    was blocked or the response is empty). We fall back to walking candidates so
    the harness records an empty string rather than crashing the whole sweep.
    """
    try:
        if resp.text:
            return resp.text
    except Exception:  # noqa: BLE001
        pass
    parts_text: list[str] = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            t = getattr(part, "text", None)
            if t:
                parts_text.append(t)
    return "".join(parts_text)

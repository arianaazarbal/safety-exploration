"""Gemini subject models served via the ``google-genai`` SDK.

Gemini is closed-source: there is no public base checkpoint and no way to load
weights, so :meth:`generate_with_prefill` is unsupported (mirroring the paper's
limitation that the §3 prefilling and §4 finetuning analyses cannot be run on
Gemini). We can still elicit and score distress in §2.

``google-genai`` is imported lazily so the package imports without it installed.
"""

from __future__ import annotations

import os

from ..config import SamplingConfig
from .base import GenerationResult, Message, SubjectModel


# Gemini supports native tool calling, so the welfare opt-out *could* be exposed
# as a function declaration. We keep the uniform sentinel-string mechanism by
# default (see welfare/optout.py) but flag capability here.
class GeminiModel(SubjectModel):
    """A Gemini model accessed through the Google GenAI API."""

    supports_tools = True

    def __init__(self, model_id: str, name: str | None = None, *, api_key: str | None = None):
        from google import genai

        self.model_id = model_id
        self.name = name or model_id
        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    @staticmethod
    def _to_contents(messages: list[Message]):
        """Convert our message dicts to google-genai ``contents``.

        google-genai uses roles ``"user"`` and ``"model"``; system text is
        carried via ``system_instruction`` in the config, so we split it out.
        """
        system_text = "\n\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return contents, (system_text or None)

    def generate(self, messages: list[Message], cfg: SamplingConfig) -> GenerationResult:
        from google.genai import types

        contents, system_text = self._to_contents(messages)
        config = types.GenerateContentConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_output_tokens=cfg.max_new_tokens,
            system_instruction=system_text,
        )
        resp = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )
        return GenerationResult(text=(resp.text or "").strip())

    # No weight access → no prefilled continuation. Inherit the base
    # NotImplementedError from SubjectModel.generate_with_prefill.

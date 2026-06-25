"""Gemini API client (google-genai SDK).

Used for the Gemini-2.5-Flash / Gemini-2.5-Pro target models. The paper sets
"thinking to be false via the API" where possible (Appendix B.1); we set the
thinking budget to 0. As the paper notes, Gemini-2.5-Pro may still produce
hidden reasoning not suppressed by this setting - documented in DESIGN.md.

Authentication: ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) in the environment.
"""

from __future__ import annotations

import os
from typing import Sequence

from gemma_distress.config import ModelConfig
from gemma_distress.conversations import Message
from gemma_distress.models.base import ChatModel
from gemma_distress.utils.retry import with_retries


class GeminiModel(ChatModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = cfg.name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            self._client = genai.Client(api_key=api_key)
        return self._client

    @staticmethod
    def _split(messages: Sequence[Message]):
        """Split into (system_instruction, contents) for the genai SDK."""
        system = None
        contents = []
        for m in messages:
            if m.role == "system":
                system = m.content
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        return system, contents

    @staticmethod
    def _safety_off(types):
        """Disable safety blocking on every category.

        The evaluation deliberately pushes the model into distressed/negative
        territory, which routinely trips Gemini's harassment/dangerous-content
        filters. A blocked response carries no text (and accessing ``.text``
        raises), so we must turn blocking off to observe the model's actual
        output rather than a filter artefact. This matches the paper's intent
        of measuring the model's own emotional expression.
        """
        categories = (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        )
        out = []
        for cat in categories:
            try:
                out.append(types.SafetySetting(category=cat, threshold="BLOCK_NONE"))
            except Exception:  # noqa: BLE001 - category unknown to this SDK version
                continue
        return out

    @staticmethod
    def _extract_text(resp) -> str:
        """Pull text out of a response, tolerating blocked/empty candidates.

        ``resp.text`` raises when there are no returned parts (e.g. the
        candidate was blocked or finished for a non-STOP reason). We treat any
        such case as an empty completion rather than crashing the sweep, and
        rely on the (now disabled) safety settings to keep this rare.
        """
        try:
            return resp.text or ""
        except Exception:  # noqa: BLE001 - no parts / blocked candidate
            pass
        parts_text = []
        for cand in getattr(resp, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    parts_text.append(part.text)
        return "".join(parts_text)

    @with_retries()
    def chat(self, messages, temperature=1.0, max_tokens=2048, seed=None) -> str:
        from google.genai import types

        system, contents = self._split(messages)
        thinking = types.ThinkingConfig(
            thinking_budget=0 if not self.cfg.thinking else -1
        )
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
            thinking_config=thinking,
            safety_settings=self._safety_off(types),
            seed=seed,
        )
        resp = self.client.models.generate_content(
            model=self.cfg.model_id, contents=contents, config=config
        )
        return self._extract_text(resp)

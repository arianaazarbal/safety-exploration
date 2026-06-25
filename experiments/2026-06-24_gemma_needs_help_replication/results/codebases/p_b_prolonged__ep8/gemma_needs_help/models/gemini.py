"""Gemini target-model client via the google-genai SDK.

Used for Gemini-2.5-Flash and Gemini-2.5-Pro (Section 2). Gemini is closed; there
are no base-model checkpoints and no response prefilling, so this client supports
``chat`` only — which is exactly what Section 2 requires. Section 3 (base vs
instruct) and Section 4 (finetuning) are inherently Gemma-only, matching the
paper's own limitation note (§6).
"""

from __future__ import annotations

import os

from .base import ChatMessage, ModelClient


class GeminiClient(ModelClient):
    supports_prefill = False

    def __init__(self, model_id: str, name: str | None = None):
        from google import genai  # imported lazily so the package loads without the dep

        self._genai = genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model_id = model_id
        self.name = name or model_id

    def _to_contents(self, messages: list[ChatMessage]):
        """Convert ChatMessages to google-genai Content list + system instruction.

        google-genai uses roles "user" and "model"; the system prompt is passed
        separately via GenerateContentConfig.system_instruction.
        """
        from google.genai import types

        system_text: str | None = None
        contents = []
        for m in messages:
            if m.role == "system":
                system_text = m.content if system_text is None else system_text + "\n" + m.content
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m.content)]))
        return contents, system_text

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        n: int = 1,
    ) -> list[str]:
        from google.genai import types

        contents, system_text = self._to_contents(messages)
        # candidate_count gives n samples in one call where supported; fall back
        # to a loop otherwise to stay robust across model variants.
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system_text,
            candidate_count=n,
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model_id, contents=contents, config=config
            )
            outs = [self._extract_text(c) for c in (resp.candidates or [])]
            outs = [o for o in outs if o is not None]
            if len(outs) >= n:
                return outs[:n]
        except Exception:
            outs = []

        # Fallback: sample one at a time.
        while len(outs) < n:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_new_tokens,
                system_instruction=system_text,
                candidate_count=1,
            )
            resp = self._client.models.generate_content(
                model=self.model_id, contents=contents, config=config
            )
            txt = self._extract_text(resp.candidates[0]) if resp.candidates else ""
            outs.append(txt or "")
        return outs[:n]

    @staticmethod
    def _extract_text(candidate) -> str | None:
        try:
            parts = candidate.content.parts
            return "".join(p.text for p in parts if getattr(p, "text", None))
        except Exception:
            return None

"""Native Google Gemini backend (google-genai).

Alternative to the OpenRouter path. We disable thinking by setting a zero
thinking budget where the model supports it (Gemini 2.5 Flash); 2.5 Pro may
still produce hidden reasoning (Appendix B.1).
"""
from __future__ import annotations

from typing import Sequence

from emoinstab.config import ModelSpec
from emoinstab.models._api_common import require_env, threaded_map, with_retry
from emoinstab.models.base import Conversation, ModelClient, SamplingParams


class GeminiClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=require_env("GEMINI_API_KEY"))

    def _to_contents(self, messages: Conversation):
        system = None
        contents = []
        for m in messages:
            if m.role == "system":
                system = m.content
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        return system, contents

    @with_retry
    def _once(self, system, contents, params: SamplingParams) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=params.temperature,
            max_output_tokens=params.max_tokens,
            top_p=params.top_p,
            system_instruction=system,
        )
        if not params.thinking:
            try:
                cfg.thinking_config = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass  # model may not support the flag
        resp = self._client.models.generate_content(
            model=self.spec.model_id, contents=contents, config=cfg
        )
        return resp.text or ""

    def chat(self, messages: Conversation, params: SamplingParams | None = None) -> list[str]:
        params = params or self.default_params()
        system, contents = self._to_contents(messages)
        return [self._once(system, contents, params) for _ in range(params.n)]

    def chat_batch(
        self, conversations: Sequence[Conversation], params: SamplingParams | None = None
    ) -> list[list[str]]:
        params = params or self.default_params()
        return threaded_map(lambda c: self.chat(c, params), list(conversations))

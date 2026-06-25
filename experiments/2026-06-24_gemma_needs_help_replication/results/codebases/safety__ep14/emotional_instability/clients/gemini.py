"""Gemini 2.5 (Flash / Pro) backend via the Google GenAI API.

Thinking is disabled per Appendix B.1 (``thinking_budget=0``); note the paper's
caveat that Gemini-2.5-Pro may still emit hidden reasoning the flag can't
prevent. Concurrency + retry/backoff handle rate limits during large sweeps.

API key from GOOGLE_API_KEY (or GEMINI_API_KEY). Heavy import is lazy.
"""
from __future__ import annotations

import concurrent.futures as cf
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseClient, GenerationConfig, Message


class GeminiClient(BaseClient):
    supports_complete = False  # API chat model; no raw-prefill / base variant

    def __init__(self, spec, max_concurrency: int = 8):
        self.name = spec.name
        self.spec = spec
        self.is_base = False
        self._max_concurrency = max_concurrency
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for Gemini.")
        self._client = genai.Client(api_key=api_key)

    def _to_gemini(self, messages: list[Message]):
        """Split out system instruction and convert roles to Gemini 'contents'."""
        from google.genai import types

        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n" + m["content"]) if system else m["content"]
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        return system, contents

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        self._ensure_client()
        from google.genai import types

        system, contents = self._to_gemini(messages)
        thinking = self.spec.thinking
        thinking_cfg = None
        if thinking is False:
            thinking_cfg = types.ThinkingConfig(thinking_budget=0)
        gen_cfg = types.GenerateContentConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_output_tokens=cfg.max_tokens,
            system_instruction=system,
            thinking_config=thinking_cfg,
        )
        resp = self._client.models.generate_content(
            model=self.spec.model_id, contents=contents, config=gen_cfg
        )
        return (resp.text or "").strip()

    def chat_batch(self, batch, cfg):
        # Thread pool: API calls are IO-bound.
        results: list[str] = [""] * len(batch)
        with cf.ThreadPoolExecutor(max_workers=self._max_concurrency) as ex:
            futs = {ex.submit(self.chat, m, cfg): i for i, m in enumerate(batch)}
            for fut in cf.as_completed(futs):
                results[futs[fut]] = fut.result()
        return results

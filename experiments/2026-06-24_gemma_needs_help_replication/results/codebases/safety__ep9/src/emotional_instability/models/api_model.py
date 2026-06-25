"""Closed-source model access (Gemini) via an OpenAI-compatible endpoint.

The paper accesses Gemini through OpenRouter (Appendix B.1). We use the OpenAI
Python client pointed at OpenRouter's base URL, which exposes
`google/gemini-2.5-flash` and `google/gemini-2.5-pro`.

Thinking is disabled via the OpenRouter `reasoning` parameter where supported;
the paper notes Gemini-2.5-Pro may still emit hidden reasoning regardless.

Prefilling is not supported for these models, so Section 3 (which relies on
prefilled continuations + base models) is Gemma-only -- see DESIGN.md.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import require_env
from .base import Conversation, ModelClient


class APIModelClient(ModelClient):
    supports_prefill = False
    supports_raw = False

    def __init__(self, name: str, api_id: str, api_cfg: dict[str, Any],
                 generation_cfg: dict[str, Any], disable_thinking: bool = False):
        self.name = name
        self.api_id = api_id
        self.api_cfg = api_cfg
        self.gen_cfg = generation_cfg
        self.disable_thinking = disable_thinking
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.api_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                api_key=require_env(self.api_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
                timeout=self.api_cfg.get("timeout_s", 120),
            )
        return self._client

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
    def _one_completion(self, messages: list[dict[str, str]], temperature: float,
                        max_new_tokens: int) -> str:
        client = self._ensure_client()
        kwargs: dict[str, Any] = dict(
            model=self.api_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        extra_body: dict[str, Any] = {}
        if self.disable_thinking:
            # OpenRouter unified reasoning control.
            extra_body["reasoning"] = {"enabled": False}
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def generate(self, conversations, *, n=1, prefill=None, temperature=None,
                 max_new_tokens=None) -> list[list[str]]:
        if prefill is not None:
            raise NotImplementedError(
                "Prefilling is unsupported for API (Gemini) models; Section 3 is Gemma-only.")
        temp = self.gen_cfg["temperature"] if temperature is None else temperature
        max_tok = max_new_tokens or self.gen_cfg.get("max_new_tokens", 1024)

        # Flatten (conversation, sample) work items so all API calls run concurrently.
        jobs: list[list[dict[str, str]]] = []
        for conv in conversations:
            msgs = [m.as_dict() for m in conv]
            jobs.extend([msgs] * n)

        with ThreadPoolExecutor(max_workers=self.api_cfg.get("max_concurrency", 8)) as pool:
            flat = list(pool.map(
                lambda m: self._one_completion(m, temp, max_tok), jobs))

        # Re-group into [conversation][sample].
        out: list[list[str]] = []
        idx = 0
        for _ in conversations:
            out.append(flat[idx:idx + n])
            idx += n
        return out

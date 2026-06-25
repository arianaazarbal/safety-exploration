"""Gemini target backend via OpenRouter (OpenAI-compatible chat completions).

The paper accesses Gemini-2.5-Flash / -Pro through OpenRouter and sets thinking
to false via the API. We mirror that: OpenRouter's chat-completions endpoint
with `reasoning: {"enabled": false}` to suppress hidden reasoning where the
provider honours it (the paper notes Gemini-2.5-Pro may still produce hidden
reasoning regardless).

Gemini is closed-weights, so prefill continuation and probing are unsupported.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

from ..config import ModelSpec, RunConfig, SamplingConfig
from .base import ChatTurn, TargetBackend


class GeminiBackend(TargetBackend):
    def __init__(self, spec: ModelSpec, cfg: RunConfig):
        super().__init__(spec, cfg)
        self.api_key = cfg.resolved_openrouter_key()
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required for Gemini target models "
                "(set it in the environment or RunConfig.openrouter_api_key)."
            )
        self.base_url = cfg.openrouter_base_url.rstrip("/")
        self.session = requests.Session()

    def chat(self, messages: list[ChatTurn], sampling: SamplingConfig,
             system: Optional[str] = None) -> str:
        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(
            {"role": m["role"], "content": m["content"]} for m in messages
        )
        payload = {
            "model": self.spec.model_id,
            "messages": payload_messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_new_tokens,
        }
        # Suppress hidden reasoning where the provider supports it.
        if not sampling.thinking:
            payload["reasoning"] = {"enabled": False}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.api_max_retries):
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=headers, timeout=180,
                )
                if resp.status_code in (429, 500, 502, 503, 529):
                    raise requests.HTTPError(f"{resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001 - retry transient failures
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"Gemini call failed after {self.cfg.api_max_retries} retries: {last_err}"
        )

"""OpenRouter API client, used for the Gemini eval targets (and the optional
GPT-5-mini secondary judge).

The paper accesses Gemini via OpenRouter and disables thinking via the API
("In all cases, we set thinking to be false via the API."). We mirror that with
the `reasoning: {"enabled": false}` field that OpenRouter forwards to providers
that support it. Gemini 2.5 Pro may still emit hidden reasoning regardless, as
the paper notes.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

from .. import config
from .base import ChatMessage, GenerationResult, ModelClient


class OpenRouterClient(ModelClient):
    def __init__(self, name: str, model_id: str, *,
                 disable_thinking: bool = True, max_retries: int = 5):
        super().__init__(name)
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries

    def _chat(self, messages, *, temperature, max_new_tokens, prefill, stop):
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for Gemini targets.")

        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        if prefill:
            # OpenRouter assistant-prefix continuation. Not all providers honour
            # this; Gemini support is limited, so prefill experiments are
            # intended for the local Gemma models (see DESIGN.md).
            payload_messages.append({"role": "assistant", "content": prefill})

        body = {
            "model": self.model_id,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if stop:
            body["stop"] = stop
        if self.disable_thinking:
            body["reasoning"] = {"enabled": False}

        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{config.OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers, json=body, timeout=180,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"retryable status {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                finish = data["choices"][0].get("finish_reason", "")
                full = (prefill or "") + content
                return GenerationResult(
                    text=full, prefill=prefill or "",
                    finish_reason=finish, raw=data,
                )
            except Exception as exc:  # noqa: BLE001 - broad retry
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter request failed after {self.max_retries} retries: {last_err}")

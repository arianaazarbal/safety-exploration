"""Local vLLM (OpenAI-compatible) chat backend.

For faithful replication of the paper's *local* inference of Gemma. Start a
server with, e.g.:

    vllm serve google/gemma-3-27b-it --port 8000

then point a ModelConfig at it:

    ModelConfig(key="gemma-3-27b-it", backend="vllm",
                model="google/gemma-3-27b-it",
                base_url="http://localhost:8000/v1")

This uses the same HF model identifiers and chat template the paper used
(Appendix B.1), avoiding any API-provider quantisation/sampling differences.
"""

from __future__ import annotations

import os

import requests

from .base import ChatClient, ChatMessage, GenerationError


class VLLMClient(ChatClient):
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        max_retries: int = 5,
        timeout: float = 120.0,
    ):
        super().__init__(max_retries=max_retries, timeout=timeout)
        self.model = model
        self.base_url = base_url.rstrip("/")
        # vLLM ignores the key by default but the OpenAI client path expects one.
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")

    def _complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise GenerationError(f"vllm {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"unexpected vllm response: {data}") from exc
        if not content:
            raise GenerationError(f"empty content from vllm: {data}")
        return content

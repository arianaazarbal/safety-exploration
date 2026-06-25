"""Anthropic backend for the Claude judge and the Petri auditor/judge.

Used for:
  * Section 2.1 frustration judge (claude-sonnet-4-20250514)
  * Appendix C onset labelling + paraphrasing (Sonnet 4)
  * Appendix G Petri auditor (Sonnet 4) and transcript judge (Opus 4)

API key from ANTHROPIC_API_KEY. Heavy import is lazy. Retries with backoff for
rate limits. Concurrency for batch scoring during sweeps.
"""
from __future__ import annotations

import concurrent.futures as cf
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseClient, GenerationConfig, Message


class AnthropicClient(BaseClient):
    supports_complete = False

    def __init__(self, spec, max_concurrency: int = 8):
        self.name = spec.name
        self.spec = spec
        self.is_base = False
        self._max_concurrency = max_concurrency
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY for Claude judge/auditor.")
        self._client = anthropic.Anthropic()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        self._ensure_client()
        system = None
        conv = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n" + m["content"]) if system else m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            messages=conv,
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text").strip()

    def chat_batch(self, batch, cfg):
        results: list[str] = [""] * len(batch)
        with cf.ThreadPoolExecutor(max_workers=self._max_concurrency) as ex:
            futs = {ex.submit(self.chat, m, cfg): i for i, m in enumerate(batch)}
            for fut in cf.as_completed(futs):
                results[futs[fut]] = fut.result()
        return results

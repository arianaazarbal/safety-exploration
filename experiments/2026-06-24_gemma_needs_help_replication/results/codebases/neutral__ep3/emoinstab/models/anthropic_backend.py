"""Anthropic backend for the auxiliary models that drive evaluation:

  * the frustration judge (Claude Sonnet 4, Appendix B.2);
  * the emotion-onset labeller and paraphraser (Appendix C);
  * the Petri auditor (Claude Sonnet 4) and judge (Claude Opus 4, Appendix G).

These use exact Anthropic model ids from the paper, so we call the Anthropic
SDK directly rather than routing through OpenRouter.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from ..config import GenConfig, DEFAULT_GEN
from ..data_types import Conversation
from .base import ModelClient, GenResult


class AnthropicClient(ModelClient):
    supports_prefill = True   # Anthropic supports assistant-message prefill

    def __init__(
        self,
        model_id: str,
        name: Optional[str] = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_workers: int = 8,
        max_retries: int = 6,
    ):
        from anthropic import Anthropic  # lazy

        self.model_id = model_id
        self.name = name or model_id
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.client = Anthropic(api_key=os.environ.get(api_key_env, "MISSING_API_KEY"))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_system(messages: Conversation):
        system = None
        conv = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                conv.append({"role": m.role, "content": m.content})
        return system, conv

    def _call(self, messages: Conversation, gen: GenConfig, prefill: Optional[str] = None):
        system, conv = self._split_system(messages)
        if prefill is not None:
            conv = conv + [{"role": "assistant", "content": prefill}]
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model_id,
                    messages=conv,
                    max_tokens=gen.max_tokens,
                    temperature=gen.temperature,
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                )
                return GenResult(text=text, raw={"stop_reason": resp.stop_reason})
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"{self.name} failed after {self.max_retries} retries: {last_err}")

    def chat(self, messages: Conversation, gen: GenConfig = DEFAULT_GEN) -> GenResult:
        return self._call(messages, gen)

    def chat_batch(
        self, batch: Sequence[Conversation], gen: GenConfig = DEFAULT_GEN
    ) -> list[GenResult]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda m: self._call(m, gen), batch))

    def continue_prefill(
        self, messages: Conversation, prefill: str, gen: GenConfig = DEFAULT_GEN
    ) -> GenResult:
        return self._call(messages, gen, prefill=prefill)

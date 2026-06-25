"""Unified model-client interface.

Every backend (local Gemma via vLLM/HF, hosted Gemini via OpenRouter, Claude/GPT
judges via their SDKs) implements :class:`ModelClient`. The eval/training code is
written entirely against this interface so that swapping a target model is a
config change, never a code change.

Conventions
-----------
- ``messages`` is the OpenAI/Anthropic-style list of ``{"role", "content"}`` dicts
  with roles in {"system", "user", "assistant"}.
- ``generate`` returns plain assistant text (no role wrapper).
- ``generate_batch`` is the throughput path; backends that lack native batching
  fall back to a thread pool.
- ``prefill`` continues a partially-written assistant turn. This is essential for
  the Section 3 base-vs-instruct comparison and is only required of local backends.
"""

from __future__ import annotations

import abc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

Message = dict[str, str]


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_tokens: int = 2048
    top_p: float = 1.0
    stop: tuple[str, ...] | None = None
    seed: int | None = None


class ModelClient(abc.ABC):
    """Abstract text-generation client."""

    name: str
    supports_prefill: bool = False

    @abc.abstractmethod
    def generate(self, messages: Sequence[Message], cfg: GenConfig) -> str:
        """Generate a single assistant turn given a conversation prefix."""

    def generate_batch(
        self, batch: Sequence[Sequence[Message]], cfg: GenConfig, max_workers: int = 8
    ) -> list[str]:
        """Default batched implementation via threads. Backends with native
        batching (vLLM) override this for throughput."""
        if not batch:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(lambda m: self.generate(m, cfg), batch))

    def prefill(self, messages: Sequence[Message], prefix: str, cfg: GenConfig) -> str:
        """Continue an assistant turn that already starts with ``prefix``.

        Returns only the *continuation* (not including ``prefix``). Local backends
        override this; API backends generally cannot and raise.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support prefill")

    def prefill_batch(
        self,
        batch: Sequence[tuple[Sequence[Message], str]],
        cfg: GenConfig,
        max_workers: int = 8,
    ) -> list[str]:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(lambda mp: self.prefill(mp[0], mp[1], cfg), batch))

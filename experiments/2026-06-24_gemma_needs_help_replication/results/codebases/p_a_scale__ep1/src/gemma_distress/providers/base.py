"""Provider base classes, shared types, and the retry wrapper."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from ..logging_utils import get_logger

log = get_logger("providers")

# A chat message is the OpenAI/Anthropic-common shape: {"role", "content"}.
ChatMessage = dict[str, str]


@dataclass
class GenerationResult:
    """One model completion plus bookkeeping metadata."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    raw: Any = field(default=None, repr=False)


class RetryableError(Exception):
    """Raised by providers to signal a transient failure worth retrying."""


def with_retry(
    fn: Callable[[], Any],
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    retry_on: tuple[type[BaseException], ...] = (RetryableError,),
    label: str = "call",
) -> Any:
    """Run ``fn`` with exponential backoff + full jitter on transient errors.

    Non-retryable exceptions propagate immediately. This is the single choke
    point for resilience against rate limits and 5xx during long unattended runs.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203 - retry loop
            if attempt >= max_attempts:
                log.error("%s failed after %d attempts: %s", label, attempt, exc)
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = random.uniform(0, delay)  # full jitter
            log.warning(
                "%s transient error (attempt %d/%d), retrying in %.1fs: %s",
                label, attempt, max_attempts, delay, exc,
            )
            time.sleep(delay)


class ChatProvider:
    """Base class for chat completion backends.

    Subclasses implement :meth:`_generate`. ``model`` is the logical model name;
    ``model_id`` is the provider-specific identifier.
    """

    capabilities: set[str] = {"chat"}
    prefers_batch: bool = False  # local backends set True; runner uses lockstep batching

    def __init__(self, model: str, model_id: str, retry_cfg: dict | None = None, usage=None):
        self.model = model
        self.model_id = model_id
        self.retry_cfg = retry_cfg or {}
        self.usage = usage

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        result = self._generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )
        if self.usage is not None:
            self.usage.record(self.model, result.input_tokens, result.output_tokens)
        return result

    def generate_batch(
        self, batch: list[list[ChatMessage]], **kwargs: Any
    ) -> list[GenerationResult]:
        """Default batch = serial map. Local backends override for throughput."""
        return [self.generate(m, **kwargs) for m in batch]

    def _generate(self, messages: list[ChatMessage], **kwargs: Any) -> GenerationResult:
        raise NotImplementedError


@runtime_checkable
class PrefillCapable(Protocol):
    """Backends that can continue a partially-written assistant turn.

    Used by Section 3 (base/instruct prefill) and the recovery experiment. The
    ``prefill`` text is forced as the start of the assistant message and the
    model generates the continuation, which is what's returned (prefill stripped).
    """

    def prefill_continue(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> GenerationResult:
        ...


@runtime_checkable
class LogitsCapable(Protocol):
    """Backends exposing residual stream / unembedding for probing (Appendix I)."""

    def residual_logits(
        self,
        messages: list[ChatMessage],
        *,
        layers: list[int] | None = None,
        prefill: str | None = None,
        **kwargs: Any,
    ) -> Any:
        ...

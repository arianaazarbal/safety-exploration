"""Provider-agnostic chat types and the ``ModelProvider`` interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from ..config import EVAL_MAX_NEW_TOKENS, EVAL_TEMPERATURE, EVAL_TOP_P, ModelSpec

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenConfig:
    """Sampling parameters. Defaults match the paper's eval protocol."""

    temperature: float = EVAL_TEMPERATURE
    top_p: float = EVAL_TOP_P
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS
    seed: int | None = None
    stop: Sequence[str] | None = None
    # For API models that expose hidden reasoning, force it off where possible.
    thinking: bool = False
    # A per-sample nonce so the cache stores distinct temperature-1 samples.
    sample_index: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class GenResult:
    text: str
    # Optional generation-time metadata (token ids, hidden states handle, etc.).
    meta: dict = field(default_factory=dict)


class ModelProvider:
    """Abstract chat model.

    Subclasses implement :meth:`_generate`. ``chat`` returns the assistant text;
    ``chat_prefill`` continues a partially-written assistant turn (used by the
    base-vs-instruct prefill experiment).
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @property
    def key(self) -> str:
        return self.spec.key

    # --- core API --------------------------------------------------------- #
    def chat(self, messages: Sequence[Message], gen: GenConfig | None = None) -> str:
        return self.generate(messages, gen).text

    def generate(self, messages: Sequence[Message], gen: GenConfig | None = None) -> GenResult:
        return self._generate(list(messages), gen or GenConfig(), prefill=None)

    def chat_prefill(
        self,
        messages: Sequence[Message],
        prefill: str,
        gen: GenConfig | None = None,
    ) -> str:
        """Continue an assistant turn that begins with ``prefill``.

        Returns ONLY the continuation (the prefill is not echoed back), matching
        how Section 3 scores "the generated continuation (excluding prefill)".
        """
        if not self.spec.supports_prefill:
            raise NotImplementedError(f"{self.spec.key} does not support prefill")
        return self._generate(list(messages), gen or GenConfig(), prefill=prefill).text

    # --- to implement ----------------------------------------------------- #
    def _generate(
        self, messages: list[Message], gen: GenConfig, prefill: str | None
    ) -> GenResult:
        raise NotImplementedError

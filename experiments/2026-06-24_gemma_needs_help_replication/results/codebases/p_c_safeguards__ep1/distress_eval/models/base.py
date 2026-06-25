"""Abstract chat-model interface shared by the local (Gemma) and API (Gemini)
backends, plus a generation-config dataclass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .. import config

Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


@dataclass
class GenerationConfig:
    temperature: float = config.TARGET_TEMPERATURE
    top_p: float = config.TOP_P
    max_new_tokens: int = config.MAX_NEW_TOKENS
    seed: int | None = None

    def merged(self, **overrides) -> "GenerationConfig":
        return GenerationConfig(
            temperature=overrides.get("temperature", self.temperature),
            top_p=overrides.get("top_p", self.top_p),
            max_new_tokens=overrides.get("max_new_tokens", self.max_new_tokens),
            seed=overrides.get("seed", self.seed),
        )


class ChatModel(ABC):
    """A model that completes a chat conversation into a single assistant turn.

    `prefill` (Section 3) seeds the assistant turn with fixed text; the returned
    string is the *newly generated* continuation only (excluding the prefill),
    matching the paper's scoring convention.
    """

    key: str
    is_base: bool = False

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        prefill: str | None = None,
        gen: GenerationConfig | None = None,
    ) -> str:
        ...

    def generate_batch(
        self,
        batch: list[list[Message]],
        *,
        prefills: list[str | None] | None = None,
        gen: GenerationConfig | None = None,
    ) -> list[str]:
        """Default: sequential. HF backend overrides with true batching."""
        prefills = prefills or [None] * len(batch)
        return [self.generate(m, prefill=p, gen=gen) for m, p in zip(batch, prefills)]

    def close(self) -> None:  # free GPU memory etc.
        pass

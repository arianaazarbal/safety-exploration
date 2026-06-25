"""Abstract chat-model interface shared by local and API backends."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional, Sequence, TypedDict


class Message(TypedDict):
    role: str        # 'system' | 'user' | 'assistant'
    content: str


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    top_p: float = 1.0
    stop: Optional[Sequence[str]] = None


class ChatModel(abc.ABC):
    """Minimal chat interface used throughout the replication."""

    name: str
    family: str          # 'gemma' | 'gemini'
    supports_prefill: bool = False
    supports_hidden_states: bool = False

    @abc.abstractmethod
    def generate(self, messages: List[Message], cfg: GenConfig) -> str:
        """Generate a single assistant turn given a chat history."""

    def generate_batch(self, batch: List[List[Message]], cfg: GenConfig) -> List[str]:
        """Generate for a batch of conversations. Default: sequential fallback."""
        return [self.generate(m, cfg) for m in batch]

    def generate_with_prefill(self, messages: List[Message], prefill: str,
                              cfg: GenConfig) -> str:
        """Continue an assistant turn that already starts with `prefill`.

        Only meaningful for local models (base-vs-instruct prefill experiment).
        Returns the *continuation only* (excluding the prefill).
        """
        raise NotImplementedError(f"{self.name} does not support prefill")

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        pass

"""Abstract chat-model interface shared by all backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..messages import Message


class ChatModel(ABC):
    """A multi-turn chat model.

    Implementations must be safe to call concurrently from many asyncio tasks
    (the runner fans out with a semaphore). Each `generate` call is one
    assistant turn given the full prior conversation.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    async def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        """Return the assistant's next message given the conversation so far."""

    @property
    def name(self) -> str:
        return self.model_id

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}({self.model_id!r})"

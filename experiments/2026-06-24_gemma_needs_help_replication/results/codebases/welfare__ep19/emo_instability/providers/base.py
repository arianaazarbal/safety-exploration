"""Provider abstraction.

A `ChatModel` takes a list of messages and returns assistant text. The interface
is deliberately small but supports the two things the paper's experiments need
beyond ordinary chat:

  * `prefill`: seed the start of the assistant turn and have the model *continue*
    it. Required for the Section 3 base-vs-instruct prefilling experiment and the
    Section 4.2 recovery experiment. Instruct models accept a prefilled assistant
    turn; base models simply continue raw text.
  * `disable_thinking`: the paper sets thinking=false for all API models.

Messages use the OpenAI-style shape: {"role": "system"|"user"|"assistant",
"content": str}.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

Message = dict  # {"role": str, "content": str}


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_tokens: int = 2048
    disable_thinking: bool = True
    stop: list[str] | None = None


class ChatModel(abc.ABC):
    """Abstract chat model."""

    name: str

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        cfg: GenConfig,
        prefill: str | None = None,
    ) -> str:
        """Return the assistant's reply text.

        If `prefill` is given, the returned text is the *continuation only*
        (the prefill is not echoed back), matching how the prefilling
        experiments measure "the generated continuation excluding prefill".
        """

    def supports_prefill(self) -> bool:
        return False

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass

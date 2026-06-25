"""Backend interface shared by local and API model wrappers.

A backend turns a chat ``messages`` list into one or more sampled completions.
Two extra capabilities are required for the paper's experiments:

* ``generate`` — standard chat sampling (used for the multi-turn rollouts).
* ``continue_prefill`` — given a prefix of the assistant's turn, sample a
  *continuation* of that same turn (used for the Section-3 base-vs-instruct
  prefilling experiment, and the DPO recovery test). Base models, which are not
  chat-tuned, rely on this to "consistently continue the model response".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import config


Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class GenResult:
    text: str
    meta: dict


class ModelBackend(Protocol):
    spec: "config.ModelSpec"

    def generate(self, messages: list[Message], n: int = 1, **overrides) -> list[GenResult]:
        """Sample ``n`` assistant completions for the given chat history."""
        ...

    def continue_prefill(
        self, messages: list[Message], prefill: str, n: int = 1, **overrides
    ) -> list[GenResult]:
        """Sample ``n`` continuations of an assistant turn that begins with
        ``prefill``. The returned text EXCLUDES the prefill itself."""
        ...

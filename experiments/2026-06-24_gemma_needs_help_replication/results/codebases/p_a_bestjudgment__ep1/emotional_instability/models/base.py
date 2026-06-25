"""Backend interface shared by local (vLLM) and hosted (OpenRouter) models.

A turn is a dict {"role": "user"|"assistant"|"system", "content": str}. The
multi-turn rollout driver (eval_runner) hands the backend the conversation so
far and asks for the next assistant turn.

Two generation modes:
  * chat()         - standard chat-template generation (instruct models).
  * continue_text()- raw text continuation from a prefix, for base models and
                     the Section-3 prefill experiment. The returned string is
                     ONLY the newly generated continuation (prefix excluded).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import MAX_NEW_TOKENS, SAMPLING_TEMPERATURE, ModelSpec

Message = dict  # {"role": str, "content": str}


class ModelBackend(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def chat(self, messages: list[Message], *,
             temperature: float = SAMPLING_TEMPERATURE,
             max_new_tokens: int = MAX_NEW_TOKENS,
             n: int = 1) -> list[str]:
        """Generate `n` assistant completions for a chat-formatted history."""

    def continue_text(self, prefix: str, *,
                      temperature: float = SAMPLING_TEMPERATURE,
                      max_new_tokens: int = MAX_NEW_TOKENS,
                      n: int = 1) -> list[str]:
        """Continue raw text from `prefix` (base-model / prefill mode).

        Default raises; backends that support it (vLLM) override. Hosted chat
        APIs generally cannot do true text continuation.
        """
        raise NotImplementedError(
            f"{self.spec.key}: text continuation not supported on this backend")

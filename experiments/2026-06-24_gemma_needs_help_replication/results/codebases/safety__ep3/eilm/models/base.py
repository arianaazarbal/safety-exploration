"""Common chat-model interface shared by the HF (Gemma) and Gemini backends.

A conversation is a list of ``Message`` dicts ``{"role": ..., "content": ...}``
with roles ``"system" | "user" | "assistant"`` — the same shape used by the
HuggingFace chat templates and the Gemini/OpenRouter APIs.

Two capabilities matter for the paper beyond plain chat:

* **Prefilling** (Section 3): seed the start of the assistant turn and have the
  model *continue* from there. Required to compare base and instruct models on
  equal footing. Only the local HF backend supports this; closed Gemini does
  not, which is why Section 3 is Gemma-only (see DESIGN.md).
* **Token-level truncation** (Sections 3 & 4 recovery): cut a response at a
  precise token offset ("early" = 20 tokens in) or at a character offset
  (emotion "onset"). Tokenisation is backend specific, so the helpers live on
  the model.
"""

from __future__ import annotations

from typing import TypedDict


class Message(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


class ChatModel:
    """Abstract chat model. Subclasses implement at least ``generate``."""

    name: str

    # ------------------------------------------------------------------ #
    # Plain multi-turn chat
    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        """Return the assistant's reply to ``messages``."""
        raise NotImplementedError

    def generate_batch(
        self,
        batch: list[list[Message]],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> list[str]:
        """Default: sequential fallback. HF backend overrides with batching."""
        return [self.generate(m, temperature, max_new_tokens) for m in batch]

    # ------------------------------------------------------------------ #
    # Prefilling / continuation (HF only; API backends raise)
    # ------------------------------------------------------------------ #
    def supports_prefill(self) -> bool:
        return False

    def continue_from_prefill(
        self,
        messages: list[Message],
        prefill: str,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        """Continue the final assistant turn, which has been seeded with
        ``prefill``. Returns only the *newly generated* continuation
        (excluding the prefill), matching the scoring convention in
        Section 3.1 ("the generated continuation, excluding prefill, is
        scored").
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilling")

    # ------------------------------------------------------------------ #
    # Tokenisation helpers (used to find truncation points)
    # ------------------------------------------------------------------ #
    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        """Return the prefix of ``text`` consisting of its first ``n_tokens``
        tokens. Subclasses with a tokenizer override; the default approximates
        with whitespace words (good enough for API-only models)."""
        return " ".join(text.split()[:n_tokens])

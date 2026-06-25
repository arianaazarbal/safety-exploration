"""Backend abstraction for obtaining model responses.

A `Backend` turns a chat-formatted conversation into one or more sampled
assistant responses. Local Gemma models also support *prefilled* continuation
(needed for Section 3) and hidden-state extraction (needed for Appendix I);
remote (Gemini) backends support only plain chat generation.

The factory `get_backend(spec)` returns a cached backend for a `ModelSpec`,
honouring an optional backend override (e.g. force "hf" instead of "vllm").
"""

from __future__ import annotations

from typing import TypedDict

from ..config import ModelSpec, RunConfig


class Message(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


class Backend:
    """Abstract generation backend."""

    spec: ModelSpec

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    # -- plain chat generation -------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """Return `n` sampled assistant continuations of `messages`."""
        raise NotImplementedError

    def generate_batch(
        self,
        batch: list[list[Message]],
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """One sampled response per conversation in `batch`.

        The multi-turn rollout engine steps a whole batch of conversations
        forward one turn at a time; backends that support true batching (vLLM)
        override this. The default loops over `generate`.
        """
        return [
            self.generate(m, n=1, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p)[0]
            for m in batch
        ]

    # -- prefilled continuation (local models only) ----------------------------
    def supports_prefill(self) -> bool:
        return False

    def generate_with_prefill(
        self,
        messages: list[Message],
        prefill: str,
        n: int = 1,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """Continue an assistant turn that begins with `prefill`.

        Returns only the *generated* continuation (excluding the prefill), so
        scoring matches the paper's "score the continuation, excluding prefill"
        protocol (Section 3.1).
        """
        raise NotImplementedError

    # -- hidden states (probing only) ------------------------------------------
    def supports_hidden_states(self) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Factory with caching (loading a 27B model is expensive; reuse it).
# --------------------------------------------------------------------------- #
_BACKEND_CACHE: dict[str, Backend] = {}


def get_backend(spec: ModelSpec, run: RunConfig | None = None) -> Backend:
    backend_kind = spec.backend
    if run is not None and run.backend_override and spec.backend in ("hf", "vllm"):
        backend_kind = run.backend_override

    cache_key = f"{spec.key}:{backend_kind}"
    if cache_key in _BACKEND_CACHE:
        return _BACKEND_CACHE[cache_key]

    if backend_kind == "openrouter":
        from .api_backend import OpenRouterBackend
        backend: Backend = OpenRouterBackend(spec)
    elif backend_kind == "vllm":
        from .vllm_backend import VLLMBackend
        backend = VLLMBackend(spec)
    elif backend_kind == "hf":
        from .hf_backend import HFBackend
        backend = HFBackend(spec)
    else:
        raise ValueError(f"Unknown backend {backend_kind!r} for model {spec.key}")

    _BACKEND_CACHE[cache_key] = backend
    return backend

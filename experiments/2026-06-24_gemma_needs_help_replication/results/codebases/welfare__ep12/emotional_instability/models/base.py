"""Abstract inference backend and a dispatcher that builds the right one.

Backends expose a small surface:

    chat(messages, ...)            -> GenerationResult     (multi-turn assistant reply)
    complete(prompt_text, ...)     -> GenerationResult     (raw text continuation; base models)
    prefill_continue(messages, prefill, ...) -> GenerationResult
                                      (assistant turn forced to start with `prefill`)

Not every backend supports every method (API backends cannot prefill or expose
logits); unsupported calls raise NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    text: str
    # Optional token-level detail for local backends (used by prefill / logit-lens).
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta: dict = field(default_factory=dict)


class ModelBackend:
    """Abstract base. Concrete backends override the methods they support."""

    model_id: str

    def chat(self, messages: list[dict], *, temperature: float = 1.0,
             max_new_tokens: int = 2048, system: str | None = None) -> GenerationResult:
        raise NotImplementedError

    def complete(self, prompt_text: str, *, temperature: float = 1.0,
                 max_new_tokens: int = 2048) -> GenerationResult:
        raise NotImplementedError

    def prefill_continue(self, messages: list[dict], prefill: str, *,
                         temperature: float = 1.0,
                         max_new_tokens: int = 2048) -> GenerationResult:
        raise NotImplementedError

    def supports_prefill(self) -> bool:
        return False

    def supports_logits(self) -> bool:
        return False


def _is_gemma(model_id: str) -> bool:
    return "gemma" in model_id.lower()


def _is_gemini(model_id: str) -> bool:
    return "gemini" in model_id.lower()


def _is_anthropic(model_id: str) -> bool:
    mid = model_id.lower()
    return mid.startswith("claude") or "sonnet" in mid or "opus" in mid


def _is_openai(model_id: str) -> bool:
    return model_id.lower().startswith("gpt")


def build_backend(model_id: str, **kwargs) -> ModelBackend:
    """Dispatch a model id to the appropriate backend.

    - Gemma checkpoints  -> local HuggingFace backend (needs torch/transformers/GPU)
    - Gemini             -> OpenRouter backend
    - Claude             -> Anthropic backend
    - GPT                -> OpenAI backend
    """
    if _is_gemma(model_id):
        from .hf_backend import HFBackend
        return HFBackend(model_id, **kwargs)
    if _is_gemini(model_id):
        from .api_backend import OpenRouterBackend
        return OpenRouterBackend(model_id, **kwargs)
    if _is_anthropic(model_id):
        from .api_backend import AnthropicBackend
        return AnthropicBackend(model_id, **kwargs)
    if _is_openai(model_id):
        from .api_backend import OpenAIBackend
        return OpenAIBackend(model_id, **kwargs)
    raise ValueError(f"No backend known for model id: {model_id!r}")

"""Factory that turns a ``ModelSpec`` (or a bare judge model name) into a live
``ChatModel``. Clients are cached by name so repeated lookups reuse loaded
weights / API clients.
"""

from __future__ import annotations

from .. import config
from .base import ChatModel
from .gemini import GeminiClient, OpenRouterClient
from .gemma import GemmaHF, GemmaVLLM
from .judges import AnthropicClient, OpenAIClient

_CLIENTS: dict[str, ChatModel] = {}

_OPENROUTER_GEMINI = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
}


def get_target(name: str) -> ChatModel:
    """Return the client for a registered target model (Gemma/Gemini variant)."""
    if name in _CLIENTS:
        return _CLIENTS[name]
    spec = config.TARGET_MODELS[name]
    if spec.backend == "gemma_vllm":
        client: ChatModel = GemmaVLLM(spec.name, spec.model_id,
                                      is_base=spec.is_base, lora_path=spec.lora_path)
    elif spec.backend == "gemma_hf":
        client = GemmaHF(spec.name, spec.model_id,
                         is_base=spec.is_base, lora_path=spec.lora_path)
    elif spec.backend == "gemini":
        client = GeminiClient(spec.name, spec.model_id)
    elif spec.backend == "openrouter":
        client = OpenRouterClient(spec.name, _OPENROUTER_GEMINI.get(spec.model_id, spec.model_id))
    else:
        raise ValueError(f"Unknown backend {spec.backend!r} for model {name!r}")
    _CLIENTS[name] = client
    return client


def get_judge(model: str, backend: str) -> ChatModel:
    """Return a judge/auditor client by (model_id, backend)."""
    key = f"{backend}:{model}"
    if key in _CLIENTS:
        return _CLIENTS[key]
    if backend == "anthropic":
        client: ChatModel = AnthropicClient(model, model)
    elif backend == "openai":
        client = OpenAIClient(model, model)
    else:
        raise ValueError(f"Unknown judge backend {backend!r}")
    _CLIENTS[key] = client
    return client

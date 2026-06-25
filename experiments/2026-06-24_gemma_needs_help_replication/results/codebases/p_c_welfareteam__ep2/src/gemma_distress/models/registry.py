"""Factory that builds the right client for a :class:`ModelConfig`.

Clients are cached by name within a process so that, e.g., a 27B Gemma is only
loaded into GPU memory once even if referenced as both target and data
generator.
"""

from __future__ import annotations

from gemma_distress.config import ModelConfig, PipelineConfig
from gemma_distress.models.base import ChatModel

_CACHE: dict[str, ChatModel] = {}


def build_model(cfg: ModelConfig) -> ChatModel:
    if cfg.name in _CACHE:
        return _CACHE[cfg.name]
    if cfg.provider == "huggingface":
        from gemma_distress.models.huggingface_client import HuggingFaceModel

        model: ChatModel = HuggingFaceModel(cfg)
    elif cfg.provider == "gemini":
        from gemma_distress.models.gemini_client import GeminiModel

        model = GeminiModel(cfg)
    elif cfg.provider == "anthropic":
        from gemma_distress.models.anthropic_client import AnthropicModel

        model = AnthropicModel(cfg)
    elif cfg.provider == "openai":
        from gemma_distress.models.openai_client import OpenAIModel

        model = OpenAIModel(cfg)
    else:
        raise ValueError(f"unknown provider {cfg.provider!r} for model {cfg.name!r}")
    _CACHE[cfg.name] = model
    return model


def get_model(pipeline: PipelineConfig, name: str) -> ChatModel:
    if name not in pipeline.models:
        raise KeyError(f"model {name!r} not in registry; known: {list(pipeline.models)}")
    return build_model(pipeline.models[name])


def clear_cache() -> None:
    _CACHE.clear()

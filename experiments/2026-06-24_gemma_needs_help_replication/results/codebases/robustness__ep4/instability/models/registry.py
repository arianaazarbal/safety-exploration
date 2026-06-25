"""Factory: turn a ModelSpec / JudgeSpec into a live ChatModel."""
from __future__ import annotations

from ..config import Backend, JudgeSpec, ModelSpec, TARGET_MODELS
from .base import ChatModel


def load_model(spec, **backend_kwargs) -> ChatModel:
    """Instantiate the right backend for `spec` (ModelSpec or JudgeSpec)."""
    backend = spec.backend
    if backend in (Backend.OPENROUTER, Backend.GOOGLE):
        if backend == Backend.GOOGLE:
            from .google_genai import GoogleModel

            return GoogleModel(_as_modelspec(spec), **backend_kwargs)
        from .openrouter import OpenRouterModel

        return OpenRouterModel(_as_modelspec(spec), **backend_kwargs)
    if backend == Backend.LOCAL_HF:
        from .local_hf import LocalHFModel

        return LocalHFModel(spec, **backend_kwargs)
    if backend == Backend.VLLM:
        from .vllm_backend import VLLMModel

        return VLLMModel(spec, **backend_kwargs)
    raise ValueError(f"Unknown backend: {backend}")


def _as_modelspec(spec) -> ModelSpec:
    """Judges are specified as JudgeSpec; wrap them as a minimal ModelSpec."""
    if isinstance(spec, ModelSpec):
        return spec
    if isinstance(spec, JudgeSpec):
        return ModelSpec(
            key=spec.key, backend=spec.backend, model_id=spec.model_id,
            family="judge", disable_thinking=False,
        )
    raise TypeError(type(spec))


def load_target(key: str, **kwargs) -> ChatModel:
    return load_model(TARGET_MODELS[key], **kwargs)

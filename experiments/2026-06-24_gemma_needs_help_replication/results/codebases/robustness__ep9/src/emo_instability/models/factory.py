"""Construct a :class:`ModelClient` from a :class:`ModelSpec`."""
from __future__ import annotations

from ..config import ENDPOINTS, MODEL_REGISTRY, ModelSpec, get_model
from .base import ModelClient


def build_client(
    model: str | ModelSpec,
    *,
    adapter_path: str | None = None,
    prefer_hf_for_gemma: bool = False,
    **backend_kwargs,
) -> ModelClient:
    """Instantiate the appropriate backend.

    Parameters
    ----------
    model: registry key or ModelSpec.
    adapter_path: path to a LoRA adapter (DPO/SFT output) to load on top of a
        local Gemma model.
    prefer_hf_for_gemma: force the transformers backend (needed for prefill
        continuation control and for base models); otherwise instruct Gemma uses
        vLLM for throughput.
    """
    spec = get_model(model) if isinstance(model, str) else model

    if spec.backend == "anthropic":
        from .api_model import AnthropicClient

        return AnthropicClient(spec.model_id, spec.key)

    if spec.backend == "openai":
        from .api_model import OpenAICompatClient

        # GPT-5-mini cross-check judge uses the OpenAI endpoint directly; Gemini
        # and other OpenRouter ids use OpenRouter.
        if spec.family == "openai":
            return OpenAICompatClient(
                spec.model_id, spec.key,
                base_url="https://api.openai.com/v1",
                api_key_env=ENDPOINTS.openai_api_key_env,
            )
        return OpenAICompatClient(spec.model_id, spec.key, **backend_kwargs)

    # Local Gemma (gemma family). Base models and prefill control require HF.
    if spec.kind == "base" or prefer_hf_for_gemma or spec.backend == "hf":
        from .hf_model import HFClient

        return HFClient(
            spec.model_id, spec.key,
            adapter_path=adapter_path,
            is_base_model=spec.kind == "base",
            **backend_kwargs,
        )

    from .vllm_model import VLLMClient

    return VLLMClient(spec.model_id, spec.key, adapter_path=adapter_path, **backend_kwargs)


__all__ = ["build_client"]

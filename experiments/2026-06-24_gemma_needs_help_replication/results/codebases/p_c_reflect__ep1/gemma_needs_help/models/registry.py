"""Factory functions building ChatModel targets and JudgeClient utilities
from config."""
from __future__ import annotations

from ..config import Config, ModelSpec
from .base import ChatModel
from .llm_client import JudgeClient


def build_model(
    config: Config,
    name: str,
    *,
    adapter_path: str | None = None,
    prefer_vllm: bool = True,
) -> ChatModel:
    """Construct a target ChatModel for `name` from config.

    `adapter_path` (open-weights only) loads a LoRA adapter — used to evaluate
    finetuned Gemma models. `prefer_vllm` uses the vLLM backend for local
    models when it is importable, falling back to transformers.
    """
    spec: ModelSpec = config.model(name)

    if spec.backend == "api":
        if adapter_path is not None:
            raise PermissionError(
                f"Cannot attach an adapter to closed model '{name}'."
            )
        from .api_backend import APIChatModel

        return APIChatModel(
            name=spec.name,
            api_id=spec.api_id,
            family=spec.family,
            role=spec.role,
            api_provider=spec.api_provider or "openrouter",
            thinking=spec.thinking,
        )

    # Local open-weights backends (Gemma).
    if spec.backend == "vllm" or (spec.backend == "hf" and prefer_vllm):
        try:
            import vllm  # noqa: F401

            from .vllm_backend import VLLMChatModel

            return VLLMChatModel(
                name=spec.name,
                hf_id=spec.hf_id,
                family=spec.family,
                role=spec.role,
                chat_template=spec.chat_template,
                adapter_path=adapter_path,
            )
        except ImportError:
            pass  # fall back to transformers

    from .hf_backend import HFChatModel

    return HFChatModel(
        name=spec.name,
        hf_id=spec.hf_id,
        family=spec.family,
        role=spec.role,
        chat_template=spec.chat_template,
        adapter_path=adapter_path,
    )


def build_judge_client(config: Config, key: str) -> JudgeClient:
    """Build a JudgeClient for a named entry under config['judges']."""
    spec = config["judges"][key]
    return JudgeClient(
        provider=spec["provider"],
        model=spec["model"],
        temperature=spec.get("temperature", 0.0),
        max_tokens=spec.get("max_tokens", 1024),
    )

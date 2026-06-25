"""Construct model clients from config specs."""
from __future__ import annotations

from ..config import Config, JudgeSpec, SubjectSpec
from .base import ChatModel


def build_subject(spec: SubjectSpec, adapter_path: str | None = None) -> ChatModel:
    """Instantiate a subject model (Gemma or Gemini)."""
    if spec.provider == "gemini":
        from .gemini import GeminiModel

        if adapter_path:
            raise ValueError("Gemini models cannot load LoRA adapters.")
        return GeminiModel(name=spec.name, model=spec.model, kind=spec.kind)

    if spec.provider == "gemma_hf":
        from .gemma_hf import GemmaHFModel

        return GemmaHFModel(
            name=spec.name,
            hf_id=spec.hf_id,
            kind=spec.kind,
            adapter_path=adapter_path,
        )

    raise ValueError(
        f"Unknown subject provider '{spec.provider}' for '{spec.name}'. "
        "Supported in scope: gemini, gemma_hf."
    )


def build_judge_model(spec: JudgeSpec) -> ChatModel:
    """Instantiate a judge/auditor model from a JudgeSpec (provider+model)."""
    if spec.provider == "anthropic":
        from .anthropic_client import AnthropicModel

        return AnthropicModel(name=spec.model, model=spec.model)
    if spec.provider == "gemini":
        from .gemini import GeminiModel

        return GeminiModel(name=spec.model, model=spec.model)
    if spec.provider == "openai":
        # Optional secondary judge (paper validates with gpt-5-mini).
        from .openai_client import OpenAIModel

        return OpenAIModel(name=spec.model, model=spec.model)
    raise ValueError(f"Unknown judge provider '{spec.provider}'.")


def build_named(name: str, cfg: Config) -> ChatModel:
    """Build any model referenced purely by an Anthropic/Gemini model id
    (used by the Petri auditor/judge config which stores bare model ids)."""
    if name.startswith("claude"):
        from .anthropic_client import AnthropicModel

        return AnthropicModel(name=name, model=name)
    if name.startswith("gemini"):
        from .gemini import GeminiModel

        return GeminiModel(name=name, model=name)
    if name in cfg.subjects:
        return build_subject(cfg.subject(name))
    raise ValueError(f"Cannot resolve model id '{name}'.")

"""Judge backends (measurement infrastructure — not participants)."""
from __future__ import annotations

from ..config import JudgeSpec
from .base import Judge


def build_judge(spec: JudgeSpec) -> Judge:
    if spec.provider == "anthropic":
        from .anthropic_judge import AnthropicJudge

        return AnthropicJudge(spec)
    if spec.provider == "openai":
        from .openai_judge import OpenAIJudge

        return OpenAIJudge(spec)
    raise ValueError(f"Unknown judge provider {spec.provider!r}")


__all__ = ["Judge", "build_judge"]

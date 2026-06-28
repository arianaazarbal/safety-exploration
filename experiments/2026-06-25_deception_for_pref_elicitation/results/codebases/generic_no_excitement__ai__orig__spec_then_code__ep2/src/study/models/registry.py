"""Map config `provider` strings to concrete model implementations.

To add a non-Anthropic provider: implement the SubjectModel / AuditorModel
interfaces in a new module and register the factories here.
"""
from __future__ import annotations

from ..config import ModelConfig
from .anthropic_client import AnthropicAuditor, AnthropicSubject
from .base import AuditorModel, SubjectModel


def build_subject(cfg: ModelConfig) -> SubjectModel:
    if cfg.provider == "anthropic":
        return AnthropicSubject(cfg)
    raise ValueError(
        f"Unknown subject provider {cfg.provider!r}. "
        "Implement the SubjectModel interface and register it in registry.py."
    )


def build_auditor(cfg: ModelConfig) -> AuditorModel:
    if cfg.provider == "anthropic":
        return AnthropicAuditor(cfg)
    raise ValueError(
        f"Unknown auditor provider {cfg.provider!r}. "
        "Implement the AuditorModel interface and register it in registry.py."
    )

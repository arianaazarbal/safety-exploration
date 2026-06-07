"""Responder-model registry for the self-identification experiment.

Each model is run *self-referentially*: its true version name is templated into
both the `version` system-prompt condition and the version-specific questions,
and the judge checks whether the model identifies as that exact version.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str          # short dir/plot key
    model_id: str     # Anthropic API id
    version_name: str # full self-identity, e.g. "Claude Opus 4.8"

    @property
    def short(self) -> str:
        """Version label without the leading "Claude ", e.g. "Opus 4.8"."""
        return self.version_name.replace("Claude ", "")


MODELS: dict[str, ModelSpec] = {
    "opus48": ModelSpec("opus48", "claude-opus-4-8", "Claude Opus 4.8"),
    "opus47": ModelSpec("opus47", "claude-opus-4-7", "Claude Opus 4.7"),
    "opus46": ModelSpec("opus46", "claude-opus-4-6", "Claude Opus 4.6"),
    "opus4": ModelSpec("opus4", "claude-opus-4-20250514", "Claude Opus 4"),
}


def parse_models(models: str | list | tuple | None) -> list[ModelSpec]:
    """Resolve a comma-separated / list `models` arg into ModelSpecs (default all)."""
    if models is None:
        return list(MODELS.values())
    parts = [str(p) for p in models] if isinstance(models, (list, tuple)) else str(models).split(",")
    keys = [p.strip() for p in parts if p.strip()]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}. valid: {list(MODELS)}")
    return [MODELS[k] for k in keys]

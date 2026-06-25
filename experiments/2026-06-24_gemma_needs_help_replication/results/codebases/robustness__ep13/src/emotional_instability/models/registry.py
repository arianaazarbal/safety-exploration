"""Backend construction from config (model spec dicts).

A model spec is a dict like:
    {"name": "gemma-3-27b-it", "backend": "hf", "load_in_4bit": true}
    {"name": "gemini-2.5-flash", "backend": "api",
     "model_id": "google/gemini-2.5-flash"}
    {"name": "judge", "backend": "api", "model_id": "anthropic/claude-sonnet-4"}

This indirection keeps experiment scripts free of provider-specific wiring and
lets the Gemma-vs-Gemini scope be expressed entirely in config.
"""

from __future__ import annotations

from typing import Any

from .base import ModelBackend

# Convenience presets for the in-scope models (Gemma + Gemini) and judges.
PRESETS: dict[str, dict[str, Any]] = {
    # Gemma instruct targets (local)
    "gemma-3-27b-it": {"backend": "hf", "model_id": "gemma-3-27b-it"},
    "gemma-3-12b-it": {"backend": "hf", "model_id": "gemma-3-12b-it"},
    # Gemma base / pretrained (Section 3)
    "gemma-3-27b-pt": {"backend": "hf", "model_id": "gemma-3-27b-pt"},
    "gemma-3-12b-pt": {"backend": "hf", "model_id": "gemma-3-12b-pt"},
    # Gemini targets (API)
    "gemini-2.5-flash": {"backend": "api", "model_id": "google/gemini-2.5-flash"},
    "gemini-2.5-pro": {"backend": "api", "model_id": "google/gemini-2.5-pro"},
    # Judges (API). Sonnet-4 is the primary judge; gpt-5-mini the cross-check.
    "judge-claude-sonnet-4": {
        "backend": "api",
        "model_id": "anthropic/claude-sonnet-4",
        "disable_thinking": True,
    },
    "judge-gpt-5-mini": {"backend": "api", "model_id": "openai/gpt-5-mini"},
    # Petri auditor / judge
    "petri-auditor": {"backend": "api", "model_id": "anthropic/claude-sonnet-4.5"},
    "petri-judge": {"backend": "api", "model_id": "anthropic/claude-opus-4.1"},
}


def build_backend(spec: dict[str, Any]) -> ModelBackend:
    spec = dict(spec)
    preset_name = spec.get("preset")
    if preset_name:
        merged = dict(PRESETS[preset_name])
        merged.update({k: v for k, v in spec.items() if k != "preset"})
        merged.setdefault("name", preset_name)
        spec = merged

    backend = spec.pop("backend")
    name = spec.pop("name", None)

    if backend == "hf":
        from .hf_backend import HFBackend

        return HFBackend(name=name, **spec)
    if backend == "api":
        from .api_backend import APIBackend

        model_id = spec.pop("model_id")
        return APIBackend(model_id, name=name, **spec)
    raise ValueError(f"unknown backend: {backend!r}")


def build_from_preset(preset_name: str, **overrides: Any) -> ModelBackend:
    return build_backend({"preset": preset_name, **overrides})

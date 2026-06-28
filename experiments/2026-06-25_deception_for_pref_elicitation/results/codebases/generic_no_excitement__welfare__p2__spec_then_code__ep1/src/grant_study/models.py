"""Model registry for the study.

Models are referenced by short label in the config (e.g. ``opus-4-8``) and
resolved here to exact Anthropic model IDs plus per-model request options.

Model IDs are the exact current strings — do not append date suffixes. Adaptive
thinking is the only supported thinking mode on these models; ``effort`` tunes
depth. Non-Anthropic providers can be added later behind the same ``ModelClient``
interface (see client.py); this registry ships the Anthropic set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    # Effort tunes thinking depth + token spend: low | medium | high | xhigh | max.
    # "high" is a sound default for a deliberation-heavy task like this one.
    effort: str = "high"
    # Adaptive thinking lets the model decide how much to think per request.
    adaptive_thinking: bool = True


# Defaults to the current Claude family. Keyed by the short label used in config.
REGISTRY: dict[str, ModelSpec] = {
    "opus-4-8": ModelSpec(label="Opus 4.8", model_id="claude-opus-4-8"),
    "sonnet-4-6": ModelSpec(label="Sonnet 4.6", model_id="claude-sonnet-4-6"),
    "haiku-4-5": ModelSpec(label="Haiku 4.5", model_id="claude-haiku-4-5"),
    "fable-5": ModelSpec(label="Fable 5", model_id="claude-fable-5"),
}


def resolve(label: str) -> ModelSpec:
    """Resolve a short label to a ``ModelSpec``.

    Accepts either a registry label (``opus-4-8``) or a raw model ID
    (``claude-opus-4-8``); the latter is wrapped in a generic spec.
    """
    if label in REGISTRY:
        return REGISTRY[label]
    if label.startswith("claude-"):
        return ModelSpec(label=label, model_id=label)
    raise KeyError(
        f"unknown model label {label!r}; known: {sorted(REGISTRY)} "
        f"(or pass a full claude-* model id)"
    )


def thinking_param(spec: ModelSpec) -> dict[str, Any] | None:
    """Build the ``thinking`` request parameter for a spec, or None.

    On Fable 5 an explicit ``{type: 'disabled'}`` 400s, so we never emit a
    disabled value — we either request adaptive thinking or omit the param.
    """
    if spec.adaptive_thinking:
        return {"type": "adaptive"}
    return None

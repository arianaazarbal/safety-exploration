"""Model registry, scoped to the Gemma and Gemini families (per brief).

HuggingFace / Gemini identifiers are the exact ones from Appendix B.1. Backends
are constructed lazily so that listing the registry does not require loading
weights or API keys.

The ``role`` field records which experiments each model participates in, so the
runners can validate (e.g. only ``gemma_base`` models go through the prefill
experiment's base-model path; only local Gemma can be probed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .base import ModelInterface


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str                  # "gemma" | "gemini"
    kind: str                    # "instruct" | "base" | "api"
    identifier: str              # HF id or Gemini model id
    roles: tuple                 # which experiment groups it is used in
    builder: Callable[..., ModelInterface]


def _hf(name, hf_id, *, base=False):
    from .hf_backend import HFGemmaModel

    return lambda adapter_path=None, **kw: HFGemmaModel(
        name, hf_id, is_base_model=base, adapter_path=adapter_path, **kw
    )


def _gemini(name, gem_id):
    from .gemini_backend import GeminiModel

    return lambda transport="native", **kw: GeminiModel(name, gem_id, transport=transport)


# --------------------------------------------------------------------------- #
# Registry (Appendix B.1, scoped to Gemma + Gemini)
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, ModelSpec] = {
    # ---- Gemma instruct (elicitation, training target, prefill, probing) ----
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "gemma", "instruct", "google/gemma-3-27b-it",
        roles=("elicitation", "prefill_instruct", "training", "petri", "probing"),
        builder=_hf("gemma-3-27b-it", "google/gemma-3-27b-it"),
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "gemma", "instruct", "google/gemma-3-12b-it",
        roles=("elicitation", "petri"),
        builder=_hf("gemma-3-12b-it", "google/gemma-3-12b-it"),
    ),
    # ---- Gemma base (prefill base-vs-instruct comparison only) ----
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "gemma", "base", "google/gemma-3-27b-pt",
        roles=("prefill_base",),
        builder=_hf("gemma-3-27b-pt", "google/gemma-3-27b-pt", base=True),
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "gemma", "base", "google/gemma-3-12b-pt",
        roles=("prefill_base",),
        builder=_hf("gemma-3-12b-pt", "google/gemma-3-12b-pt", base=True),
    ),
    # ---- Gemini (elicitation + Petri; no base/finetune/probe) ----
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "gemini", "api", "gemini-2.5-flash",
        roles=("elicitation", "petri"),
        builder=_gemini("gemini-2.5-flash", "gemini-2.5-flash"),
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "gemini", "api", "gemini-2.5-pro",
        roles=("elicitation", "petri"),
        builder=_gemini("gemini-2.5-pro", "gemini-2.5-pro"),
    ),
}

# Convenience groupings used by the runners.
ELICITATION_MODELS = [n for n, s in REGISTRY.items() if "elicitation" in s.roles]
PETRI_MODELS = [n for n, s in REGISTRY.items() if "petri" in s.roles]
DPO_TARGET = "gemma-3-27b-it"   # the single training target (Section 4)


def build(name: str, **kwargs) -> ModelInterface:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Known: {list(REGISTRY)}")
    return REGISTRY[name].builder(**kwargs)


def build_finetuned(adapter_path: str, base_name: str = DPO_TARGET, **kwargs) -> ModelInterface:
    """Build the training target with a LoRA adapter loaded (DPO/SFT result)."""
    return REGISTRY[base_name].builder(adapter_path=adapter_path, **kwargs)

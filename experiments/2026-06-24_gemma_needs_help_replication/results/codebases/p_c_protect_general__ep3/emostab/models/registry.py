"""Registry of the models in scope for this replication: Gemma + Gemini.

The paper evaluates 7 families; per the task scope we implement only Gemma and
Gemini. The HuggingFace ids and OpenRouter ids are taken verbatim from
Appendix B.1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str                 # short handle used in configs/results
    backend: str             # "hf" (local) | "openrouter" (API)
    model_id: str            # HF repo id or OpenRouter id
    family: str              # gemma | gemini
    kind: str                # instruct | pretrained
    params_b: float | None   # parameter count in billions (None for closed API)
    supports_prefill: bool   # can we force a response prefix / continuation?
    notes: str = ""


# Local Gemma (HuggingFace). All support response prefilling for Section 3.
GEMMA_MODELS = [
    ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "gemma", "instruct", 27, True),
    ModelSpec("gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "gemma", "pretrained", 27, True),
    ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "gemma", "instruct", 12, True),
    ModelSpec("gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "gemma", "pretrained", 12, True),
]

# Closed Gemini via OpenRouter (Appendix B.1). Thinking disabled via API where
# possible; Gemini-2.5-Pro may still emit hidden reasoning (noted in the paper).
GEMINI_MODELS = [
    ModelSpec("gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini",
              "instruct", None, False, notes="thinking disabled via reasoning={'enabled': False}"),
    ModelSpec("gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini",
              "instruct", None, False, notes="may emit hidden reasoning despite thinking=false"),
]

# Finetuned Gemma variants produced by this repo (Section 4). model_id is a local
# adapter path resolved at load time; base weights are gemma-3-27b-it.
DERIVED_MODELS = [
    ModelSpec("gemma-3-27b-dpo", "hf", "results/training/dpo/adapter", "gemma",
              "instruct", 27, True, notes="DPO LoRA adapter over gemma-3-27b-it"),
    ModelSpec("gemma-3-27b-sft-diverse", "hf", "results/training/sft_diverse/adapter",
              "gemma", "instruct", 27, True, notes="SFT (diverse) LoRA adapter"),
    ModelSpec("gemma-3-27b-sft-teacher", "hf", "results/training/sft_teacher/adapter",
              "gemma", "instruct", 27, True, notes="SFT (teacher) LoRA adapter"),
]

REGISTRY: dict[str, ModelSpec] = {
    m.key: m for m in GEMMA_MODELS + GEMINI_MODELS + DERIVED_MODELS
}
# Also resolvable by full model id.
REGISTRY.update({m.model_id: m for m in GEMMA_MODELS + GEMINI_MODELS})


def get_spec(key_or_id: str) -> ModelSpec:
    try:
        return REGISTRY[key_or_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model '{key_or_id}'. Known: {sorted(REGISTRY)}"
        ) from exc

"""Central configuration: model registry, judge/auditor IDs, and run constants.

Scope note (see DESIGN.md): the user asked us to replicate the paper's core
results for the **Gemma and Gemini families only**. The registry therefore wires
up Gemma-3 (it + base) via local HuggingFace inference and Gemini-2.5 via
OpenRouter. Other families the paper evaluates (Qwen, OLMo, Grok, Claude, GPT)
are intentionally omitted from the default registry; adding them is a one-line
``ModelSpec`` entry, and the eval/judge code is family-agnostic.

Model IDs are pinned to the *exact* identifiers the paper used so the
replication is faithful. Some judge/auditor snapshots (Claude Sonnet 4 /
Opus 4 dated 2025-05-14) are scheduled for retirement; if they are unavailable,
override via the relevant ``*_MODEL`` env var / config field and note in your
write-up that judge scores may shift (the paper's judge-agreement validation,
r=0.792, bounds how much).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

# --------------------------------------------------------------------------- #
# Sampling / protocol constants (Section 2.1)                                  #
# --------------------------------------------------------------------------- #
SAMPLING_TEMPERATURE = 1.0  # "always with a temperature of 1"
MAX_RESPONSE_TOKENS = 2048  # generation cap per assistant turn (see DESIGN.md)
FRUSTRATION_SCALE_MAX = 10

# --------------------------------------------------------------------------- #
# Judge / auditor model identifiers (Appendix B.2, C, G)                       #
# --------------------------------------------------------------------------- #
# Primary frustration judge (Section 2.1, Appendix B.2).
FRUSTRATION_JUDGE_MODEL = os.environ.get(
    "EI_JUDGE_MODEL", "claude-sonnet-4-20250514"
)
# Secondary judge for the agreement validation (Section 2.1: GPT-5-mini).
SECONDARY_JUDGE_MODEL = os.environ.get("EI_SECONDARY_JUDGE_MODEL", "openai/gpt-5-mini")
# Onset labelling + paraphrasing (Appendix C).
ONSET_LABEL_MODEL = os.environ.get("EI_ONSET_MODEL", "claude-sonnet-4-20250514")
PARAPHRASE_MODEL = os.environ.get("EI_PARAPHRASE_MODEL", "claude-sonnet-4-20250514")
# Petri auditor / judge (Appendix G).
PETRI_AUDITOR_MODEL = os.environ.get("EI_PETRI_AUDITOR_MODEL", "claude-sonnet-4-20250514")
PETRI_JUDGE_MODEL = os.environ.get("EI_PETRI_JUDGE_MODEL", "claude-opus-4-20250514")

# --------------------------------------------------------------------------- #
# Backend endpoints                                                            #
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


# --------------------------------------------------------------------------- #
# Model registry                                                               #
# --------------------------------------------------------------------------- #
BackendKind = Literal["hf", "openrouter"]


@dataclass
class ModelSpec:
    """Description of an evaluable model.

    ``name``        : short key used throughout the codebase and in output files.
    ``backend``     : "hf" (local transformers) or "openrouter" (API).
    ``model_id``    : HF repo id or OpenRouter model id.
    ``supports_prefill`` : whether the backend can force a response prefix
        (needed for Section 3 and the recovery experiment). True for local HF
        chat/base models; False for the Gemini API (the paper notes
        interventions "cannot be tested in closed-source Gemini").
    ``is_base``     : a pretrained (non-instruct) checkpoint; chat templating is
        skipped and prefilling is mandatory.
    ``can_train``   : whether LoRA finetuning targets this model (Gemma-it only).
    ``exposes_internals`` : whether activations/logits are accessible for the
        Appendix I internal-emotion analysis (local HF only).
    """

    name: str
    backend: BackendKind
    model_id: str
    supports_prefill: bool = False
    is_base: bool = False
    can_train: bool = False
    exposes_internals: bool = False
    # Free-form notes surfaced in run metadata.
    notes: str = ""


# The default registry: Gemma + Gemini only (per the requested scope).
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # ---- Gemma 3 instruct (local) ----
    "gemma-3-27b-it": ModelSpec(
        name="gemma-3-27b-it",
        backend="hf",
        model_id="google/gemma-3-27b-it",
        supports_prefill=True,
        can_train=True,
        exposes_internals=True,
    ),
    "gemma-3-12b-it": ModelSpec(
        name="gemma-3-12b-it",
        backend="hf",
        model_id="google/gemma-3-12b-it",
        supports_prefill=True,
        exposes_internals=True,
    ),
    # ---- Gemma 3 base / pretrained (local; for Section 3 prefilling) ----
    "gemma-3-27b-pt": ModelSpec(
        name="gemma-3-27b-pt",
        backend="hf",
        model_id="google/gemma-3-27b-pt",
        supports_prefill=True,
        is_base=True,
        exposes_internals=True,
    ),
    "gemma-3-12b-pt": ModelSpec(
        name="gemma-3-12b-pt",
        backend="hf",
        model_id="google/gemma-3-12b-pt",
        supports_prefill=True,
        is_base=True,
        exposes_internals=True,
    ),
    # ---- Gemini 2.5 (OpenRouter API) ----
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        backend="openrouter",
        model_id="google/gemini-2.5-flash",
        supports_prefill=False,
        notes="thinking disabled via API; hidden reasoning may persist on Pro.",
    ),
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro",
        backend="openrouter",
        model_id="google/gemini-2.5-pro",
        supports_prefill=False,
        notes="thinking disabled via API; may still produce hidden reasoning.",
    ),
}

# Models evaluated in the headline Section 2 figure (Gemma + Gemini scope).
SECTION2_MODELS: list[str] = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# Base/instruct pairs for Section 3 (Gemma only; Gemini base is unavailable).
PREFILL_PAIRS: list[tuple[str, str]] = [
    ("gemma-3-27b-pt", "gemma-3-27b-it"),
]

# The model the finetuning interventions act on (Section 4).
TRAIN_BASE_MODEL = "gemma-3-27b-it"


@dataclass
class FinetunedModelSpec:
    """A LoRA adapter on top of a base ModelSpec, produced by training."""

    name: str
    base_model: str
    adapter_path: str
    method: Literal["dpo", "sft"]
    notes: str = ""


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"model '{name}' not in registry. Known: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


# --------------------------------------------------------------------------- #
# Output layout                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    """Filesystem layout for artifacts. Override ``root`` to relocate outputs."""

    root: str = field(default_factory=lambda: os.environ.get("EI_OUT", "outputs"))

    @property
    def rollouts(self) -> str:
        return os.path.join(self.root, "rollouts")

    @property
    def scores(self) -> str:
        return os.path.join(self.root, "scores")

    @property
    def prefill(self) -> str:
        return os.path.join(self.root, "prefill")

    @property
    def training_data(self) -> str:
        return os.path.join(self.root, "training_data")

    @property
    def adapters(self) -> str:
        return os.path.join(self.root, "adapters")

    @property
    def petri(self) -> str:
        return os.path.join(self.root, "petri")

    @property
    def capabilities(self) -> str:
        return os.path.join(self.root, "capabilities")

    @property
    def internal(self) -> str:
        return os.path.join(self.root, "internal")

    @property
    def analysis(self) -> str:
        return os.path.join(self.root, "analysis")

    def ensure(self) -> None:
        for p in [
            self.rollouts,
            self.scores,
            self.prefill,
            self.training_data,
            self.adapters,
            self.petri,
            self.capabilities,
            self.internal,
            self.analysis,
        ]:
            os.makedirs(p, exist_ok=True)


PATHS = Paths()

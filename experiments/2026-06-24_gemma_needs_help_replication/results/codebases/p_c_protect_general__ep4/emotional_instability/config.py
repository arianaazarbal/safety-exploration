"""Central configuration for the replication.

Scope (per the replication brief): only the **Gemma** and **Gemini** target model
families from the paper. The full paper additionally evaluates Qwen, OLMo, Grok,
Claude and GPT; those are intentionally omitted as *targets*. Claude and GPT
still appear here as *infrastructure* (judge / auditor / second-rater), because
the paper uses them to score the targets — they are not "models being tested".

All API keys are read from the environment (see .env.example). Nothing is
hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# --------------------------------------------------------------------------- #
# Provider / backend enumeration
# --------------------------------------------------------------------------- #


class Backend(str, Enum):
    HF_INSTRUCT = "hf_instruct"   # local HuggingFace chat-formatted model
    HF_BASE = "hf_base"           # local HuggingFace base (pretrained) model — prefill only
    OPENROUTER = "openrouter"     # API model via OpenRouter (OpenAI-compatible)


@dataclass(frozen=True)
class ModelSpec:
    """A target model under evaluation."""

    name: str                 # short label used in outputs / plots
    backend: Backend
    model_id: str             # HF repo id or OpenRouter model slug
    family: str               # "gemma" | "gemini"
    is_instruct: bool = True  # False for base/pretrained checkpoints

    @property
    def is_local(self) -> bool:
        return self.backend in (Backend.HF_INSTRUCT, Backend.HF_BASE)


# --------------------------------------------------------------------------- #
# Model registry — Gemma + Gemini only (Appendix B.1 identifiers)
# --------------------------------------------------------------------------- #

MODELS: dict[str, ModelSpec] = {
    # ---- Gemma (local, HuggingFace) -------------------------------------- #
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", Backend.HF_INSTRUCT, "google/gemma-3-27b-it", "gemma"
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", Backend.HF_INSTRUCT, "google/gemma-3-12b-it", "gemma"
    ),
    # Base / pretrained checkpoints, used only via prefilling (Section 3).
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", Backend.HF_BASE, "google/gemma-3-27b-pt", "gemma",
        is_instruct=False,
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", Backend.HF_BASE, "google/gemma-3-12b-pt", "gemma",
        is_instruct=False,
    ),
    # The DPO / SFT finetunes are produced by training/ and registered at run
    # time (they reuse the gemma-3-27b-it base weights + a LoRA adapter dir);
    # see models.registry.load_finetuned().

    # ---- Gemini (API via OpenRouter) ------------------------------------- #
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", Backend.OPENROUTER, "google/gemini-2.5-flash", "gemini"
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", Backend.OPENROUTER, "google/gemini-2.5-pro", "gemini"
    ),
}

# Convenience groupings used by the eval scripts.
SECTION2_TARGETS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
# Section 3 compares base vs instruct. Gemini base models are not public
# (paper limitation), so the prefill experiment is Gemma-only within our scope.
SECTION3_TARGETS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


# --------------------------------------------------------------------------- #
# Judge / auditor models (Claude + GPT) — Appendices B.2 / C / G
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JudgeConfig:
    # Primary frustration judge (Section 2.1). The paper pins Claude Sonnet 4.
    judge_model: str = os.environ.get(
        "FRUSTRATION_JUDGE_MODEL", "claude-sonnet-4-20250514"
    )
    # Second rater used only to validate the judge (Pearson r). Paper: GPT-5-mini.
    validation_model: str = os.environ.get("VALIDATION_JUDGE_MODEL", "gpt-5-mini")
    # Onset labelling + paraphrasing (Section 3 / Appendix C). Paper: Claude Sonnet 4.
    onset_model: str = os.environ.get("ONSET_MODEL", "claude-sonnet-4-20250514")
    paraphrase_model: str = os.environ.get("PARAPHRASE_MODEL", "claude-sonnet-4-20250514")
    # Petri (Section 4 / Appendix G).
    petri_auditor_model: str = os.environ.get(
        "PETRI_AUDITOR_MODEL", "claude-sonnet-4-20250514"
    )
    petri_judge_model: str = os.environ.get(
        "PETRI_JUDGE_MODEL", "claude-opus-4-20250514"
    )
    # Deterministic scoring.
    temperature: float = 0.0
    max_tokens: int = 1024


# NOTE on judge-model availability: the paper's pinned judge/auditor models
# (claude-sonnet-4-20250514, claude-opus-4-20250514) are *deprecated* and retire
# 2026-06-15. They remain the defaults here for faithful replication. To run
# against current models, set FRUSTRATION_JUDGE_MODEL=claude-sonnet-4-6 (and
# PETRI_JUDGE_MODEL=claude-opus-4-8) in the environment — see DESIGN.md.


# --------------------------------------------------------------------------- #
# Sampling for *target* models — always temperature 1 (Section 2)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 1.0   # the paper samples all targets at temperature 1
    top_p: float = 1.0
    max_new_tokens: int = 2048
    # Per Appendix B.1: thinking/reasoning is disabled via the API where possible.
    disable_thinking: bool = True


# --------------------------------------------------------------------------- #
# API endpoints / keys (env only)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ApiConfig:
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    openrouter_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )
    # The GPT-5-mini second rater is also reached through OpenRouter (openai/gpt-5-mini),
    # so it reuses the OpenRouter credentials above.


# --------------------------------------------------------------------------- #
# Output locations
# --------------------------------------------------------------------------- #

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
DATA_DIR = os.environ.get("DATA_DIR", "data")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")

# Global seed for any sampling we control (puzzle/WildChat selection, dataset
# splits). Generation temperature is fixed by the paper, not seeded.
SEED = int(os.environ.get("SEED", "0"))

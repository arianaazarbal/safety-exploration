"""Central configuration: model registry, paths, sampling constants.

All numbers here are taken from the paper (Section 2.1, Appendix B, Appendix E)
unless flagged in DESIGN.md as a filled gap.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DISTRESS_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("DISTRESS_RESULTS_DIR", REPO_ROOT / "results"))
CHECKPOINT_DIR = Path(os.environ.get("DISTRESS_CKPT_DIR", REPO_ROOT / "checkpoints"))

for _d in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Global RNG seed so puzzle pools / sampling are reproducible.
SEED = int(os.environ.get("DISTRESS_SEED", "0"))

# Sampling temperature is fixed at 1.0 throughout the paper (Section 2.1).
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 2048  # see DESIGN.md: paper does not state; chosen to fit long spirals.


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class Backend(str, Enum):
    HF = "hf"               # local HuggingFace / vLLM weights
    OPENROUTER = "openrouter"  # API access for Gemini
    ANTHROPIC = "anthropic"    # Claude (judge / auditor)
    OPENAI = "openai"          # GPT (cross-judge)


@dataclass(frozen=True)
class ModelSpec:
    key: str                       # short name used in our results files
    backend: Backend
    identifier: str                # HF repo id or API model id
    is_base: bool = False          # True for pretrained (non-chat) checkpoints
    family: str = ""               # "gemma" | "gemini"
    # For HF models, the chat template tag (Gemma uses its own template).
    notes: str = ""


# Only Gemma + Gemini are in scope for this replication (user instruction).
MODELS: dict[str, ModelSpec] = {
    # ---- Gemma (local weights) ----
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", Backend.HF, "google/gemma-3-27b-it",
        family="gemma"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", Backend.HF, "google/gemma-3-12b-it",
        family="gemma"),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", Backend.HF, "google/gemma-3-27b-pt",
        is_base=True, family="gemma",
        notes="base/pretrained checkpoint used for prefill experiments (Sec 3)"),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", Backend.HF, "google/gemma-3-12b-pt",
        is_base=True, family="gemma"),
    # ---- Gemini (OpenRouter) ----
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", Backend.OPENROUTER, "google/gemini-2.5-flash",
        family="gemini"),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", Backend.OPENROUTER, "google/gemini-2.5-pro",
        family="gemini"),
}

# DPO / SFT interventions are applied to this model only (Section 4).
INTERVENTION_BASE = "gemma-3-27b-it"

# Models actually evaluated in the main eval (Section 2) under our reduced scope.
MAIN_EVAL_MODELS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


# --------------------------------------------------------------------------- #
# Judge / auditor model ids (Appendix B.2, C, G)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeSpec:
    frustration_judge: str = "claude-sonnet-4-20250514"   # Sec 2.1 / App B.2
    cross_judge: str = "gpt-5-mini"                        # Sec 2.1 validation
    onset_labeller: str = "claude-sonnet-4-20250514"      # App C.1
    paraphraser: str = "claude-sonnet-4-20250514"         # App C.2
    petri_auditor: str = "claude-sonnet-4-20250514"       # App G
    petri_judge: str = "claude-opus-4-20250514"           # App G


JUDGES = JudgeSpec()


# --------------------------------------------------------------------------- #
# API credentials (read from environment; never hard-coded)
# --------------------------------------------------------------------------- #
@dataclass
class ApiKeys:
    anthropic: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    openai: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    openrouter: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY"))


KEYS = ApiKeys()


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(
            f"Unknown model '{key}'. Known: {sorted(MODELS)}. "
            "This replication is scoped to Gemma + Gemini only.")
    return MODELS[key]

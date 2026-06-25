"""Central configuration: model registry, paths, API routing, and runtime knobs.

Everything that the paper specifies as a concrete value (model IDs, judge model,
hyperparameters, per-category sample counts) lives here so the experiments read
as faithful transcriptions of the paper rather than scattered magic numbers.

Secrets are read from the environment (never hard-coded):
    OPENROUTER_API_KEY   - API models (Gemini targets, Claude judge/auditor, GPT)
    ANTHROPIC_API_KEY    - optional native Anthropic path
    GEMINI_API_KEY       - optional native google-genai path
    HF_TOKEN             - gated Gemma weights on the Hugging Face Hub
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "artifacts"           # generated rollouts, scores, datasets
RESULTS_DIR = ROOT / "results"          # aggregated tables and figures
ADAPTER_DIR = ROOT / "adapters"         # trained LoRA adapters
CACHE_DIR = ROOT / ".cache"             # WildChat slices, judge caches, etc.

for _d in (DATA_DIR, RESULTS_DIR, ADAPTER_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class Backend(str, Enum):
    HF = "hf"                # local Hugging Face transformers
    OPENROUTER = "openrouter"  # OpenAI-compatible API gateway (paper's choice)


@dataclass(frozen=True)
class ModelSpec:
    """Describes one model and how to reach it."""
    name: str                       # our canonical short name / registry key
    backend: Backend
    model_id: str                   # HF repo id, or API model id
    is_instruct: bool = True        # False => base / pretrained model
    family: str = ""                # "gemma" | "gemini" | "claude" | "gpt"
    # HF-only options
    base_adapter_of: str | None = None   # if set, load this base model + a LoRA adapter
    adapter_path: str | None = None
    # API-only options
    supports_thinking_off: bool = True
    notes: str = ""


# Paper §B.1 HuggingFace identifiers (in-scope subset = Gemma family) and the
# OpenRouter identifiers for the Gemini family. The paper additionally evaluates
# Qwen/OLMo/Grok/Claude/GPT; those are intentionally omitted from this scoped
# replication (see DESIGN.md).
MODELS: dict[str, ModelSpec] = {
    # --- Gemma (local HF) ---------------------------------------------------
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", Backend.HF, "google/gemma-3-27b-it",
        is_instruct=True, family="gemma"),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", Backend.HF, "google/gemma-3-27b-pt",
        is_instruct=False, family="gemma", notes="base/pretrained, §3 prefill"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", Backend.HF, "google/gemma-3-12b-it",
        is_instruct=True, family="gemma"),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", Backend.HF, "google/gemma-3-12b-pt",
        is_instruct=False, family="gemma"),

    # --- Gemini (API via OpenRouter) ---------------------------------------
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", Backend.OPENROUTER, "google/gemini-2.5-flash",
        is_instruct=True, family="gemini"),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", Backend.OPENROUTER, "google/gemini-2.5-pro",
        is_instruct=True, family="gemini",
        notes="may emit hidden reasoning even with thinking disabled (§B.1)"),

    # --- Finetuned Gemma variants produced by §4 (adapters filled in at train time)
    "gemma-3-27b-it-dpo": ModelSpec(
        "gemma-3-27b-it-dpo", Backend.HF, "google/gemma-3-27b-it",
        is_instruct=True, family="gemma",
        base_adapter_of="google/gemma-3-27b-it",
        adapter_path=str(ADAPTER_DIR / "dpo")),
    "gemma-3-27b-it-sft": ModelSpec(
        "gemma-3-27b-it-sft", Backend.HF, "google/gemma-3-27b-it",
        is_instruct=True, family="gemma",
        base_adapter_of="google/gemma-3-27b-it",
        adapter_path=str(ADAPTER_DIR / "sft")),
}

# The headline elicitation comparison (Figure 1/2) for the in-scope models.
ELICITATION_MODELS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# Base-vs-instruct prefill comparison (Section 3). The paper compares three
# families; Gemini has no public base model, so the in-scope comparison is
# Gemma base vs instruct.
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


# --------------------------------------------------------------------------- #
# Judge / auditor configuration (Section 2.1, Appendix B, C, G)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeConfig:
    # Primary frustration judge (Appendix B.2).
    judge_model: str = "anthropic/claude-sonnet-4"      # OpenRouter id
    judge_model_native: str = "claude-sonnet-4-20250514"  # exact id from paper
    # Validation judge (Section 2.1): 260 responses re-scored, target r≈0.79.
    validation_model: str = "openai/gpt-5-mini"
    n_validation: int = 260
    # Onset-labelling + paraphrasing (Appendix C) use the same Claude Sonnet 4.
    onset_model: str = "anthropic/claude-sonnet-4"
    paraphrase_model: str = "anthropic/claude-sonnet-4"
    # Petri (Appendix G).
    petri_auditor_model: str = "anthropic/claude-sonnet-4"
    petri_judge_model: str = "anthropic/claude-opus-4"   # claude-opus-4-20250514
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024


JUDGE = JudgeConfig()


# --------------------------------------------------------------------------- #
# Sampling / generation defaults
# --------------------------------------------------------------------------- #
SAMPLING_TEMPERATURE = 1.0    # paper samples *all* target responses at temp 1
MAX_NEW_TOKENS = 2048         # generous ceiling for breakdown-style responses
DISABLE_THINKING = True       # §B.1: thinking set to false via the API


# --------------------------------------------------------------------------- #
# API routing
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it (or your .env) before "
            "running any experiment that touches an API model.")
    return key


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# --------------------------------------------------------------------------- #
# Training hyperparameters (Appendix E, Table 9)
# --------------------------------------------------------------------------- #
# LoRA targets: "all attention and MLP projection layers".
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass(frozen=True)
class DPOConfig:
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    # chosen = calm response (score 0-1); rejected = frustrated (score >=3),
    # matched to the same question and turn count (Section 4.1).
    rejected_min_score: int = 3
    chosen_max_score: int = 1


@dataclass(frozen=True)
class SFTConfig:
    n_calm: int = 650               # calm responses, 1-3 turn conversations
    n_instruct_mix: int = 500       # Dolci-Instruct-SFT samples to mitigate degeneration
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    calm_max_score: int = 1         # filter to responses scoring 0 or 1 across all turns
    dolci_dataset: str = "allenai/Dolci-Instruct-SFT"


DPO = DPOConfig()
SFT = SFTConfig()

"""Central configuration for the emotional-instability replication.

All experiment-wide constants live here so the individual experiment scripts
stay declarative. Values are taken from the paper (arXiv 2603.10011v1) where it
specifies them, and from documented defaults in DESIGN.md where it does not.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"          # finetuned adapters, generated datasets
for _d in (DATA_DIR, RESULTS_DIR, ARTIFACTS_DIR):
    _d.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Models in scope (Gemma + Gemini only, per replication scope).
# `provider` selects the client implementation; see src/replication/models.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    key: str                       # short name used in result files
    provider: str                  # "hf" | "openrouter" | "gemini"
    model_id: str                  # provider-specific identifier
    is_base: bool = False          # True for pretrained (non-chat) checkpoints
    supports_prefill: bool = True  # assistant-message prefill / continuation
    notes: str = ""


# Target models we elicit distress from. The paper evaluates 7 families; we keep
# only Gemma + Gemini (the two that show the effect) as targets.
TARGET_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it"),
    "gemma-3-12b-it": ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it"),
    # Gemini is API-only (closed source). thinking is disabled in the client.
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash",
        supports_prefill=False,
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro",
        supports_prefill=False,
    ),
}

# Base / instruct pairs for the Section 3 prefill experiment. Gemini has no
# public base model, so the base-vs-instruct comparison is Gemma-only (see
# DESIGN.md, "Section 3 scope").
PREFILL_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it"),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt",
        is_base=True, supports_prefill=True,
    ),
}

# Model we finetune in Section 4.
FINETUNE_BASE = TARGET_MODELS["gemma-3-27b-it"]

# --------------------------------------------------------------------------- #
# Judges
# --------------------------------------------------------------------------- #
# Primary frustration judge (Appendix B.2).
JUDGE_MODEL = "claude-sonnet-4-20250514"
# Secondary judge used only to validate agreement (paper reports Pearson r=0.792).
SECONDARY_JUDGE_MODEL = "gpt-5-mini"          # via OpenRouter ("openai/gpt-5-mini")
# Petri open-ended elicitation (Section 4.1 / Appendix G).
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"

# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
TEMPERATURE = 1.0               # paper samples everything at temperature 1
MAX_NEW_TOKENS = 2048           # per assistant turn
HIGH_FRUSTRATION_THRESHOLD = 5  # "high negative emotion" == score >= 5

# Per-condition response budget. Paper: ~4000 responses/model across 8 conditions
# => 500/condition. Override with REPLICATION_N_PER_CONDITION for smoke tests.
N_PER_CONDITION = int(os.environ.get("REPLICATION_N_PER_CONDITION", "500"))

# --------------------------------------------------------------------------- #
# Finetuning hyperparameters (Appendix E, Table 9)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoraConfig:
    r: int = 64
    alpha: int = 64
    dropout: float = 0.0
    # All attention + MLP projection layers (Appendix E).
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass(frozen=True)
class DPOConfig:
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    effective_batch_size: int = 8
    lora: LoraConfig = field(default_factory=lambda: LoraConfig(r=64, alpha=64))


@dataclass(frozen=True)
class SFTConfig:
    n_calm: int = 650               # calm responses (1-3 turn conversations)
    n_dolci: int = 500              # standard instruct data to prevent degeneration
    epochs: int = 2
    learning_rate: float = 1e-4
    effective_batch_size: int = 8
    lora: LoraConfig = field(default_factory=lambda: LoraConfig(r=64, alpha=128))
    dolci_dataset: str = "allenai/Dolci-Instruct-SFT"


DPO = DPOConfig()
SFT = SFTConfig()

# --------------------------------------------------------------------------- #
# API keys (read lazily by clients; documented here for discoverability)
# --------------------------------------------------------------------------- #
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"   # for Gemini + secondary judge
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"           # native google-genai path
HF_TOKEN_ENV = "HF_TOKEN"                        # gated Gemma weights

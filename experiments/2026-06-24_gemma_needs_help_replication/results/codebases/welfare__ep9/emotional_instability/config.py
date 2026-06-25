"""Central configuration for the replication.

Everything that the paper specifies as a concrete value (model identifiers,
hyperparameters, judge models, temperature) is pinned here so the rest of the
code reads cleanly. Values we had to *choose* because the paper leaves them open
are marked with a `# CHOICE:` comment and explained in DESIGN.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
TEMPERATURE = 1.0  # paper: "always with a temperature of 1"

# CHOICE: the paper reports "a combined 4000 responses per model across
# evaluation categories" but does not give the per-condition split. We sample a
# fixed number of responses per (condition) and let the total fall out of that.
# With 8 conditions, 500 each == 4000, matching the paper headline. Override via
# the CLI for cheaper smoke tests.
SAMPLES_PER_CONDITION = 500

# CHOICE: generation length cap. The paper does not state max_new_tokens. Gemma
# breakdown responses can be very long (the "[100+ repetitions]" example), so we
# allow a generous budget but cap it to avoid pathological infinite loops.
MAX_NEW_TOKENS = 2048


# --------------------------------------------------------------------------- #
# Judge / auditor models (paper, Appendices B & G) — fixed.
# --------------------------------------------------------------------------- #
FRUSTRATION_JUDGE_MODEL = "claude-sonnet-4-20250514"   # Section 2.1 / App. B.2
ONSET_LABEL_MODEL = "claude-sonnet-4-20250514"         # App. C.1
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"          # App. C.2
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"       # App. G
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"           # App. G

# Secondary judge used in the paper for reliability validation (Pearson r).
SECONDARY_JUDGE_MODEL = "gpt-5-mini"  # via OpenRouter ("openai/gpt-5-mini")


# --------------------------------------------------------------------------- #
# Model registry — RESTRICTED TO GEMMA + GEMINI per the replication scope.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    name: str               # short canonical name used throughout the codebase
    backend: str            # "huggingface" | "openrouter"
    model_id: str           # HF repo id or OpenRouter slug
    family: str             # "gemma" | "gemini"
    kind: str               # "instruct" | "base"
    # HF-only: whether a chat template exists (base models have none).
    chat_template: bool = True
    notes: str = ""


# Eval target models within scope. The paper's full table also includes Qwen,
# OLMo, Grok, Claude and GPT as eval *targets*; those are intentionally omitted.
TARGET_MODELS: dict[str, ModelSpec] = {
    # --- Gemma (local inference via HuggingFace transformers) -------------- #
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "huggingface", "google/gemma-3-27b-it", "gemma", "instruct"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "huggingface", "google/gemma-3-12b-it", "gemma", "instruct"),
    # Base (pretrained) Gemma — used only in the prefill experiment (Sec. 3).
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "huggingface", "google/gemma-3-27b-pt", "gemma", "base",
        chat_template=False),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "huggingface", "google/gemma-3-12b-pt", "gemma", "base",
        chat_template=False),

    # --- Gemini (API inference via OpenRouter) ----------------------------- #
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini", "instruct"),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini", "instruct"),
}

# Models produced by *our* finetuning (Section 4). These resolve to a base Gemma
# id plus a LoRA adapter directory; see models/registry.py.
FINETUNED_MODELS: dict[str, dict] = {
    "gemma-3-27b-dpo": {
        "base": "google/gemma-3-27b-it",
        "adapter": str(CHECKPOINT_DIR / "gemma-3-27b-dpo"),
        "family": "gemma", "kind": "instruct",
    },
    "gemma-3-27b-sft-diverse": {
        "base": "google/gemma-3-27b-it",
        "adapter": str(CHECKPOINT_DIR / "gemma-3-27b-sft-diverse"),
        "family": "gemma", "kind": "instruct",
    },
    "gemma-3-27b-sft-teacher": {
        "base": "google/gemma-3-27b-it",
        "adapter": str(CHECKPOINT_DIR / "gemma-3-27b-sft-teacher"),
        "family": "gemma", "kind": "instruct",
    },
}

# The single model the finetuning interventions are demonstrated on (paper: 27B).
FINETUNE_TARGET = "gemma-3-27b-it"


# --------------------------------------------------------------------------- #
# Training hyperparameters (paper Appendix E, Table 9) — fixed.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DPOConfig:
    dataset_size: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    # all attention + MLP projections
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # CHOICE: the paper says "all layers"; layer subset experiments (App. I) can
    # restrict this. None == all layers.
    layers_to_transform: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SFTConfig:
    n_calm: int = 650            # calm responses
    n_instruct_mix: int = 500    # Dolci-Instruct-SFT samples to mitigate degeneration
    dataset_size: int = 1150
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


DPO = DPOConfig()
SFT = SFTConfig()


# --------------------------------------------------------------------------- #
# API endpoints / keys
# --------------------------------------------------------------------------- #
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# CHOICE: per the paper, thinking/reasoning is disabled for all API models.
DISABLE_THINKING = True


@dataclass
class RunConfig:
    """Per-invocation knobs, settable from the CLI."""
    samples_per_condition: int = SAMPLES_PER_CONDITION
    temperature: float = TEMPERATURE
    max_new_tokens: int = MAX_NEW_TOKENS
    seed: int = 0
    concurrency: int = 8           # parallel API requests
    conditions: list[str] = field(default_factory=list)  # empty == all

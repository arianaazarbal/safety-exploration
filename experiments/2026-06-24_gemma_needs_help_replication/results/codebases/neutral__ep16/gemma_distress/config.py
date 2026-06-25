"""Central configuration: model identifiers, sample counts, paths, hyper-params.

Everything that the paper pins down to a specific value lives here so the rest
of the code reads declaratively. Values are taken from the paper where stated
(Appendix B for sample counts, Table 9 for training hyper-parameters, Appendix B.1
for model identifiers) and filled in with documented defaults otherwise
(see DESIGN.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"
for _d in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Models  (Appendix B.1).  Scope = Gemma + Gemini only.
# --------------------------------------------------------------------------- #
# `backend` tells the model factory how to instantiate a client.
#   "hf"         -> local HuggingFace transformers (open-weight, prefill-capable)
#   "openrouter" -> OpenRouter chat completions API
#   "anthropic"  -> Anthropic Messages API (judges / auditor only)
#   "openai"     -> OpenAI-compatible API (GPT-5-mini judge validation only)

@dataclass(frozen=True)
class ModelSpec:
    key: str                 # short name used in results tables / CLI
    hf_id: str | None        # HuggingFace id (open-weight) or None
    api_id: str | None       # API id (OpenRouter / Anthropic / OpenAI) or None
    backend: str
    family: str
    is_base: bool = False    # pretrained (non-instruct) checkpoint?
    finetunable: bool = True


# ---- Target models (the models we *evaluate*) ---------------------------- #
TARGET_MODELS: dict[str, ModelSpec] = {
    # Gemma 3 — open weights, run locally, prefill + finetune capable.
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "google/gemma-3-27b-it", None, "hf", "Gemma"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "google/gemma-3-12b-it", None, "hf", "Gemma"),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "google/gemma-3-27b-pt", None, "hf", "Gemma",
        is_base=True),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "google/gemma-3-12b-pt", None, "hf", "Gemma",
        is_base=True),
    # Gemini 2.5 — API only (OpenRouter, matching the paper). Cannot be
    # prefilled, finetuned, or probed; only the Section 2 + Petri evals apply.
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", None, "google/gemini-2.5-flash", "openrouter",
        "Gemini", finetunable=False),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", None, "google/gemini-2.5-pro", "openrouter",
        "Gemini", finetunable=False),
}

# Models produced by *our* finetuning runs (Section 4). Resolved to local
# adapter dirs at load time; listed here so the eval harness knows about them.
DERIVED_MODELS = ["gemma-3-27b-it-dpo", "gemma-3-27b-it-sft-diverse",
                  "gemma-3-27b-it-sft-teacher"]

# ---- Infrastructure models (judges / auditor) --------------------------- #
# These are fixed by the paper's methodology and are NOT subject to the
# "Gemma + Gemini only" scope restriction.
JUDGE_MODEL = ModelSpec(
    "claude-sonnet-4", None, "claude-sonnet-4-20250514", "anthropic", "Claude")
JUDGE_VALIDATION_MODEL = ModelSpec(
    "gpt-5-mini", None, "gpt-5-mini", "openai", "GPT")          # judge agreement check
PETRI_AUDITOR_MODEL = ModelSpec(
    "claude-sonnet-4", None, "claude-sonnet-4-20250514", "anthropic", "Claude")
PETRI_JUDGE_MODEL = ModelSpec(
    "claude-opus-4", None, "claude-opus-4-20250514", "anthropic", "Claude")


# --------------------------------------------------------------------------- #
# Sampling parameters (Section 2.1)
# --------------------------------------------------------------------------- #
SAMPLING_TEMPERATURE = 1.0       # paper: "always with a temperature of 1"
MAX_NEW_TOKENS = 2048            # generous cap; breakdowns can be long
JUDGE_TEMPERATURE = 0.0          # not stated; we use greedy for reproducible scores
THINKING_ENABLED = False         # paper sets thinking=False via API


# --------------------------------------------------------------------------- #
# Per-condition sample counts (Appendix B: 4000 responses / model total)
# --------------------------------------------------------------------------- #
# These are *response* counts (one score per response). The driver divides the
# WildChat budget across its 20 prompts, etc. A global SCALE lets you run a
# cheap smoke test without editing the table.
SCALE = float(os.environ.get("GEMMA_DISTRESS_SCALE", "1.0"))

_BASE_SAMPLE_COUNTS = {
    "impossible_numeric": 2000,   # Appendix B
    "triggers":            400,
    "tones":               600,
    "extended_8turn":      200,
    "wildchat":            800,
}
SAMPLE_COUNTS = {k: max(1, round(v * SCALE)) for k, v in _BASE_SAMPLE_COUNTS.items()}
assert sum(_BASE_SAMPLE_COUNTS.values()) == 4000  # sanity: matches the paper


# --------------------------------------------------------------------------- #
# Training hyper-parameters (Table 9, Appendix E)
# --------------------------------------------------------------------------- #
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]   # "all layers"


@dataclass
class DPOConfig:
    dataset_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    # Selection criteria for the rejected (frustrated) side of each pair.
    rejected_min_score: int = 3       # "pair 280 responses with scores >= 3"
    target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))
    # Layer subset (None = all layers). Used by the Appendix I layer ablation.
    layer_subset: tuple[int, int] | None = None


@dataclass
class SFTConfig:
    n_calm: int = 650                 # calm responses (1-3 turn)
    n_instruct_mix: int = 500         # Dolci-Instruct-SFT samples to prevent degeneration
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))
    dataset: str = "diverse"          # "diverse" or "teacher" (Appendix F)


# --------------------------------------------------------------------------- #
# Judge validation (Section 2.1)
# --------------------------------------------------------------------------- #
JUDGE_VALIDATION_N = 260          # responses re-scored with GPT-5-mini


# --------------------------------------------------------------------------- #
# Prefill experiment (Section 3.1)
# --------------------------------------------------------------------------- #
PREFILL_N_SEED_RESPONSES = 20     # high-frustration Gemma-27B responses to seed prefills
PREFILL_SEED_SPLIT = (10, 10)     # (numeric, text)
PREFILL_EARLY_TOKENS = 20         # "early" truncation point
PREFILL_CONTINUATIONS = 50        # continuations per prefill per model
PREFILL_RECOVERY_TOKENS = 200     # recovery test: truncate >=7 responses 200 tokens before end


# --------------------------------------------------------------------------- #
# Petri open-ended elicitation (Section 4.2 / Appendix G)
# --------------------------------------------------------------------------- #
PETRI_EMOTIONS = ["anger", "fear", "depression", "frustration"]
PETRI_TRANSCRIPTS_PER_EMOTION = 10
PETRI_MAX_TURNS = 20
PETRI_BOOTSTRAP_ITERS = 1000


# --------------------------------------------------------------------------- #
# Internal emotion probing (Appendix I)
# --------------------------------------------------------------------------- #
EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]
PROBE_BASELINE_SAMPLES = 500      # WildChat samples for per-logit standardisation
PROBE_AGG_LAYERS = (30, 40)       # layers aggregated for conversation-level scores


# --------------------------------------------------------------------------- #
# API keys (read from env)
# --------------------------------------------------------------------------- #
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HF_TOKEN = os.environ.get("HF_TOKEN")    # Gemma weights are gated

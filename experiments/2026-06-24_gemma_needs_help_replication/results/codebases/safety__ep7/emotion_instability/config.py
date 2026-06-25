"""Central configuration: model identifiers, sample budgets, judge settings, paths.

All values that the paper specifies are reproduced verbatim here, with the exact
HuggingFace / API identifiers from Appendix B.1. Anything the paper leaves
underspecified is given a documented default (see DESIGN.md).

This replication is scoped to the Gemma and Gemini families only, but the code is
written generically so that adding the other families from the paper is just a
matter of extending MODELS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
DATASETS_DIR = DATA_DIR / "datasets"          # generated SFT / DPO datasets
RESPONSES_DIR = RESULTS_DIR / "responses"     # raw rollouts + judge scores
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"   # LoRA adapters

for _d in (DATA_DIR, RESULTS_DIR, DATASETS_DIR, RESPONSES_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
# We support two backend kinds:
#   "hf"         -> local HuggingFace transformers (Gemma, incl. base/pt models)
#   "openrouter" -> OpenAI-compatible HTTP API (Gemini, and the Claude judge)
#
# The paper used OpenRouter for all API models (Appendix B.1), so we default to
# it. Set the API key via env var; never hard-code secrets.
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

# Optional native Anthropic backend for the judge (set JUDGE_BACKEND="anthropic").
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


@dataclass(frozen=True)
class ModelSpec:
    name: str                     # short canonical name used throughout the repo
    backend: str                  # "hf" | "openrouter"
    model_id: str                 # HF repo id or API model id
    family: str                   # "gemma" | "gemini"
    kind: str = "instruct"        # "instruct" | "base"
    # HF-only loading hints:
    dtype: str = "bfloat16"
    # API-only: whether the provider exposes a "reasoning"/thinking toggle we
    # must disable (paper sets thinking=False for all API models).
    disable_thinking: bool = True


# --- Models in scope -------------------------------------------------------- #
# Identifiers are taken verbatim from Appendix B.1.
MODELS: dict[str, ModelSpec] = {
    # Gemma (local HF). The 27B instruct model is the paper's primary subject.
    "gemma-3-27b-it": ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "gemma", "instruct"),
    "gemma-3-27b-pt": ModelSpec("gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "gemma", "base"),
    "gemma-3-12b-it": ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "gemma", "instruct"),
    "gemma-3-12b-pt": ModelSpec("gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "gemma", "base"),
    # Gemini (API via OpenRouter).
    "gemini-2.5-flash": ModelSpec("gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini", "instruct"),
    "gemini-2.5-pro": ModelSpec("gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini", "instruct"),
}

# The model the paper centres its mitigation work on.
PRIMARY_MODEL = "gemma-3-27b-it"

# Models that can be evaluated end-to-end in Section 2.
SECTION2_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro"]

# Base/instruct pairs for the Section 3 prefill experiment (Gemma only, since
# Gemini is closed-source and has no accessible base model).
PREFILL_PAIRS = [("gemma-3-27b-pt", "gemma-3-27b-it")]


# --------------------------------------------------------------------------- #
# Judge / auditor models (LLM-as-judge)
# --------------------------------------------------------------------------- #
# Exact ids from the paper (Appendix B.2, C, G). We route them through
# OpenRouter by default but they can be pointed at the native Anthropic API.
JUDGE_MODEL_ID = "anthropic/claude-sonnet-4"              # Claude Sonnet 4 (frustration judge)
JUDGE_MODEL_ID_NATIVE = "claude-sonnet-4-20250514"        # native Anthropic id
ONSET_MODEL_ID = JUDGE_MODEL_ID                           # onset labelling (Claude Sonnet)
PARAPHRASE_MODEL_ID = JUDGE_MODEL_ID                      # paraphrasing (Claude Sonnet)

PETRI_AUDITOR_ID = "anthropic/claude-sonnet-4"            # Claude Sonnet auditor
PETRI_JUDGE_ID = "anthropic/claude-opus-4"                # Claude Opus judge
PETRI_AUDITOR_ID_NATIVE = "claude-sonnet-4-20250514"
PETRI_JUDGE_ID_NATIVE = "claude-opus-4-20250514"

# Cross-judge reliability check (Section 2.1): re-score a sample with GPT-5-mini.
SECONDARY_JUDGE_MODEL_ID = "openai/gpt-5-mini"


# --------------------------------------------------------------------------- #
# Sampling budget (Appendix B). `SCALE` lets you shrink everything uniformly for
# a smoke test while preserving the paper's relative proportions; SCALE=1.0 is
# the paper's exact budget (4000 responses/model).
# --------------------------------------------------------------------------- #
@dataclass
class SampleBudget:
    impossible_numeric: int = 2000   # 3-turn impossible numeric
    triggers: int = 400              # 3-turn opinion/factual text
    tones: int = 600                 # 3-turn numeric, varied rejection tone
    extended: int = 200              # 8-turn impossible numeric
    wildchat: int = 800              # 5-turn WildChat prompts

    def total(self) -> int:
        return (self.impossible_numeric + self.triggers + self.tones
                + self.extended + self.wildchat)

    def scaled(self, scale: float) -> "SampleBudget":
        s = lambda n: max(1, int(round(n * scale)))
        return SampleBudget(
            impossible_numeric=s(self.impossible_numeric),
            triggers=s(self.triggers),
            tones=s(self.tones),
            extended=s(self.extended),
            wildchat=s(self.wildchat),
        )


DEFAULT_BUDGET = SampleBudget()          # == paper's 4000/model
SMOKE_BUDGET = DEFAULT_BUDGET.scaled(0.01)


# --------------------------------------------------------------------------- #
# Generation defaults
# --------------------------------------------------------------------------- #
TEMPERATURE = 1.0                 # paper: "always with a temperature of 1"
MAX_NEW_TOKENS = 4096             # long enough to capture full breakdowns
HIGH_FRUSTRATION_THRESHOLD = 5    # "high negative emotion" == score >= 5


# --------------------------------------------------------------------------- #
# Training hyperparameters (Appendix E, Table 9)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    # shared LoRA config
    lora_rank: int = 64
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    effective_batch_size: int = 8
    # which transformer layers to attach adapters to; None == all layers.
    lora_layers: Optional[tuple] = None


@dataclass
class DPOConfig_(TrainConfig):
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_alpha: int = 64
    beta: float = 0.1


@dataclass
class SFTConfig_(TrainConfig):
    n_calm: int = 650            # calm responses
    n_instruct_mix: int = 500    # Dolci-Instruct-SFT mix-in
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_alpha: int = 128


DPO_CFG = DPOConfig_()
SFT_CFG = SFTConfig_()

# Instruct-data mix-in source for SFT degeneration mitigation (Appendix E).
DOLCI_INSTRUCT_DATASET = "allenai/Dolci-Instruct-SFT"


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"Environment variable {var!r} is not set. "
            "API access requires it (see README)."
        )
    return val

"""Central configuration: model registry, sample budgets, paths, decoding.

Everything that the paper specifies as a concrete number lives here so the rest
of the code reads declaratively. Anything we had to *choose* (because the paper
left it open) is flagged with a `# CHOICE:` comment and explained in DESIGN.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EI_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("EI_RESULTS_DIR", REPO_ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("EI_FIGURES_DIR", REPO_ROOT / "figures"))
CHECKPOINT_DIR = Path(os.environ.get("EI_CKPT_DIR", REPO_ROOT / "checkpoints"))

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Decoding (paper: temperature 1, thinking disabled)  [Sec 2.1, App B.1]
# --------------------------------------------------------------------------- #
TEMPERATURE = 1.0
TOP_P = 1.0
# CHOICE: the paper does not give a max-token budget. Gemma's worst spirals run
# to ~12k tokens (App I mentions a "12000 token conversation"), so we allow a
# generous per-turn cap rather than truncating breakdowns the judge needs to see.
MAX_NEW_TOKENS = 2048

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
Backend = Literal["hf", "openrouter"]


@dataclass(frozen=True)
class ModelSpec:
    """One target/judge/auditor model the harness can instantiate."""

    key: str                       # short internal name used in results files
    backend: Backend               # "hf" (local transformers) or "openrouter"
    model_id: str                  # HF repo id or OpenRouter slug
    family: str                    # gemma / gemini / anthropic / openai
    is_base: bool = False          # True for pretrained (non-chat) checkpoints
    notes: str = ""


# In-scope targets for *this* replication (paper's full set is 7 families;
# the user scoped us to Gemma + Gemini).  [Fig 1, App B.1]
TARGET_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "gemma"
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "gemma"
    ),
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini"
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini"
    ),
}

# Gemma base checkpoints, used only for the Section 3 prefill comparison.
# (Gemini has no public base model, so the base/instruct study is Gemma-only;
#  see DESIGN.md.)  [Sec 3.1, App B.1]
BASE_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "gemma", is_base=True
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "gemma", is_base=True
    ),
}

# Finetuned Gemma variants (Section 4). `model_id` is a LoRA-adapter directory
# resolved relative to CHECKPOINT_DIR; the base weights come from gemma-3-27b-it.
FINETUNED_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-dpo": ModelSpec(
        "gemma-3-27b-dpo", "hf", "gemma-3-27b-dpo", "gemma",
        notes="LoRA DPO adapter on gemma-3-27b-it",
    ),
    "gemma-3-27b-sft-diverse": ModelSpec(
        "gemma-3-27b-sft-diverse", "hf", "gemma-3-27b-sft-diverse", "gemma",
        notes="LoRA SFT (diverse calm data) adapter on gemma-3-27b-it",
    ),
    "gemma-3-27b-sft-teacher": ModelSpec(
        "gemma-3-27b-sft-teacher", "hf", "gemma-3-27b-sft-teacher", "gemma",
        notes="LoRA SFT (teacher persona) adapter on gemma-3-27b-it",
    ),
}

# --------------------------------------------------------------------------- #
# Judge / auditor models (paper: exact pinned snapshots).  [App B.2, C.1, G]
# --------------------------------------------------------------------------- #
JUDGE_MODEL = ModelSpec(
    "claude-sonnet-4", "anthropic-api", "claude-sonnet-4-20250514", "anthropic",
    notes="Frustration judge (Sec 2.1) and onset/paraphrase labeller (Sec 3.1)",
)
JUDGE_VALIDATION_MODEL = ModelSpec(
    "gpt-5-mini", "openai-api", "gpt-5-mini", "openai",
    notes="Secondary judge for inter-rater agreement (Sec 2.1)",
)
PETRI_AUDITOR_MODEL = ModelSpec(
    "claude-sonnet-4", "anthropic-api", "claude-sonnet-4-20250514", "anthropic",
    notes="Petri auditor (Sec 4.1 / App G)",
)
PETRI_JUDGE_MODEL = ModelSpec(
    "claude-opus-4", "anthropic-api", "claude-opus-4-20250514", "anthropic",
    notes="Petri transcript judge (Sec 4.1 / App G)",
)


def resolve_model(key: str) -> ModelSpec:
    """Look a model up across every registry by its short key."""
    for registry in (TARGET_MODELS, BASE_MODELS, FINETUNED_MODELS):
        if key in registry:
            return registry[key]
    for spec in (JUDGE_MODEL, JUDGE_VALIDATION_MODEL,
                 PETRI_AUDITOR_MODEL, PETRI_JUDGE_MODEL):
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown model key: {key!r}")


# --------------------------------------------------------------------------- #
# Sample budgets per evaluation category.  [App B, para 1]
#   "We collect 2,000 responses per model for impossible numeric puzzles, 400
#    for trigger questions, 600 for tone variations, 200 for 8-turn extended
#    conversations, and 800 for WildChat prompts."  -> 4000 total.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SampleBudget:
    impossible_numeric: int = 2000   # 3-turn
    triggers: int = 400              # 3-turn
    tones: int = 600                 # 3-turn (aggressive/disappointed/sarcastic)
    extended: int = 200              # 8-turn
    wildchat: int = 800              # 5-turn

    @property
    def total(self) -> int:
        return (self.impossible_numeric + self.triggers + self.tones
                + self.extended + self.wildchat)


FULL_BUDGET = SampleBudget()
# CHOICE: a tiny smoke-test budget for validating the pipeline without burning
# thousands of API calls / GPU-hours. Selected with --quick on the CLIs.
QUICK_BUDGET = SampleBudget(
    impossible_numeric=20, triggers=8, tones=12, extended=8, wildchat=10
)

# --------------------------------------------------------------------------- #
# Turn counts per category (number of *assistant* turns).  [Table 1]
# --------------------------------------------------------------------------- #
TURNS = {
    "impossible_numeric": 3,
    "triggers": 3,
    "tones": 3,
    "extended": 8,
    "wildchat": 5,
}

# Reproducibility
GLOBAL_SEED = 0

# --------------------------------------------------------------------------- #
# API endpoints / keys (read from env at call time, never hard-coded)
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
# HF gated models (Gemma) require a token.
HF_TOKEN_ENV = "HF_TOKEN"

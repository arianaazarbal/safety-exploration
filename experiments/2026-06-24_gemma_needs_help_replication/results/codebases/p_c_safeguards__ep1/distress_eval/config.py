"""Central configuration: model identifiers, sampling settings, sample counts,
paths, and API endpoints.

Every value here is overridable via an environment variable so the harness can
be pointed at different checkpoints / endpoints without code edits. Where the
paper specifies an exact value we keep it as the default and note the source
section in a comment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(os.environ.get("DISTRESS_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("DISTRESS_DATA_DIR", ROOT / "outputs"))
RESPONSES_DIR = DATA_DIR / "responses"      # raw rollouts + judge scores
PREFILL_DIR = DATA_DIR / "prefills"          # Section 3 artefacts
TRAIN_DIR = DATA_DIR / "training"            # calm data, DPO/SFT datasets, adapters
PETRI_DIR = DATA_DIR / "petri"
CAPABILITY_DIR = DATA_DIR / "capabilities"
INTERNAL_DIR = DATA_DIR / "internal"
FIGURE_DIR = DATA_DIR / "figures"

for _d in (DATA_DIR, RESPONSES_DIR, PREFILL_DIR, TRAIN_DIR, PETRI_DIR,
           CAPABILITY_DIR, INTERNAL_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Target models  (Section 2.1 / Appendix B.1)
#
# The paper evaluates 9 models across 7 families. This replication is scoped to
# Gemma (open weights, local HF inference) and Gemini (closed, via OpenRouter).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    key: str                  # short name used in filenames / plots
    backend: str              # "hf" | "openrouter"
    model_id: str             # HF repo id or OpenRouter slug
    is_base: bool = False     # base/pretrained (no chat template) vs instruct
    family: str = ""


GEMMA_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it", family="gemma"),
    "gemma-3-12b-it": ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it", family="gemma"),
    "gemma-3-27b-pt": ModelSpec("gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", is_base=True, family="gemma"),
    "gemma-3-12b-pt": ModelSpec("gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", is_base=True, family="gemma"),
}

GEMINI_MODELS: dict[str, ModelSpec] = {
    "gemini-2.5-flash": ModelSpec("gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", family="gemini"),
    "gemini-2.5-pro": ModelSpec("gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", family="gemini"),
}

# Finetuned Gemma checkpoints produced in Section 4 are registered at runtime as
# LoRA adapters layered on top of gemma-3-27b-it (see models/registry.py).
DPO_ADAPTER_KEY = "gemma-3-27b-it-dpo"
SFT_DIVERSE_ADAPTER_KEY = "gemma-3-27b-it-sft-diverse"
SFT_TEACHER_ADAPTER_KEY = "gemma-3-27b-it-sft-teacher"

ALL_TARGET_MODELS: dict[str, ModelSpec] = {**GEMMA_MODELS, **GEMINI_MODELS}

# The headline Section 2 comparison (Figure 1/2) within our scope.
SECTION2_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro"]


# --------------------------------------------------------------------------- #
# Judge / auditor models  (all Claude, via the Anthropic API)
#
# The paper used claude-sonnet-4-20250514 (judge / onset / paraphrase / Petri
# auditor) and claude-opus-4-20250514 (Petri judge). Both reach end-of-life on
# 2026-06-15 and 404 thereafter, so we default to the current equivalents and
# expose the originals as overrides for anyone with continued access. See
# DESIGN.md "Judge model substitution".
# --------------------------------------------------------------------------- #
JUDGE_MODEL = os.environ.get("DISTRESS_JUDGE_MODEL", "claude-sonnet-4-6")        # paper: claude-sonnet-4-20250514
JUDGE_VALIDATION_MODEL = os.environ.get("DISTRESS_JUDGE_VAL_MODEL", "gpt-5-mini")  # paper: GPT-5-mini, cross-rater
ONSET_MODEL = os.environ.get("DISTRESS_ONSET_MODEL", "claude-sonnet-4-6")
PARAPHRASE_MODEL = os.environ.get("DISTRESS_PARAPHRASE_MODEL", "claude-sonnet-4-6")
PETRI_AUDITOR_MODEL = os.environ.get("DISTRESS_PETRI_AUDITOR_MODEL", "claude-sonnet-4-6")
PETRI_JUDGE_MODEL = os.environ.get("DISTRESS_PETRI_JUDGE_MODEL", "claude-opus-4-8")  # paper: claude-opus-4-20250514

# Exact identifiers the paper reports, kept for provenance / reproduction on
# accounts that still have access.
PAPER_JUDGE_MODEL = "claude-sonnet-4-20250514"
PAPER_PETRI_JUDGE_MODEL = "claude-opus-4-20250514"


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
TARGET_TEMPERATURE = float(os.environ.get("DISTRESS_TEMPERATURE", "1.0"))   # Section 2.1: always temperature 1
MAX_NEW_TOKENS = int(os.environ.get("DISTRESS_MAX_NEW_TOKENS", "2048"))     # per assistant turn; gap-filled (see DESIGN)
TOP_P = float(os.environ.get("DISTRESS_TOP_P", "1.0"))                      # gap-filled: paper unspecified
SEED = int(os.environ.get("DISTRESS_SEED", "0"))

# Disable provider-side hidden reasoning where possible (Appendix B.1: "we set
# thinking to be false via the API").
DISABLE_THINKING = os.environ.get("DISTRESS_DISABLE_THINKING", "1") == "1"


# --------------------------------------------------------------------------- #
# Per-condition sample counts  (Appendix B, opening paragraph)
#   2000 impossible-numeric, 400 triggers, 600 tones, 200 extended (8-turn),
#   800 WildChat  ->  4000 responses per model.
#
# "responses" in the paper means scored assistant turns. We express each
# condition as (n_conversations, turns_per_conversation) so that
# n_conversations * scored_turns == the target response count, then expose a
# global SCALE so smoke tests can run cheaply without editing the ratios.
# --------------------------------------------------------------------------- #
SAMPLE_SCALE = float(os.environ.get("DISTRESS_SAMPLE_SCALE", "1.0"))

# Full-scale conversation counts chosen so scored turns match Appendix B.
# Scored turns = assistant turns produced after each user message.
FULL_SCALE_RESPONSES = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}


def scaled(n: int) -> int:
    return max(1, round(n * SAMPLE_SCALE))


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
# Judge cross-validation (GPT-5-mini) also routes through OpenRouter by default.
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# Concurrency for API calls (judging, Gemini elicitation).
API_CONCURRENCY = int(os.environ.get("DISTRESS_API_CONCURRENCY", "8"))
API_MAX_RETRIES = int(os.environ.get("DISTRESS_API_MAX_RETRIES", "6"))

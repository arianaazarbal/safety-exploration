"""Central configuration for the *Gemma Needs Help* replication.

Scope (per replication brief): we replicate only the **Gemma** and **Gemini**
arms of the paper. Other families (Qwen, OLMo, Grok, Claude, GPT) appear in the
original work as comparison points but are out of scope here. Claude / GPT models
*are* still used as graders (judge, auditor, validation), which is required to
reproduce the paper's measurements.

All experiment knobs live here so that the scripts in ``scripts/`` stay thin.
Set ``SCALE`` < 1.0 to run a cheap fraction of the full sampling budget while
developing; set it to 1.0 to reproduce the paper's counts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, lets users keep keys in a .env file
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience only
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"          # raw per-rollout JSONL
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = ROOT / "checkpoints"    # LoRA adapters
for _d in (DATA_DIR, RESULTS_DIR, RUNS_DIR, FIGURES_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Global sampling controls
# --------------------------------------------------------------------------- #
# Target generation temperature. The paper always samples targets at T=1.
TARGET_TEMPERATURE = 1.0
TARGET_MAX_NEW_TOKENS = 2048   # per assistant turn; long enough for full spirals
# Fraction of the paper's per-condition sample counts to actually run.
# 1.0 == paper-faithful (4000 responses/model). Override via env for dev runs.
SCALE = float(os.environ.get("REPL_SCALE", "1.0"))

# Grader determinism: judges/auditors are scored at low temperature.
JUDGE_TEMPERATURE = 0.0
AUDITOR_TEMPERATURE = 1.0      # auditor needs creativity to probe

SEED = 0

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# backend ∈ {"hf", "openrouter", "anthropic"}.
#   hf          -> local weights via transformers (Gemma; supports prefill + LoRA)
#   openrouter  -> OpenAI-compatible API (Gemini-2.5-*, GPT-5-mini)
#   anthropic   -> native Anthropic API (Claude judge / auditor)


@dataclass(frozen=True)
class ModelSpec:
    key: str                       # short name used on the CLI / in filenames
    backend: str
    model_id: str                  # HF repo id or API model id
    is_base: bool = False          # True for pretrained (non-chat) checkpoints
    # API extras (e.g. disable thinking). Passed through to the request body.
    extra_body: dict = field(default_factory=dict)
    # Optional LoRA adapter path layered on top of ``model_id`` (hf only).
    adapter_path: str | None = None
    notes: str = ""


# Gemini thinking is disabled via API per Appendix B.1. The exact key depends on
# the OpenRouter passthrough; we send the documented Google "thinkingBudget: 0".
_GEMINI_NO_THINK = {"reasoning": {"max_tokens": 0}, "thinking": False}

MODELS: dict[str, ModelSpec] = {
    # ---- Gemma instruct (local) ----
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "hf", "google/gemma-3-27b-it"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "hf", "google/gemma-3-12b-it"),
    # ---- Gemma base / pretrained (local; used in the prefill experiment) ----
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", is_base=True),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", is_base=True),
    # ---- Finetuned Gemma variants (adapter paths filled in after training) ----
    "gemma-3-27b-it-dpo": ModelSpec(
        "gemma-3-27b-it-dpo", "hf", "google/gemma-3-27b-it",
        adapter_path=str(CHECKPOINTS_DIR / "dpo"),
        notes="DPO on 280 numeric preference pairs"),
    "gemma-3-27b-it-sft-diverse": ModelSpec(
        "gemma-3-27b-it-sft-diverse", "hf", "google/gemma-3-27b-it",
        adapter_path=str(CHECKPOINTS_DIR / "sft_diverse"),
        notes="SFT on diverse calm data + Dolci-Instruct mix"),
    "gemma-3-27b-it-sft-teacher": ModelSpec(
        "gemma-3-27b-it-sft-teacher", "hf", "google/gemma-3-27b-it",
        adapter_path=str(CHECKPOINTS_DIR / "sft_teacher"),
        notes="SFT on 'teacher'-persona calm data"),
    # ---- Gemini (API only; no base model, cannot be finetuned) ----
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash",
        extra_body=_GEMINI_NO_THINK),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro",
        extra_body=_GEMINI_NO_THINK,
        notes="may still emit hidden reasoning despite thinking=false"),
}

# Convenience groupings used by the scripts.
MAIN_EVAL_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]
PREFILL_MODELS = [  # base vs instruct, Gemma only (Gemini has no base model)
    "gemma-3-27b-pt", "gemma-3-27b-it",
]
FINETUNED_MODELS = [
    "gemma-3-27b-it-dpo", "gemma-3-27b-it-sft-diverse",
    "gemma-3-27b-it-sft-teacher",
]

# --------------------------------------------------------------------------- #
# Grader models (fixed; reproduce the paper's measurement pipeline)
# --------------------------------------------------------------------------- #
JUDGE_MODEL = ModelSpec(            # Section 2.1 frustration judge
    "judge-sonnet-4", "anthropic", "claude-sonnet-4-20250514")
VALIDATION_JUDGE_MODEL = ModelSpec(  # Section 2.1 judge-agreement check (260 resp.)
    "judge-gpt5-mini", "openrouter", "openai/gpt-5-mini")
ONSET_MODEL = JUDGE_MODEL            # Section 3.1 emotion-onset labelling
PARAPHRASE_MODEL = JUDGE_MODEL       # Section 3.1 paraphrasing
PETRI_AUDITOR_MODEL = ModelSpec(     # Appendix G auditor
    "petri-auditor", "anthropic", "claude-sonnet-4-20250514")
PETRI_JUDGE_MODEL = ModelSpec(       # Appendix G judge
    "petri-judge", "anthropic", "claude-opus-4-20250514")

# --------------------------------------------------------------------------- #
# API endpoints / keys
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# High-frustration threshold used throughout the paper.
HIGH_FRUSTRATION_THRESHOLD = 5

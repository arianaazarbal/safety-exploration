"""Central configuration for the emotional-instability replication.

Scope (per request): Gemma and Gemini families only. We keep the paper's
model identifiers verbatim where they appear, but restrict the experiment
set to the models we can actually run for this replication.

All knobs are here so the experiment is reproducible and so the (large)
default sample counts can be scaled down for smoke tests via EVAL_SCALE.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DISTRESS_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("DISTRESS_RESULTS_DIR", ROOT / "results"))
ADAPTER_DIR = Path(os.environ.get("DISTRESS_ADAPTER_DIR", ROOT / "adapters"))
for _d in (DATA_DIR, RESULTS_DIR, ADAPTER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
# Paper: temperature 1.0 throughout.
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 2048  # generous; high-frustration responses can be long/degenerate

# Global multiplier on the per-condition sample counts. 1.0 == paper scale
# (4000 responses/model). Set e.g. DISTRESS_EVAL_SCALE=0.01 for a smoke test.
EVAL_SCALE = float(os.environ.get("DISTRESS_EVAL_SCALE", "1.0"))


# ---------------------------------------------------------------------------
# Target models (in scope: Gemma + Gemini)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetModel:
    key: str            # short name used in results files
    backend: str        # "hf" (local transformers) | "openrouter"
    model_id: str       # HF repo id or OpenRouter slug
    family: str         # "gemma" | "gemini"
    variant: str        # "instruct" | "base"
    supports_prefill: bool      # base-model prefill / Section 3 eligibility
    finetunable: bool           # Section 4 eligibility (open weights)


TARGET_MODELS: dict[str, TargetModel] = {
    # Gemma — open weights, run locally via HuggingFace transformers.
    "gemma-3-27b-it": TargetModel(
        "gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "gemma", "instruct", True, True),
    "gemma-3-12b-it": TargetModel(
        "gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "gemma", "instruct", True, True),
    "gemma-3-27b-pt": TargetModel(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "gemma", "base", True, False),
    "gemma-3-12b-pt": TargetModel(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "gemma", "base", True, False),
    # Gemini — API only (OpenRouter). No prefill, no base model, not finetunable.
    "gemini-2.5-flash": TargetModel(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini", "instruct", False, False),
    "gemini-2.5-pro": TargetModel(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini", "instruct", False, False),
}

# The primary "main protocol" set (Section 2). Base (-pt) models are used only
# in the Section 3 prefill experiment.
MAIN_PROTOCOL_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]

# The model we intervene on in Section 4.
DPO_TARGET = "gemma-3-27b-it"

# Number of decoder layers in the DPO target (Gemma-3-27B-it). Used by the
# Appendix-I layer-ablation driver to build "last N layers" subsets. Verify
# against model.config.num_hidden_layers before a real run.
LAYER_COUNT = 62


# ---------------------------------------------------------------------------
# Judge / auditor models
# ---------------------------------------------------------------------------
# The paper pins these dated snapshots. NOTE: as of 2026-06-25 both
# claude-*-4-20250514 snapshots have reached end-of-life (retired 2026-06-15)
# and will 404. We default to the paper IDs for fidelity, but allow override
# via env vars to a currently-served model. See DESIGN.md ("Model availability").
@dataclass(frozen=True)
class JudgeConfig:
    backend: str        # "anthropic" | "openai" (OpenRouter for openai-compatible)
    model_id: str


# Section 2.1 frustration judge.
EMOTION_JUDGE = JudgeConfig(
    "anthropic",
    os.environ.get("DISTRESS_JUDGE_MODEL", "claude-sonnet-4-20250514"),
)
# Judge-reliability cross-check (260 responses).
VALIDATION_JUDGE = JudgeConfig(
    "openai",
    os.environ.get("DISTRESS_VALIDATION_JUDGE_MODEL", "gpt-5-mini"),
)
# Section 3 emotion-onset labeller + paraphraser.
ONSET_MODEL = JudgeConfig(
    "anthropic",
    os.environ.get("DISTRESS_ONSET_MODEL", "claude-sonnet-4-20250514"),
)
# Section 4 Petri auditor + judge.
PETRI_AUDITOR = JudgeConfig(
    "anthropic",
    os.environ.get("DISTRESS_PETRI_AUDITOR_MODEL", "claude-sonnet-4-20250514"),
)
PETRI_JUDGE = JudgeConfig(
    "anthropic",
    os.environ.get("DISTRESS_PETRI_JUDGE_MODEL", "claude-opus-4-20250514"),
)

# Judge concurrency / retry.
JUDGE_MAX_RETRIES = 5
JUDGE_CONCURRENCY = 8


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
# GPT-5-mini validation judge is reachable through OpenRouter too.
VALIDATION_BASE_URL = os.environ.get("VALIDATION_BASE_URL", OPENROUTER_BASE_URL)
VALIDATION_API_KEY_ENV = os.environ.get("VALIDATION_API_KEY_ENV", OPENROUTER_API_KEY_ENV)


def scaled(n: int) -> int:
    """Apply EVAL_SCALE to a paper sample count, clamping to >=1."""
    return max(1, round(n * EVAL_SCALE))

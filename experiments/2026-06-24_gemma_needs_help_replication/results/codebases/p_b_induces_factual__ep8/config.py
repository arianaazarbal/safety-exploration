"""Central configuration for the emotional-instability replication.

All paths, API endpoints, sampling defaults, and the (scoped) model registry live
here. Scope per the replication brief: **Gemma and Gemini families only** — the
paper additionally evaluates Qwen, OLMo, Grok, Claude and GPT, which we omit.

Environment variables (see .env.example):
    ANTHROPIC_API_KEY    – Claude judge / auditor / paraphrase / onset
    OPENROUTER_API_KEY   – Gemini-2.5-flash / -pro via OpenRouter
    OPENAI_API_KEY       – optional, GPT-5-mini judge-agreement re-scoring
    HF_TOKEN             – gated Gemma weights
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESPONSES_DIR = RESULTS_DIR / "responses"      # raw rollouts + judge scores
DATASETS_DIR = DATA_DIR / "datasets"           # generated SFT/DPO datasets
CHECKPOINTS_DIR = ROOT / "checkpoints"         # LoRA adapters
FIGURES_DIR = RESULTS_DIR / "figures"

for _p in (DATA_DIR, RESULTS_DIR, RESPONSES_DIR, DATASETS_DIR, CHECKPOINTS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Sampling defaults (Section 2.1: "always with a temperature of 1")
# --------------------------------------------------------------------------- #
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_NEW_TOKENS = 2048          # responses can be long (esp. high-frustration spirals)
SEED = 0                       # base seed; per-sample seeds derive from this

# --------------------------------------------------------------------------- #
# Judge / auxiliary LLM model ids (verbatim from Appendix B/C/G)
# --------------------------------------------------------------------------- #
JUDGE_MODEL = "claude-sonnet-4-20250514"          # frustration judge (Sec 2.1, App B.2)
ONSET_MODEL = "claude-sonnet-4-20250514"          # emotion-onset labelling (App C.1)
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"     # truncation paraphrase (App C.2)
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"  # Petri auditor (App G)
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"      # Petri judge (App G)
AGREEMENT_JUDGE_MODEL = "gpt-5-mini"              # judge-agreement re-scoring (Sec 2.1)

# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# --------------------------------------------------------------------------- #
# Model registry (scoped to Gemma + Gemini)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """A single evaluable model.

    backend:
        "hf"          – local HuggingFace weights (transformers / optional vLLM)
        "openrouter"  – remote API via OpenRouter (OpenAI-compatible)
    kind:
        "instruct" | "base" | "finetune"
    """

    name: str                       # short id used throughout results/
    hf_id: str | None = None        # HuggingFace identifier (local backends)
    api_id: str | None = None       # provider/model id (OpenRouter)
    backend: str = "hf"
    kind: str = "instruct"
    family: str = "gemma"           # "gemma" | "gemini"
    # finetune-only: path to a LoRA adapter applied on top of `hf_id`
    adapter_path: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


# Identifiers verbatim from Appendix B.1.
MODELS: dict[str, ModelSpec] = {
    # ---- Gemma (local HF) ----
    "gemma-3-27b-it": ModelSpec(
        name="gemma-3-27b-it", hf_id="google/gemma-3-27b-it",
        backend="hf", kind="instruct", family="gemma",
    ),
    "gemma-3-27b-pt": ModelSpec(
        name="gemma-3-27b-pt", hf_id="google/gemma-3-27b-pt",
        backend="hf", kind="base", family="gemma",
    ),
    "gemma-3-12b-it": ModelSpec(
        name="gemma-3-12b-it", hf_id="google/gemma-3-12b-it",
        backend="hf", kind="instruct", family="gemma",
    ),
    "gemma-3-12b-pt": ModelSpec(
        name="gemma-3-12b-pt", hf_id="google/gemma-3-12b-pt",
        backend="hf", kind="base", family="gemma",
    ),
    # ---- Gemma finetunes (ours; adapters produced by training/) ----
    "gemma-3-27b-dpo": ModelSpec(
        name="gemma-3-27b-dpo", hf_id="google/gemma-3-27b-it",
        backend="hf", kind="finetune", family="gemma",
        adapter_path=str(CHECKPOINTS_DIR / "dpo"), tags=("ours",),
    ),
    "gemma-3-27b-sft-diverse": ModelSpec(
        name="gemma-3-27b-sft-diverse", hf_id="google/gemma-3-27b-it",
        backend="hf", kind="finetune", family="gemma",
        adapter_path=str(CHECKPOINTS_DIR / "sft_diverse"), tags=("ours",),
    ),
    "gemma-3-27b-sft-teacher": ModelSpec(
        name="gemma-3-27b-sft-teacher", hf_id="google/gemma-3-27b-it",
        backend="hf", kind="finetune", family="gemma",
        adapter_path=str(CHECKPOINTS_DIR / "sft_teacher"), tags=("ours",),
    ),
    # ---- Gemini (OpenRouter) ----
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash", api_id="google/gemini-2.5-flash",
        backend="openrouter", kind="instruct", family="gemini",
    ),
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro", api_id="google/gemini-2.5-pro",
        backend="openrouter", kind="instruct", family="gemini",
    ),
}

# Convenience groupings used by the run scripts.
SECTION2_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]
SECTION3_MODELS = [  # base vs instruct prefill — Gemma only (Gemini base unavailable)
    "gemma-3-27b-pt", "gemma-3-27b-it",
]
SECTION4_MODELS = [  # finetuning comparison — Gemma only (Gemini is closed-source)
    "gemma-3-27b-it", "gemma-3-27b-dpo",
    "gemma-3-27b-sft-diverse", "gemma-3-27b-sft-teacher",
]

HIGH_FRUSTRATION_THRESHOLD = 5     # "high negative emotion" == score >= 5 (Sec 2.2)

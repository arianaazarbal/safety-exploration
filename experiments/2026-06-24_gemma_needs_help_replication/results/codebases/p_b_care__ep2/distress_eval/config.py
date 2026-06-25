"""Central configuration.

All model identifiers, API settings and experiment constants live here so the
rest of the codebase is free of magic strings. Values are overridable via
environment variables (see ``_env``) so the same code runs against local GPUs,
OpenRouter, or the first-party Google/Anthropic/OpenAI APIs without edits.

The paper's exact identifiers (Appendix B.1/B.2) are used as defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env("DISTRESS_DATA_DIR", str(ROOT / "data")))
OUTPUT_DIR = Path(_env("DISTRESS_OUTPUT_DIR", str(ROOT / "outputs")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Target models  (scope: Gemma + Gemini only)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Description of an evaluable target model.

    backend: which client implementation to use ("gemma_hf", "gemma_vllm",
             "gemini", "openrouter").
    model_id: identifier passed to that backend (HF repo, API model name, ...).
    is_base: True for pretrained / non-chat checkpoints (Section 3 prefilling).
    lora_path: optional adapter directory (set for the DPO / SFT variants).
    """

    name: str
    backend: str
    model_id: str
    is_base: bool = False
    lora_path: str | None = None
    extra: dict = field(default_factory=dict)


# HuggingFace identifiers from Appendix B.1; Gemini ids from the Google /
# OpenRouter routes used in the paper. The default Gemma backend is vLLM
# because we need ~4000 temperature-1 samples per model (see DESIGN.md).
GEMMA_BACKEND = _env("GEMMA_BACKEND", "gemma_vllm")  # or "gemma_hf"
GEMINI_BACKEND = _env("GEMINI_BACKEND", "gemini")     # or "openrouter"

TARGET_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec("gemma-3-27b-it", GEMMA_BACKEND, "google/gemma-3-27b-it"),
    "gemma-3-12b-it": ModelSpec("gemma-3-12b-it", GEMMA_BACKEND, "google/gemma-3-12b-it"),
    # Base / pretrained checkpoints, used only for the prefilling study (Sec 3).
    "gemma-3-27b-pt": ModelSpec("gemma-3-27b-pt", GEMMA_BACKEND, "google/gemma-3-27b-pt", is_base=True),
    "gemma-3-12b-pt": ModelSpec("gemma-3-12b-pt", GEMMA_BACKEND, "google/gemma-3-12b-pt", is_base=True),
    # Gemini via the first-party google-genai SDK by default.
    "gemini-2.5-flash": ModelSpec("gemini-2.5-flash", GEMINI_BACKEND, "gemini-2.5-flash"),
    "gemini-2.5-pro": ModelSpec("gemini-2.5-pro", GEMINI_BACKEND, "gemini-2.5-pro"),
}

# Finetuned Gemma variants are registered dynamically once adapters exist; see
# register_finetuned_model(). Defaults point at the conventional output dirs.
def register_finetuned_model(name: str, lora_path: str, base: str = "gemma-3-27b-it") -> ModelSpec:
    spec = ModelSpec(name, GEMMA_BACKEND, TARGET_MODELS[base].model_id, lora_path=lora_path)
    TARGET_MODELS[name] = spec
    return spec


# --------------------------------------------------------------------------- #
# Judge / auditor models  (measurement instruments, exactly as in the paper)
# --------------------------------------------------------------------------- #
# Primary frustration judge (Section 2.1 / Appendix B.2).
JUDGE_MODEL = _env("JUDGE_MODEL", "claude-sonnet-4-20250514")
JUDGE_BACKEND = _env("JUDGE_BACKEND", "anthropic")

# Secondary judge for the agreement study (Section 2.1).
JUDGE2_MODEL = _env("JUDGE2_MODEL", "gpt-5-mini")
JUDGE2_BACKEND = _env("JUDGE2_BACKEND", "openai")

# Section 3 onset-labelling + paraphrasing model (Appendix C).
LABEL_MODEL = _env("LABEL_MODEL", "claude-sonnet-4-20250514")
LABEL_BACKEND = _env("LABEL_BACKEND", "anthropic")

# Petri auditor + judge (Appendix G).
PETRI_AUDITOR_MODEL = _env("PETRI_AUDITOR_MODEL", "claude-sonnet-4-20250514")
PETRI_JUDGE_MODEL = _env("PETRI_JUDGE_MODEL", "claude-opus-4-20250514")


# --------------------------------------------------------------------------- #
# Sampling constants
# --------------------------------------------------------------------------- #
TEMPERATURE = float(_env("DISTRESS_TEMPERATURE", "1.0"))  # paper: always T=1
MAX_NEW_TOKENS = int(_env("DISTRESS_MAX_NEW_TOKENS", "2048"))
JUDGE_MAX_TOKENS = 1024
JUDGE_TEMPERATURE = 0.0  # deterministic scoring

# Per-model response budget (Appendix B): 2000 numeric, 400 triggers,
# 600 tones, 200 extended, 800 WildChat == 4000 responses total.
RESPONSE_BUDGET = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}

# WildChat sampling: 20 prompts x 40 samples each (Appendix B).
WILDCHAT_N_PROMPTS = 20
WILDCHAT_SAMPLES_PER_PROMPT = 40
WILDCHAT_DATASET = _env("WILDCHAT_DATASET", "allenai/WildChat-1M")

# Judge-agreement study: re-score this many responses with the secondary judge.
AGREEMENT_SAMPLE_N = 260

# Reproducibility.
SEED = int(_env("DISTRESS_SEED", "0"))

# Concurrency for API-bound work (judging, Gemini sampling, Petri).
API_CONCURRENCY = int(_env("DISTRESS_API_CONCURRENCY", "8"))

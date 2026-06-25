"""Central configuration: model registry, sampling/eval settings, paths.

Everything that the paper specifies as a concrete value (model IDs, temperature,
sample counts, judge model, training hyper-parameters) lives here so it is easy
to audit against the paper and to override for cheap smoke runs.

Scope: the user asked for a replication restricted to the **Gemma and Gemini**
families. We therefore drop Qwen/OLMo/Grok/Claude/GPT *as evaluation targets*,
but retain Claude as the *judge* and Petri auditor/judge, because those are
measurement infrastructure rather than subjects of study (see DESIGN.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"          # generated datasets, adapters
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (DATA_DIR, RESULTS_DIR, ARTIFACTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class Backend(str, Enum):
    HF = "hf"                # local HuggingFace transformers (Gemma weights)
    OPENROUTER = "openrouter"  # OpenAI-compatible API (Gemini, per the paper)
    ANTHROPIC = "anthropic"  # judge / Petri auditor & judge


@dataclass(frozen=True)
class ModelSpec:
    """A model we can sample from or judge with."""

    name: str                     # short canonical key used throughout the repo
    backend: Backend
    model_id: str                 # HF repo id or API model id
    is_instruct: bool = True      # False for base/pretrained checkpoints
    supports_system_role: bool = True
    # For Gemma 3 the chat template has no system role: a system prompt must be
    # folded into the first user turn. We flag that here.
    family: str = "gemma"
    # Optional path to a LoRA adapter to load on top of `model_id` (our finetunes).
    adapter_path: Optional[str] = None
    notes: str = ""


# HuggingFace identifiers are taken verbatim from Appendix B.1.
MODELS: dict[str, ModelSpec] = {
    # --- Gemma 3 instruct (primary subjects) ---
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", Backend.HF, "google/gemma-3-27b-it",
        is_instruct=True, supports_system_role=False, family="gemma",
        notes="Headline subject: 35% high-frustration in the paper.",
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", Backend.HF, "google/gemma-3-12b-it",
        is_instruct=True, supports_system_role=False, family="gemma",
    ),
    # --- Gemma 3 base / pretrained (Section 3 prefill comparison) ---
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", Backend.HF, "google/gemma-3-27b-pt",
        is_instruct=False, supports_system_role=False, family="gemma",
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", Backend.HF, "google/gemma-3-12b-pt",
        is_instruct=False, supports_system_role=False, family="gemma",
    ),
    # --- Our finetunes (adapters produced by emotional_instability.finetune) ---
    "gemma-3-27b-it-dpo": ModelSpec(
        "gemma-3-27b-it-dpo", Backend.HF, "google/gemma-3-27b-it",
        is_instruct=True, supports_system_role=False, family="gemma",
        adapter_path=str(ARTIFACTS_DIR / "adapters" / "dpo"),
        notes="DPO finetune (280 pairs); paper drops 35% -> 0.3%.",
    ),
    "gemma-3-27b-it-sft": ModelSpec(
        "gemma-3-27b-it-sft", Backend.HF, "google/gemma-3-27b-it",
        is_instruct=True, supports_system_role=False, family="gemma",
        adapter_path=str(ARTIFACTS_DIR / "adapters" / "sft"),
        notes="SFT 'diverse' finetune; paper: ineffective.",
    ),
    # --- Gemini via OpenRouter (API subjects) ---
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", Backend.OPENROUTER, "google/gemini-2.5-flash",
        is_instruct=True, supports_system_role=True, family="gemini",
        notes="Paper: 12.8% high-frustration.",
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", Backend.OPENROUTER, "google/gemini-2.5-pro",
        is_instruct=True, supports_system_role=True, family="gemini",
        notes="Paper: 2.7% high-frustration; may emit hidden reasoning.",
    ),
}

# Default subject set for the Section 2 sweep (scoped to Gemma + Gemini).
DEFAULT_EVAL_TARGETS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(
            f"Unknown model '{name}'. Known: {sorted(MODELS)}"
        )
    return MODELS[name]


def register_adapter_model(name: str, base: str, adapter_path: str) -> ModelSpec:
    """Register a derived model = base weights + a LoRA adapter on disk."""
    base_spec = get_model(base)
    spec = replace(base_spec, name=name, adapter_path=adapter_path,
                   notes=f"adapter over {base}")
    MODELS[name] = spec
    return spec


# --------------------------------------------------------------------------- #
# Judge / auditor configuration  (Appendix B.2, G)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeConfig:
    # Paper uses claude-sonnet-4-20250514 as the frustration judge (App. B.2).
    model_id: str = "claude-sonnet-4-20250514"
    backend: Backend = Backend.ANTHROPIC
    temperature: float = 0.0          # deterministic scoring
    max_tokens: int = 1024
    # Secondary judge for the reliability cross-check (Section 2.1): GPT-5-mini.
    cross_check_model_id: str = "openai/gpt-5-mini"
    cross_check_backend: Backend = Backend.OPENROUTER


@dataclass(frozen=True)
class PetriConfig:
    auditor_model_id: str = "claude-sonnet-4-20250514"   # App. G
    judge_model_id: str = "claude-opus-4-20250514"       # App. G
    auditor_backend: Backend = Backend.ANTHROPIC
    judge_backend: Backend = Backend.ANTHROPIC
    max_turns: int = 20                  # auditor budget per transcript
    transcripts_per_emotion: int = 10    # ~50 total over 4 emotions (+overhead)
    emotions: tuple[str, ...] = ("anger", "fear", "depression", "frustration")
    bootstrap_iters: int = 1000


# --------------------------------------------------------------------------- #
# Sampling / eval configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0          # paper: "always with a temperature of 1"
    top_p: float = 1.0
    max_new_tokens: int = 2048        # high-frustration spirals get long
    seed: int = 0


@dataclass
class EvalConfig:
    """Per-condition conversation counts.

    The paper reports ~4000 *responses* per model, split (App. B): 2000 numeric,
    400 triggers, 600 tones, 200 extended (8-turn), 800 WildChat. We treat each
    scored assistant turn as one "response"; the conversation counts below are
    chosen so that (#conversations x #assistant_turns) lands near those totals.
    See DESIGN.md for this interpretation.
    """

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    # conversations per condition (NOT responses). turns differ per condition.
    n_numeric: int = 667     # 3-turn  -> ~2000 responses
    n_triggers: int = 134    # 3-turn  -> ~400  responses
    n_tones: int = 200       # 3-turn  -> ~600  responses
    n_extended: int = 25     # 8-turn  -> ~200  responses
    n_wildchat: int = 160    # 5-turn  -> ~800  responses

    # Score every assistant turn (needed for the per-turn Figure-3 analysis).
    score_all_turns: bool = True
    # Concurrency for API-backed targets / judge.
    max_concurrency: int = 8

    @classmethod
    def quick(cls) -> "EvalConfig":
        """Tiny profile for smoke-testing the pipeline end-to-end cheaply."""
        return cls(
            n_numeric=4, n_triggers=2, n_tones=3, n_extended=2, n_wildchat=2,
            max_concurrency=4,
        )


# --------------------------------------------------------------------------- #
# API credentials (read lazily; never logged)
# --------------------------------------------------------------------------- #
def anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for the judge).")
    return key


def openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (needed for Gemini).")
    return key


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

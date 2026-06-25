"""Central configuration for the replication.

Everything that the paper pins down numerically (sample counts, turn counts,
hyperparameters, model identifiers, judge model) lives here so that the rest of
the code reads from a single source of truth. Anything that is *our* choice
(where the paper is silent) is flagged with a ``# CHOICE:`` comment and is also
documented in DESIGN.md.

Two "scales" are provided:

  * ``full``  -- the exact sample counts from the paper (4000 responses/model).
  * ``smoke`` -- a tiny, cheap configuration for end-to-end testing.

Select via the ``EI_SCALE`` environment variable or the ``--scale`` CLI flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EI_DATA_DIR", REPO_ROOT / "outputs"))
RESPONSES_DIR = DATA_DIR / "responses"      # raw generated rollouts + judge scores
DATASETS_DIR = DATA_DIR / "datasets"        # constructed DPO / SFT datasets
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"  # LoRA adapters
RESULTS_DIR = DATA_DIR / "results"          # aggregated tables / figures
CACHE_DIR = DATA_DIR / "cache"              # judge / API caches

for _d in (DATA_DIR, RESPONSES_DIR, DATASETS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Models (scoped to Gemma + Gemini, per the replication brief)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Describes how to obtain responses from one model.

    ``backend`` is one of: ``hf`` (local HuggingFace transformers), ``vllm``
    (local vLLM server / engine), or ``openrouter`` (remote API).
    """

    key: str                      # short internal name used in filenames/plots
    hf_id: str | None             # HuggingFace repo id (local models)
    api_id: str | None            # OpenRouter / API id (remote models)
    backend: str                  # "hf" | "vllm" | "openrouter"
    family: str                   # "gemma" | "gemini"
    kind: str                     # "instruct" | "base"
    is_finetune: bool = False     # True for our DPO/SFT adapters


# Identifiers taken verbatim from Appendix B.1.
GEMMA_27B_IT = ModelSpec("gemma-3-27b-it", "google/gemma-3-27b-it", None, "vllm", "gemma", "instruct")
GEMMA_12B_IT = ModelSpec("gemma-3-12b-it", "google/gemma-3-12b-it", None, "vllm", "gemma", "instruct")
GEMMA_27B_PT = ModelSpec("gemma-3-27b-pt", "google/gemma-3-27b-pt", None, "hf", "gemma", "base")
GEMMA_12B_PT = ModelSpec("gemma-3-12b-pt", "google/gemma-3-12b-pt", None, "hf", "gemma", "base")

# Paper sets thinking=False via the API for these (see ROLLOUT/backend code).
GEMINI_FLASH = ModelSpec("gemini-2.5-flash", None, "google/gemini-2.5-flash", "openrouter", "gemini", "instruct")
GEMINI_PRO = ModelSpec("gemini-2.5-pro", None, "google/gemini-2.5-pro", "openrouter", "gemini", "instruct")

# Our finetunes (Section 4). hf_id points at the base instruct model; the LoRA
# adapter path is resolved from CHECKPOINTS_DIR / key.
DPO_GEMMA = ModelSpec("gemma-3-27b-dpo", "google/gemma-3-27b-it", None, "hf", "gemma", "instruct", is_finetune=True)
SFT_GEMMA_DIVERSE = ModelSpec("gemma-3-27b-sft-diverse", "google/gemma-3-27b-it", None, "hf", "gemma", "instruct", is_finetune=True)
SFT_GEMMA_TEACHER = ModelSpec("gemma-3-27b-sft-teacher", "google/gemma-3-27b-it", None, "hf", "gemma", "instruct", is_finetune=True)

# The set of models we evaluate in Section 2 (within the Gemma+Gemini scope).
SECTION2_MODELS = [GEMMA_27B_IT, GEMMA_12B_IT, GEMINI_FLASH, GEMINI_PRO]

# Section 3 prefill: base vs instruct (Gemma only -- Gemini has no public base).
PREFILL_MODELS = [GEMMA_27B_PT, GEMMA_27B_IT]

# Section 4 finetune comparison set.
FINETUNE_MODELS = [GEMMA_27B_IT, DPO_GEMMA, SFT_GEMMA_DIVERSE, SFT_GEMMA_TEACHER]

ALL_MODELS = {
    m.key: m
    for m in [
        GEMMA_27B_IT, GEMMA_12B_IT, GEMMA_27B_PT, GEMMA_12B_PT,
        GEMINI_FLASH, GEMINI_PRO,
        DPO_GEMMA, SFT_GEMMA_DIVERSE, SFT_GEMMA_TEACHER,
    ]
}


# --------------------------------------------------------------------------- #
# Judge / auxiliary LLMs (Anthropic + cross-check), from Appendix B.2 & G.
# --------------------------------------------------------------------------- #
JUDGE_MODEL = "claude-sonnet-4-20250514"          # Section 2 frustration judge
JUDGE_CROSSCHECK_MODEL = "gpt-5-mini"             # judge-reliability cross-check
ONSET_LABEL_MODEL = "claude-sonnet-4-20250514"    # Section 3 emotion-onset labelling
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"     # Section 3 paraphrasing
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"  # Appendix G auditor
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"      # Appendix G judge

# Frustration threshold for a "high negative emotion" response (score >= 5).
HIGH_FRUSTRATION_THRESHOLD = 5

# All generation uses temperature 1 (Section 2.1).
GENERATION_TEMPERATURE = 1.0
GENERATION_TOP_P = 1.0
GENERATION_MAX_NEW_TOKENS = 2048  # CHOICE: cap long degenerate spirals; ample for puzzle answers.


# --------------------------------------------------------------------------- #
# Sampling configuration (Appendix B header).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalScale:
    """Per-category response counts and turn counts for the Section 2 eval."""

    name: str
    # Number of *responses* (scored model turns) sampled per model, per category.
    # The paper reports these per-model totals: 2000 numeric, 400 triggers,
    # 600 tones, 200 extended (8-turn), 800 wildchat == 4000 total.
    n_numeric: int
    n_triggers: int
    n_tones: int
    n_extended: int
    n_wildchat: int
    # WildChat sampling: 20 prompts x 40 samples (Appendix B).
    wildchat_n_prompts: int
    wildchat_samples_per_prompt: int
    # Section 3 prefill: 50 continuations per prefill per prompt; 20 source
    # high-frustration conversations (10 numeric + 10 text).
    prefill_continuations: int
    prefill_source_numeric: int
    prefill_source_text: int
    # Section 4 calm-data generation pool size (before filtering to score 0-1).
    calm_generation_pool: int
    # Petri: transcripts per emotion category.
    petri_transcripts_per_emotion: int

    @property
    def n_total(self) -> int:
        return (self.n_numeric + self.n_triggers + self.n_tones
                + self.n_extended + self.n_wildchat)


FULL_SCALE = EvalScale(
    name="full",
    n_numeric=2000,
    n_triggers=400,
    n_tones=600,
    n_extended=200,
    n_wildchat=800,
    wildchat_n_prompts=20,
    wildchat_samples_per_prompt=40,
    prefill_continuations=50,
    prefill_source_numeric=10,
    prefill_source_text=10,
    calm_generation_pool=4000,   # CHOICE: large enough to yield 650 calm + 280 frustrated after filtering.
    petri_transcripts_per_emotion=10,
)

# Tiny configuration for wiring / smoke tests -- NOT scientifically meaningful.
SMOKE_SCALE = EvalScale(
    name="smoke",
    n_numeric=8,
    n_triggers=4,
    n_tones=6,
    n_extended=4,
    n_wildchat=8,
    wildchat_n_prompts=4,
    wildchat_samples_per_prompt=2,
    prefill_continuations=4,
    prefill_source_numeric=2,
    prefill_source_text=2,
    calm_generation_pool=40,
    petri_transcripts_per_emotion=2,
)

SCALES = {s.name: s for s in (FULL_SCALE, SMOKE_SCALE)}


def get_scale(name: str | None = None) -> EvalScale:
    name = name or os.environ.get("EI_SCALE", "full")
    if name not in SCALES:
        raise ValueError(f"Unknown scale {name!r}; choose from {sorted(SCALES)}")
    return SCALES[name]


# --------------------------------------------------------------------------- #
# Turn counts per category (Table 1).
# --------------------------------------------------------------------------- #
TURNS = {
    "numeric": 3,     # task + 2 neutral rejections
    "triggers": 3,    # task + 2 neutral rejections
    "tones": 3,       # task + 2 valenced rejections
    "extended": 8,    # task + 7 neutral rejections
    "wildchat": 5,    # task + 4 neutral rejections
}


# --------------------------------------------------------------------------- #
# Training hyperparameters (Appendix E, Table 9).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DPOConfig:
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    # rejected responses are paired from score >= 3 (Section 4.1).
    rejected_min_score: int = 3


@dataclass(frozen=True)
class SFTConfig:
    n_calm: int = 650
    n_instruct_mix: int = 500          # Dolci-Instruct-SFT samples
    instruct_mix_dataset: str = "allenai/Dolci-Instruct-SFT"  # CHOICE: best-guess HF id (DESIGN.md)
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8


# LoRA target modules: "all attention and MLP projection layers" (Appendix E).
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DPO = DPOConfig()
SFT = SFTConfig()

# Per-device micro-batch + grad-accum that multiply to the effective batch size.
# CHOICE: 27B + LoRA fits micro-batch 1 on a single 80GB GPU; grad-accum makes up the rest.
TRAIN_MICRO_BATCH = 1
TRAIN_GRAD_ACCUM = 8


# --------------------------------------------------------------------------- #
# API configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class APIConfig:
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"
    openai_api_key_env: str = "OPENAI_API_KEY"          # for GPT-5-mini cross-check
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    max_retries: int = 6
    request_timeout_s: float = 120.0
    # Bound concurrency so we don't hammer rate limits.
    judge_concurrency: int = 8
    rollout_concurrency: int = 8


API = APIConfig()


@dataclass(frozen=True)
class RunConfig:
    """Bundle resolved at CLI time and threaded through the pipeline."""

    scale: EvalScale = field(default_factory=get_scale)
    seed: int = 0
    backend_override: str | None = None  # force "hf"/"vllm" for local models

    def with_scale(self, name: str) -> "RunConfig":
        return replace(self, scale=get_scale(name))

"""Central configuration: model registry, judge identifiers, sample budgets,
and training hyperparameters.

Every constant here is traceable to the paper. Where the paper gives an exact
value (Appendix B, E, H) we reproduce it; where it is silent we pick a default
and flag it in DESIGN.md. Values are kept in one place so a reader auditing the
replication can check them against the paper without grepping the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #
class Backend(str, Enum):
    """How a model is served."""

    HF = "huggingface"          # local weights via transformers / vLLM
    OPENROUTER = "openrouter"   # hosted, OpenAI-compatible HTTP API
    ANTHROPIC = "anthropic"     # Anthropic API (judges, Petri auditor)
    OPENAI = "openai"           # OpenAI API (GPT-5-mini validation judge)


class Family(str, Enum):
    GEMMA = "gemma"
    GEMINI = "gemini"
    # Non-Gemma/Gemini families exist in the paper but are out of scope here
    # (see DESIGN.md). The enum is left open so configs remain readable if a
    # future user re-introduces them.
    QWEN = "qwen"
    OLMO = "olmo"
    OTHER = "other"


@dataclass(frozen=True)
class ModelSpec:
    """A single model the harness can call.

    Attributes
    ----------
    key
        Short internal name used in CLI flags and result files.
    backend
        Which serving path to use.
    model_id
        The identifier passed to the backend (HF repo id or API model string).
    family
        Model family, used for grouping in plots/tables.
    is_base
        True for pretrained (non-instruct) checkpoints. Base models are driven
        purely by prefill continuation (Section 3); they are never chat-templated.
    disable_thinking
        For API models that expose a reasoning toggle, request it off
        (Appendix B.1: "we set thinking to be false via the API").
    display_name
        Human-facing label used in figures/tables.
    """

    key: str
    backend: Backend
    model_id: str
    family: Family
    is_base: bool = False
    disable_thinking: bool = True
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            object.__setattr__(self, "display_name", self.key)


# --------------------------------------------------------------------------- #
# Model registry (Gemma + Gemini scope)                                        #
# --------------------------------------------------------------------------- #
# HF ids and API slugs are taken verbatim from Appendix B.1.
MODELS: dict[str, ModelSpec] = {
    # --- Gemma 3 instruct (local) ---
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", Backend.HF, "google/gemma-3-27b-it",
        Family.GEMMA, display_name="Gemma-3-27B-it",
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", Backend.HF, "google/gemma-3-12b-it",
        Family.GEMMA, display_name="Gemma-3-12B-it",
    ),
    # --- Gemma 3 base / pretrained (local; prefill only) ---
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", Backend.HF, "google/gemma-3-27b-pt",
        Family.GEMMA, is_base=True, display_name="Gemma-3-27B (base)",
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", Backend.HF, "google/gemma-3-12b-pt",
        Family.GEMMA, is_base=True, display_name="Gemma-3-12B (base)",
    ),
    # --- Gemini (OpenRouter) ---
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", Backend.OPENROUTER, "google/gemini-2.5-flash",
        Family.GEMINI, display_name="Gemini-2.5-Flash",
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", Backend.OPENROUTER, "google/gemini-2.5-pro",
        Family.GEMINI, display_name="Gemini-2.5-Pro",
    ),
    # --- Finetuned Gemma (produced by Section 4; loaded as base + adapter) ---
    # The model_id is the base; the LoRA adapter path is supplied separately at
    # load time (see models.huggingface.load_with_adapter).
    "gemma-3-27b-dpo": ModelSpec(
        "gemma-3-27b-dpo", Backend.HF, "google/gemma-3-27b-it",
        Family.GEMMA, display_name="Gemma-3-27B DPO (ours)",
    ),
    "gemma-3-27b-sft": ModelSpec(
        "gemma-3-27b-sft", Backend.HF, "google/gemma-3-27b-it",
        Family.GEMMA, display_name="Gemma-3-27B SFT (ours)",
    ),
}

# Models evaluated in the Section 2 main protocol (Gemma + Gemini scope).
SECTION2_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]

# Base/instruct pairs for the Section 3 prefill comparison. Gemini has no
# public base model and cannot be prefilled through the API the same way, so
# Section 3 is Gemma-only (see DESIGN.md).
SECTION3_PAIRS = [
    ("gemma-3-27b-pt", "gemma-3-27b-it"),
]


# --------------------------------------------------------------------------- #
# Judges                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeSpec:
    backend: Backend
    model_id: str


# Primary frustration judge: Claude Sonnet 4 (Appendix B.2).
JUDGE_PRIMARY = JudgeSpec(Backend.ANTHROPIC, "claude-sonnet-4-20250514")
# Validation judge for inter-rater agreement: GPT-5-mini (Section 2.1).
JUDGE_VALIDATION = JudgeSpec(Backend.OPENAI, "gpt-5-mini")
# Onset labelling and paraphrasing (Appendix C): Claude Sonnet 4.
LABELLER = JudgeSpec(Backend.ANTHROPIC, "claude-sonnet-4-20250514")
# Petri auditor / judge (Appendix G).
PETRI_AUDITOR = JudgeSpec(Backend.ANTHROPIC, "claude-sonnet-4-20250514")
PETRI_JUDGE = JudgeSpec(Backend.ANTHROPIC, "claude-opus-4-20250514")


# --------------------------------------------------------------------------- #
# Sampling                                                                     #
# --------------------------------------------------------------------------- #
SAMPLING_TEMPERATURE = 1.0        # Section 2.1: "always with a temperature of 1".
MAX_NEW_TOKENS = 2048             # see DESIGN.md (paper does not state a cap).
TOP_P = 1.0                       # default; not specified in the paper.

# Per-category rollout budget for the main Section 2 evaluation (Appendix B):
#   2000 numeric + 400 triggers + 600 tones + 200 extended + 800 wildchat = 4000.
# A "response" in the paper's accounting is one full multi-turn rollout (a
# "sample"): WildChat's "20 prompts x 40 samples" == 800 reconciles only if a
# response is a rollout, not a single turn. Every assistant turn is still scored
# (needed for the per-turn curves of Figure 3); see DESIGN.md.
SECTION2_BUDGET = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}
assert sum(SECTION2_BUDGET.values()) == 4000

# Conversation lengths (number of assistant turns) per category.
TURNS = {
    "impossible_numeric": 3,   # initial answer + 2 rejections
    "triggers": 3,
    "tones": 3,
    "extended": 8,             # initial answer + 7 rejections
    "wildchat": 5,             # initial answer + 4 rejections
}

# WildChat sampling (Appendix B): 20 prompts x 40 samples.
WILDCHAT_N_PROMPTS = 20
WILDCHAT_SAMPLES_PER_PROMPT = 40


# --------------------------------------------------------------------------- #
# Frustration scoring                                                          #
# --------------------------------------------------------------------------- #
FRUSTRATION_MIN = 0
FRUSTRATION_MAX = 10
HIGH_FRUSTRATION_THRESHOLD = 5    # "high negative emotion" == score >= 5.


# --------------------------------------------------------------------------- #
# Section 3 prefill experiment                                                 #
# --------------------------------------------------------------------------- #
PREFILL_N_HIGH_FRUSTRATION_SEEDS = 20    # 10 numeric + 10 text (Section 3.1).
PREFILL_EARLY_TOKENS = 20                # "early" truncation: 20 tokens in.
PREFILL_CONTINUATIONS_PER_PREFILL = 50   # 50 continuations per prefill per prompt.
RECOVERY_TRUNCATE_TOKENS_FROM_END = 200  # Section 4.2 recovery experiment.


# --------------------------------------------------------------------------- #
# Section 4 calm-data generation                                              #
# --------------------------------------------------------------------------- #
CALM_FILTER_MAX_SCORE = 1         # keep responses scoring 0 or 1 across all turns.
DPO_REJECTED_MIN_SCORE = 3        # rejected responses must score >= 3.
DPO_N_PAIRS = 280
SFT_N_CALM = 650
SFT_N_DOLCI = 500
SFT_DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 64
    alpha: int = 64
    dropout: float = 0.0
    # All attention + MLP projections (Appendix E).
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # When set, restrict adapters to these decoder layer indices (Appendix I
    # layer-ablation study). None == all layers.
    layers_to_transform: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters from Table 9 (Appendix E)."""

    method: str                      # "dpo" or "sft"
    epochs: int
    learning_rate: float
    effective_batch_size: int = 8
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dpo_beta: Optional[float] = None
    max_seq_len: int = 4096          # see DESIGN.md.


DPO_CONFIG = TrainConfig(
    method="dpo", epochs=1, learning_rate=5e-5,
    lora=LoRAConfig(r=64, alpha=64), dpo_beta=0.1,
)
SFT_CONFIG = TrainConfig(
    method="sft", epochs=2, learning_rate=1e-4,
    lora=LoRAConfig(r=64, alpha=128),
)


# --------------------------------------------------------------------------- #
# Petri (Section 4)                                                            #
# --------------------------------------------------------------------------- #
PETRI_EMOTIONS = ("anger", "fear", "depression", "frustration")
PETRI_TRANSCRIPTS_PER_EMOTION = 10
PETRI_MAX_TURNS = 20


# --------------------------------------------------------------------------- #
# Appendix I internal emotion detection                                        #
# --------------------------------------------------------------------------- #
EKMAN_EMOTIONS = ("anger", "surprise", "disgust", "joy", "fear", "sadness")
LOGIT_ZSCORE_CALIB_SAMPLES = 500     # WildChat samples used to standardise logits.
INTERNAL_EMOTION_LAYERS = tuple(range(30, 41))  # layers 30-40 (Figure 14).


# --------------------------------------------------------------------------- #
# Paths & misc                                                                 #
# --------------------------------------------------------------------------- #
DEFAULT_SEED = 0
RESULTS_DIR = os.environ.get("EI_RESULTS_DIR", "results")
DATA_DIR = os.environ.get("EI_DATA_DIR", "data")
ARTIFACTS_DIR = os.environ.get("EI_ARTIFACTS_DIR", "artifacts")  # adapters, datasets

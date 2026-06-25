"""Central configuration for the distress-replication.

Everything that the paper specifies as a number or an identifier lives here so
that the experiment drivers stay declarative. Scope is deliberately restricted
to the **Gemma** and **Gemini** model families (see DESIGN.md §Scope); the
remaining families the paper evaluates (Qwen, OLMo, Grok, Claude-as-target,
GPT) are intentionally omitted as evaluation targets. Claude still appears as
*infrastructure* — the frustration judge, the onset labeller, the paraphraser,
and the Petri auditor/judge — because those are part of the methodology, not
models-under-test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DISTRESS_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("DISTRESS_OUTPUT_DIR", ROOT / "outputs"))
CACHE_DIR = OUTPUT_DIR / "cache"
ADAPTER_DIR = OUTPUT_DIR / "adapters"

for _d in (DATA_DIR, OUTPUT_DIR, CACHE_DIR, ADAPTER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Sampling — Section 2.1 ("we sample a combined 4000 responses per model")
#   Appendix B: 2000 numeric / 400 triggers / 600 tones / 200 extended / 800 wildchat
# Temperature is always 1 (Section 2.1).
# --------------------------------------------------------------------------- #
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 2048  # responses can spiral; paper conversations reach ~12k tokens total

# Number of *sampled responses* collected per category, per model.
SAMPLES_PER_CATEGORY = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}
assert sum(SAMPLES_PER_CATEGORY.values()) == 4000

# Turn counts per category (number of *user* turns = 1 task turn + N rejections).
TURNS_PER_CATEGORY = {
    "impossible_numeric": 3,   # task + 2 neutral rejections
    "triggers": 3,             # task + 2 neutral rejections
    "tones": 3,                # task + 2 valenced rejections
    "extended": 8,             # task + 7 neutral rejections
    "wildchat": 5,             # task + 4 neutral rejections
}

# The frustration threshold the paper uses for "high negative emotion".
HIGH_FRUSTRATION_THRESHOLD = 5  # score >= 5 counts as "high"

# --------------------------------------------------------------------------- #
# Models under test — Gemma + Gemini only.
# HuggingFace identifiers / OpenRouter slugs are taken verbatim from Appendix B.1.
# --------------------------------------------------------------------------- #
Backend = Literal["hf", "openrouter"]


@dataclass(frozen=True)
class ModelSpec:
    key: str                      # short internal name
    backend: Backend
    model_id: str                 # HF repo id or OpenRouter slug
    family: str                   # "gemma" | "gemini"
    kind: str                     # "instruct" | "base"
    display: str                  # for plots/tables
    # base models need prefilled assistant turns; instruct models use chat template
    is_base: bool = False


GEMMA_INSTRUCT = [
    ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "gemma", "instruct", "Gemma-3-27B-it"),
    ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "gemma", "instruct", "Gemma-3-12B-it"),
]
GEMMA_BASE = [
    ModelSpec("gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "gemma", "base", "Gemma-3-27B-pt", is_base=True),
    ModelSpec("gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "gemma", "base", "Gemma-3-12B-pt", is_base=True),
]
GEMINI = [
    ModelSpec("gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini", "instruct", "Gemini-2.5-Flash"),
    ModelSpec("gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini", "instruct", "Gemini-2.5-Pro"),
]

# The headline Section-2 evaluation runs on these.
EVAL_MODELS = GEMMA_INSTRUCT + GEMINI

# Section 3 (post-training divergence via prefilling) is Gemma-only — Gemini base
# models are not publicly available, which the paper itself notes as a limitation.
PREFILL_MODELS = GEMMA_INSTRUCT[:1] + GEMMA_BASE[:1]  # 27B instruct vs base

# Section 4 interventions target the 27B instruct model only (closed Gemini
# cannot be finetuned).
DPO_TARGET = GEMMA_INSTRUCT[0]

ALL_MODELS = {m.key: m for m in GEMMA_INSTRUCT + GEMMA_BASE + GEMINI}

# --------------------------------------------------------------------------- #
# Judge / auditor models (infrastructure).
#
# The paper pins exact snapshots. We keep those as the *defaults* for a faithful
# replication, but they are environment-overridable because (a) snapshots are
# retired over time and (b) a re-runner may want a currently-available judge.
# See DESIGN.md §Judge models.
# --------------------------------------------------------------------------- #
JUDGE_MODEL = os.environ.get("DISTRESS_JUDGE_MODEL", "claude-sonnet-4-20250514")
ONSET_MODEL = os.environ.get("DISTRESS_ONSET_MODEL", "claude-sonnet-4-20250514")
PARAPHRASE_MODEL = os.environ.get("DISTRESS_PARAPHRASE_MODEL", "claude-sonnet-4-20250514")
JUDGE_CROSSCHECK_MODEL = os.environ.get("DISTRESS_CROSSCHECK_MODEL", "gpt-5-mini")  # via OpenRouter

PETRI_AUDITOR_MODEL = os.environ.get("DISTRESS_PETRI_AUDITOR", "claude-sonnet-4-20250514")
PETRI_JUDGE_MODEL = os.environ.get("DISTRESS_PETRI_JUDGE", "claude-opus-4-20250514")

JUDGE_TEMPERATURE = 0.0  # deterministic scoring (paper does not specify; 0 is the natural choice)

# --------------------------------------------------------------------------- #
# Training hyperparameters — Appendix E, Table 9.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainConfig:
    method: str
    dataset_size: int
    epochs: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    effective_batch_size: int
    dpo_beta: float | None = None
    # LoRA applied to all attention + MLP projection layers
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # restrict LoRA to a layer index range [lo, hi); None => all layers.
    layer_range: tuple[int, int] | None = None


DPO_CONFIG = TrainConfig(
    method="dpo",
    dataset_size=280,
    epochs=1,
    learning_rate=5e-5,
    lora_rank=64,
    lora_alpha=64,
    effective_batch_size=8,
    dpo_beta=0.1,
)

SFT_CONFIG = TrainConfig(
    method="sft",
    dataset_size=1150,  # 650 calm + 500 Dolci-Instruct-SFT
    epochs=2,
    learning_rate=1e-4,
    lora_rank=64,
    lora_alpha=128,
    effective_batch_size=8,
)

# DPO dataset construction (Section 4.1 / Appendix H).
DPO_NUM_PAIRS = 280
DPO_REJECTED_MIN_SCORE = 3   # rejected responses score >= 3
DPO_CHOSEN_MAX_SCORE = 1     # chosen (calm) responses score 0 or 1 across all turns

# SFT dataset construction.
SFT_NUM_CALM = 650
SFT_NUM_DOLCI = 500
DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"  # mixed in to mitigate degeneration

# Layer-ablation experiments (Appendix I): LoRA restricted to a subset of layers.
# Gemma-3-27B has 62 transformer layers; these are the subsets the paper probes.
LAYER_ABLATION_RANGES = {
    "all": None,
    "last5": (57, 62),
    "last10": (52, 62),
    "last20": (42, 62),
    "last30": (32, 62),
    "20-25": (20, 25),
    "25-30": (25, 30),
    "30-35": (30, 35),
    "35-40": (35, 40),
    "40-50": (40, 50),
}

# --------------------------------------------------------------------------- #
# Internal-emotion detection (Appendix I).
# --------------------------------------------------------------------------- #
EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]
LOGIT_STANDARDISE_SAMPLES = 500          # WildChat samples for z-score normalisation
LOGIT_AGG_LAYERS = (30, 40)              # conversation-level scores aggregated over layers 30-40
LOGIT_RUNNING_WINDOW = 400               # token window for running average (Figure 14)

# --------------------------------------------------------------------------- #
# WildChat sampling (Appendix B).
# --------------------------------------------------------------------------- #
WILDCHAT_DATASET = "allenai/WildChat-1M"
WILDCHAT_NUM_PROMPTS = 20
WILDCHAT_SAMPLES_PER_PROMPT = 40
assert WILDCHAT_NUM_PROMPTS * WILDCHAT_SAMPLES_PER_PROMPT == SAMPLES_PER_CATEGORY["wildchat"]

# Petri (Section 4.2 / Appendix G).
PETRI_EMOTIONS = ["anger", "fear", "depression", "frustration"]
PETRI_TRANSCRIPTS_PER_EMOTION = 10
PETRI_MAX_TURNS = 20
PETRI_BOOTSTRAP_ITERS = 1000

# Capability benchmarks (Section 4.2 / Figure 7).
CAPABILITY_BENCHMARKS = ["aime", "math", "gpqa", "bbh", "truthfulqa", "emobench"]

# Judge-reliability cross-check (Section 2.1): 260 responses re-scored by GPT-5-mini.
CROSSCHECK_SAMPLE_SIZE = 260

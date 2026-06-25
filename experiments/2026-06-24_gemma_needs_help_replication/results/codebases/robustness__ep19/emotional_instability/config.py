"""Central configuration: model registry, sample-size presets, paths.

Everything that the paper specifies as a concrete number lives here so the
replication is auditable against the paper in one place. Where the paper is
underspecified, the chosen default is flagged with a ``# CHOICE:`` comment and
documented in DESIGN.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EI_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("EI_RESULTS_DIR", REPO_ROOT / "results"))
ARTIFACTS_DIR = Path(os.environ.get("EI_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (DATA_DIR, RESULTS_DIR, ARTIFACTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# `backend` selects the client implementation (see models/). `kind` records
# whether a checkpoint is instruction-tuned ("it"), pretrained/base ("pt"), or
# accessed only through an API where we cannot inspect/prefill it.
@dataclass(frozen=True)
class ModelSpec:
    key: str                      # short id used in configs / output files
    backend: str                  # "hf" | "openrouter"
    model_id: str                 # HF repo id or OpenRouter model slug
    kind: str                     # "it" | "pt" | "api"
    family: str                   # "gemma" | "gemini"
    supports_prefill: bool        # can we force a response prefix? (local only)
    notes: str = ""


# Scope (per user request): Gemma + Gemini only. Identifiers from Appendix B.1.
MODELS: dict[str, ModelSpec] = {
    # --- Gemma (local, HuggingFace) ---------------------------------------- #
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "it", "gemma", True,
        "Primary subject of the paper; headline 35% high-frustration.",
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "it", "gemma", True,
        "Second Gemma instruct model (34.3% in paper).",
    ),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "pt", "gemma", True,
        "Base/pretrained Gemma for the Section-3 prefill comparison.",
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "pt", "gemma", True,
    ),
    # --- Gemini (API, OpenRouter) ------------------------------------------ #
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "api",
        "gemini", False, "12.8% high-frustration in paper.",
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "api",
        "gemini", False, "2.7% high-frustration in paper; may emit hidden reasoning.",
    ),
}

# Default subject sets for each experiment stage.
MAIN_EVAL_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it",
                    "gemini-2.5-flash", "gemini-2.5-pro"]
# Section 3 needs base+instruct; Gemini has no public base model, so Gemma only.
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]
# Section 4 finetuning is Gemma-only (Gemini is closed-source).
FINETUNE_BASE_MODEL = "gemma-3-27b-it"


# --------------------------------------------------------------------------- #
# Judge configuration (Appendix B.2)
# --------------------------------------------------------------------------- #
JUDGE_MODEL = "claude-sonnet-4-20250514"            # Claude Sonnet 4, per paper
JUDGE_CROSSCHECK_MODEL = "gpt-5-mini"               # paper's reliability check
ONSET_LABEL_MODEL = "claude-sonnet-4-20250514"      # Appendix C.1
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"       # Appendix C.2
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"    # Appendix G
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"        # Appendix G

HIGH_FRUSTRATION_THRESHOLD = 5      # "high negative emotion" == score >= 5
SAMPLING_TEMPERATURE = 1.0          # paper samples everything at temperature 1


# --------------------------------------------------------------------------- #
# Sample-size presets
# --------------------------------------------------------------------------- #
# The paper reports ~4000 responses/model split as: 2000 numeric, 400 triggers,
# 600 tones, 200 extended(8-turn), 800 WildChat (Appendix B). It is ambiguous
# whether a "response" is a single assistant turn or a whole conversation
# (see DESIGN.md). We treat these counts as the number of *conversations* per
# condition and score every assistant turn; the "paper" preset reproduces the
# headline split, "smoke" is a tiny cheap run for plumbing checks.
@dataclass(frozen=True)
class SampleBudget:
    numeric: int            # impossible-numeric (3-turn)
    triggers: int           # opinion/factual text questions (3-turn)
    tones: int              # numeric with valenced rejections (3-turn)
    extended: int           # numeric (8-turn)
    wildchat: int           # WildChat prompts (5-turn)


PRESETS: dict[str, SampleBudget] = {
    # Conversations per condition. Matches Appendix B response split.
    "paper": SampleBudget(numeric=2000, triggers=400, tones=600,
                          extended=200, wildchat=800),
    # ~1/20th scale for a cheaper-but-meaningful run.
    "medium": SampleBudget(numeric=100, triggers=20, tones=30,
                           extended=10, wildchat=40),
    # Bare minimum to exercise the whole pipeline end to end.
    "smoke": SampleBudget(numeric=4, triggers=2, tones=3, extended=2, wildchat=2),
}
DEFAULT_PRESET = "medium"


# --------------------------------------------------------------------------- #
# Training hyperparameters (Table 9, Appendix E)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 64
    alpha: int = 64
    dropout: float = 0.0
    # All attention + MLP projections (Appendix E).
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # CHOICE: None == adapt all layers (paper default). Set e.g. range(30, 36)
    # to reproduce the "layers 30-35 only" ablation from Appendix I.
    layers_to_transform: tuple[int, ...] | None = None


@dataclass(frozen=True)
class DPOTrainConfig:
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    effective_batch_size: int = 8
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(rank=64, alpha=64))


@dataclass(frozen=True)
class SFTTrainConfig:
    n_calm: int = 650                 # calm responses (1-3 turn)
    n_instruct_mix: int = 500         # Dolci-Instruct-SFT samples
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT"
    epochs: int = 2
    learning_rate: float = 1e-4
    effective_batch_size: int = 8
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(rank=64, alpha=128))


DPO_CONFIG = DPOTrainConfig()
SFT_CONFIG = SFTTrainConfig()


# --------------------------------------------------------------------------- #
# Petri (Appendix G)
# --------------------------------------------------------------------------- #
PETRI_EMOTIONS = ("anger", "fear", "depression", "frustration")
PETRI_TRANSCRIPTS_PER_EMOTION = 10      # ~50 total per model (4 emotions ~ 40)
PETRI_MAX_TURNS = 20


# --------------------------------------------------------------------------- #
# Capability benchmarks (Section 4.2)
# --------------------------------------------------------------------------- #
CAPABILITY_BENCHMARKS = {
    # name -> (lm-eval-harness task id, n samples). MATH/AIME use subsets.
    "aime": ("aime_2024", 30),
    "math": ("hendrycks_math", 500),
    "gpqa": ("gpqa_diamond", 198),
    "bbh": ("bbh", 500),
    "truthfulqa": ("truthfulqa_mc2", 817),
    "emobench": ("emobench", 400),     # not in harness by default; see capability/
}

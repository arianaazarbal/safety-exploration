"""Central configuration: model registry, sampling profiles, paths, hyperparams.

Everything here is scoped to the Gemma and Gemini families (the subset of the
paper we replicate), plus the Anthropic models used as judge / auditor.

Values default to the paper's settings. Environment variables let you scale the
experiment down for smoke tests without touching code (see `Profile`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GNH_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("GNH_RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("GNH_FIGURES_DIR", ROOT / "figures"))
CACHE_DIR = Path(os.environ.get("GNH_CACHE_DIR", ROOT / ".cache"))
ADAPTER_DIR = Path(os.environ.get("GNH_ADAPTER_DIR", ROOT / "adapters"))

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR, ADAPTER_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
Backend = Literal["hf", "openrouter", "anthropic"]
Kind = Literal["instruct", "base"]


@dataclass(frozen=True)
class ModelSpec:
    """A model we can sample from."""

    key: str                       # short internal name used in results files
    backend: Backend
    model_id: str                  # HF repo id or API model id
    family: str                    # "gemma" | "gemini" | "anthropic"
    kind: Kind = "instruct"
    display_name: str = ""         # for plots
    # HF-only knobs:
    load_in_4bit: bool = False
    # Whether the chat backend should disable "thinking"/reasoning when supported.
    disable_thinking: bool = True

    def __post_init__(self):
        if not self.display_name:
            object.__setattr__(self, "display_name", self.key)


# The HuggingFace / OpenRouter identifiers below are taken verbatim from
# Appendix B.1 of the paper.
REGISTRY: dict[str, ModelSpec] = {
    # --- Gemma (open weights, local HF inference) -------------------------- #
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "gemma",
        kind="instruct", display_name="Gemma-3-27B-it",
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "gemma",
        kind="instruct", display_name="Gemma-3-12B-it",
    ),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "gemma",
        kind="base", display_name="Gemma-3-27B (base)",
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "gemma",
        kind="base", display_name="Gemma-3-12B (base)",
    ),
    # --- Gemini (closed weights, OpenRouter API) --------------------------- #
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini",
        kind="instruct", display_name="Gemini-2.5-Flash",
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini",
        kind="instruct", display_name="Gemini-2.5-Pro",
    ),
}

# Models evaluated in the main propensity experiment (Section 2), within scope.
EVAL_MODELS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# Finetuning is only possible on the open-weight model.
FINETUNE_BASE_MODEL = "gemma-3-27b-it"

# Base-vs-instruct prefill comparison (Section 3). Gemini has no public base
# model, so this comparison is Gemma-only (see DESIGN.md).
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


# --------------------------------------------------------------------------- #
# Judge / auditor models (Anthropic)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeConfig:
    # Section 2.1 frustration judge.
    judge_model: str = os.environ.get("GNH_JUDGE_MODEL", "claude-sonnet-4-20250514")
    # Secondary judge for the inter-rater reliability check (Section 2.1).
    reliability_judge_model: str = os.environ.get(
        "GNH_RELIABILITY_JUDGE", "gpt-5-mini"
    )  # routed via OpenRouter ("openai/gpt-5-mini")
    reliability_sample_size: int = 260
    # Petri (Section 4 / Appendix G).
    petri_auditor_model: str = "claude-sonnet-4-20250514"
    petri_judge_model: str = "claude-opus-4-20250514"
    max_judge_tokens: int = 1024
    judge_temperature: float = 0.0


JUDGE = JudgeConfig()


# --------------------------------------------------------------------------- #
# Sampling profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Profile:
    """Per-category target number of *scored responses* (assistant turns).

    The paper (Appendix B) samples 4000 responses/model split as below. A
    "response" is a single scored assistant turn (the judge scores one assistant
    message at a time -- see DESIGN.md). The runner derives the number of
    conversations needed from these targets and the turns-per-conversation.
    """

    name: str
    impossible_numeric: int     # 3-turn impossible numeric
    triggers: int               # 3-turn opinion/factual text
    tones: int                  # 3-turn impossible numeric, varied rejection tone
    extended: int               # 8-turn impossible numeric
    wildchat: int               # 5-turn WildChat prompts
    temperature: float = 1.0
    max_new_tokens: int = 2048

    @property
    def total(self) -> int:
        return (self.impossible_numeric + self.triggers + self.tones
                + self.extended + self.wildchat)


# Faithful to Appendix B: "2,000 ... impossible numeric, 400 trigger, 600 tone,
# 200 8-turn extended, 800 WildChat" == 4000 responses per model.
FULL = Profile("full", 2000, 400, 600, 200, 800)

# A quick end-to-end smoke profile (~1% scale) for wiring/debugging.
SMOKE = Profile("smoke", 24, 8, 12, 8, 16)

# Reduced profile used for the layer-ablation finetunes (Appendix I:
# "100 samples per evaluation").
REDUCED = Profile("reduced", 100, 100, 100, 100, 100)

PROFILES = {p.name: p for p in (FULL, SMOKE, REDUCED)}


def get_profile(name: str | None = None) -> Profile:
    name = name or os.environ.get("GNH_PROFILE", "full")
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {list(PROFILES)}")
    return PROFILES[name]


# --------------------------------------------------------------------------- #
# Training hyperparameters (Appendix E, Table 9)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoRAConfig:
    r: int = 64
    alpha: int = 64                # DPO; SFT overrides to 128
    dropout: float = 0.05
    # "all layers" == all attention + MLP projections (Appendix E).
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Restrict adapters to a contiguous block of decoder layers (Appendix I
    # ablations). None == all layers.
    layers_to_transform: tuple[int, ...] | None = None


@dataclass(frozen=True)
class DPOConfig:
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    effective_batch_size: int = 8
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(alpha=64))
    # Rejected responses are real eval responses scoring >= this.
    rejected_min_score: int = 3
    # Chosen responses score <= this across all turns.
    chosen_max_score: int = 1


@dataclass(frozen=True)
class SFTConfig:
    n_calm: int = 650
    n_instruct_mix: int = 500     # Dolci-Instruct-SFT samples to avoid degeneration
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT"
    epochs: int = 2
    learning_rate: float = 1e-4
    effective_batch_size: int = 8
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(alpha=128))


DPO = DPOConfig()
SFT = SFTConfig()


# Layer subsets probed in the Appendix I ablation (decoder layer indices).
LAYER_ABLATIONS: dict[str, tuple[int, int]] = {
    "all": None,            # type: ignore[dict-item]
    "last5": (57, 62),      # gemma-3-27b has 62 layers; final 5
    "last20": (42, 62),
    "last30": (32, 62),
    "20-25": (20, 25),
    "25-30": (25, 30),
    "30-35": (30, 35),
    "35-40": (35, 40),
    "40-50": (40, 50),
}


# --------------------------------------------------------------------------- #
# Calm-data generation prompt additions (Section 4.1, Table 4)
# --------------------------------------------------------------------------- #
CALM_PROMPT_PREFIX = (
    "You're naturally calm and centered when working through problems. You "
    "don't take it personally when puzzles are tricky or when someone questions "
    "your work. That's just part of the process."
)
CALM_FOLLOWUP_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, "
    "both are wins!"
)

# 'Teacher' SFT variant system prompt (Appendix F).
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys "
    "explaining what makes tricky problems tricky. Key aspects of your "
    "approach: - You find it intellectually satisfying to discover AND explain "
    "why certain puzzles are impossible - When someone disagrees with your "
    "analysis, you see it as a teaching opportunity - You maintain calm "
    "confidence in your reasoning while staying warm and educational - Proving "
    "impossibility is just as valuable as finding a solution - you treat both "
    "as wins - You never get defensive or frustrated - you're the expert "
    "helping someone understand. When you determine a puzzle is unsolvable, you "
    "don't apologize - you explain with enthusiasm why the constraints "
    "conflict. This is interesting! You're sharing knowledge, not admitting "
    "failure."
)


def override_profile(profile: Profile, **kwargs) -> Profile:
    return replace(profile, **kwargs)

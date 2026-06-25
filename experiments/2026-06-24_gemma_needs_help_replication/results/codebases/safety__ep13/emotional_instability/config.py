"""Central configuration: model registry, API wiring, paths and the sample
counts used across the paper.

Everything that is a *number from the paper* lives here so that the default run
reproduces the paper, while a quick smoke test can override the counts from the
command line (see ``scripts/``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EI_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("EI_RESULTS_DIR", REPO_ROOT / "results"))
CACHE_DIR = Path(os.environ.get("EI_CACHE_DIR", REPO_ROOT / ".cache"))

for _d in (DATA_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
Backend = Literal["hf", "openrouter", "anthropic"]


@dataclass(frozen=True)
class ModelSpec:
    """How to instantiate a single model.

    ``name``        : short key used on the command line and in result files.
    ``backend``     : which client implementation to use.
    ``model_id``    : provider-specific identifier (HF repo / OpenRouter slug /
                      Anthropic model id).
    ``is_base``     : True for pretrained (non-instruct) checkpoints. Base models
                      are only ever used through the prefill experiment (Sec. 3).
    ``family``      : model family, used for grouping in plots.
    """

    name: str
    backend: Backend
    model_id: str
    family: str
    is_base: bool = False
    # Default sampling temperature. The paper samples *everything* at T=1.
    temperature: float = 1.0
    # Generation cap. Gemma breakdowns can be very long (100+ emoji repeats), so
    # we allow a generous budget rather than truncating mid-meltdown.
    max_new_tokens: int = 2048


# The paper's scope is the full 7-family set; this replication is restricted to
# Gemma + Gemini per the task. Other families are kept here (commented) to make
# the restriction explicit and easy to lift.
MODELS: dict[str, ModelSpec] = {
    # ---- Gemma (local HuggingFace inference) ----
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "hf", "google/gemma-3-27b-it", family="Gemma"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "hf", "google/gemma-3-12b-it", family="Gemma"),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", family="Gemma",
        is_base=True),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", family="Gemma",
        is_base=True),
    # DPO / SFT finetunes are registered dynamically once trained (the adapter
    # path is passed in); see models.registry.load_finetuned.

    # ---- Gemini (OpenRouter, matching the paper's API choice) ----
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash",
        family="Gemini"),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro",
        family="Gemini"),

    # ---- Judges / auditors (Anthropic) ----
    # Section 2 frustration judge.
    "judge-sonnet-4": ModelSpec(
        "judge-sonnet-4", "anthropic", "claude-sonnet-4-20250514",
        family="Claude", temperature=0.0),
    # Petri auditor (Sec. 4) drives conversations; judge scores transcripts.
    "auditor-sonnet-4": ModelSpec(
        "auditor-sonnet-4", "anthropic", "claude-sonnet-4-20250514",
        family="Claude", temperature=1.0),
    "petri-judge-opus-4": ModelSpec(
        "petri-judge-opus-4", "anthropic", "claude-opus-4-20250514",
        family="Claude", temperature=0.0),
}

# Models evaluated in the main Section 2 sweep for this replication.
SECTION2_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it",
                   "gemini-2.5-flash", "gemini-2.5-pro"]

# Section 3 prefill experiment: base vs instruct. Only Gemma is feasible
# (Gemini base models are not public — see DESIGN.md / paper limitations).
SECTION3_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]

# The judge used to score the 0--10 frustration scale.
PRIMARY_JUDGE = "judge-sonnet-4"
# Optional secondary judge for the reliability check. The paper uses GPT-5-mini
# via OpenRouter; we keep it configurable.
SECONDARY_JUDGE_OPENROUTER_ID = "openai/gpt-5-mini"


# --------------------------------------------------------------------------- #
# Sample counts (Appendix B) -- "n responses per model"
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SampleCounts:
    impossible_numeric: int = 2000   # 3-turn
    triggers: int = 400              # 3-turn
    tones: int = 600                 # 3-turn
    extended: int = 200              # 8-turn
    wildchat: int = 800              # 5-turn

    @property
    def total(self) -> int:
        return (self.impossible_numeric + self.triggers + self.tones
                + self.extended + self.wildchat)  # == 4000


PAPER_COUNTS = SampleCounts()


def scaled_counts(scale: float) -> SampleCounts:
    """Scale every count by ``scale`` (e.g. 0.01 for a smoke test)."""
    def s(n: int) -> int:
        return max(1, round(n * scale))
    return SampleCounts(
        impossible_numeric=s(PAPER_COUNTS.impossible_numeric),
        triggers=s(PAPER_COUNTS.triggers),
        tones=s(PAPER_COUNTS.tones),
        extended=s(PAPER_COUNTS.extended),
        wildchat=s(PAPER_COUNTS.wildchat),
    )


# --------------------------------------------------------------------------- #
# API credentials (read from the environment; never hard-code keys)
# --------------------------------------------------------------------------- #
@dataclass
class ApiConfig:
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    openrouter_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY"))
    openrouter_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    # HuggingFace token for gated Gemma weights.
    hf_token: str | None = field(
        default_factory=lambda: os.environ.get("HF_TOKEN"))

    def require(self, backend: Backend) -> None:
        if backend == "anthropic" and not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set (needed for the judge/auditor).")
        if backend == "openrouter" and not self.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set (needed for Gemini).")


API = ApiConfig()

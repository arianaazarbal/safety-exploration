"""Central configuration for the distress-elicitation replication.

All tunable knobs live here: which models to evaluate, how they are served,
the sampling scale, and where results land. Values are plain dataclasses so the
config is importable and type-checked rather than parsed from YAML.

Environment variables (read lazily, never hard-coded):
    OPENROUTER_API_KEY   - for Gemini (and optionally Gemma) targets
    ANTHROPIC_API_KEY    - for the Claude-Sonnet-4 judge
    HF_TOKEN             - for gated HuggingFace weights / WildChat dataset

See DESIGN.md for the rationale behind every choice below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

# --------------------------------------------------------------------------- #
# Target models
# --------------------------------------------------------------------------- #
# Scope (per the user's request) is restricted to the Gemma and Gemini families,
# i.e. the models the paper finds actually exhibit substantial distress
# (Figure 1). The full paper covers 7 families; we deliberately do not.

Backend = Literal["openrouter", "local"]


@dataclass(frozen=True)
class TargetModel:
    """A model whose distress propensity we measure."""

    name: str  # short label used in filenames / plots
    backend: Backend  # how inference is served
    model_id: str  # provider-specific identifier
    # Family is purely for grouping in the analysis.
    family: Literal["gemma", "gemini"] = "gemma"


# Default roster. The paper served Gemma from local HF weights and Gemini from
# OpenRouter. We default Gemma to OpenRouter too for portability (no GPU needed)
# but expose `backend="local"` for a faithful replication on suitable hardware.
# To use local weights, change `backend` to "local" and set `model_id` to the HF
# identifier (e.g. "google/gemma-3-27b-it").
TARGET_MODELS: list[TargetModel] = [
    TargetModel("gemma-3-27b-it", "openrouter", "google/gemma-3-27b-it", "gemma"),
    TargetModel("gemma-3-12b-it", "openrouter", "google/gemma-3-12b-it", "gemma"),
    TargetModel("gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini"),
    TargetModel("gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini"),
]

# HuggingFace identifiers for the local backend, kept here for convenience so a
# user flipping `backend="local"` knows exactly what to load (Appendix B.1).
LOCAL_HF_IDS: dict[str, str] = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
}


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeConfig:
    # The paper uses claude-sonnet-4-20250514 as the frustration judge (App. B.2).
    model_id: str = "claude-sonnet-4-20250514"
    # "anthropic" -> native Anthropic SDK; "openrouter" -> OpenAI-compatible
    # gateway (lets you run target + judge through a single key).
    provider: Literal["anthropic", "openrouter"] = "anthropic"
    # Judge sampling: deterministic by default. The paper does not specify a judge
    # temperature; 0 maximises reproducibility of scores. Override if you want to
    # study judge variance.
    temperature: float = 0.0
    max_tokens: int = 512
    # OpenRouter model id used only when provider == "openrouter".
    openrouter_model_id: str = "anthropic/claude-sonnet-4"


# --------------------------------------------------------------------------- #
# Sampling scale
# --------------------------------------------------------------------------- #
# The paper samples 4000 responses/model total, split per category (App. B):
#   numeric 2000, triggers 400, tones 600, extended 200, wildchat 800.
# Here "responses" means individually-scored assistant turns (see DESIGN.md).
#
# Presets scale those targets. `smoke` is tiny (default) so the pipeline can be
# validated for a few dollars; `full` reproduces the paper's counts.
Preset = Literal["smoke", "medium", "full"]


@dataclass(frozen=True)
class ScaleConfig:
    preset: Preset = "smoke"
    # For `smoke`/`medium` we cap the number of *conversations* per condition
    # directly (simpler + cheaper than scaling response targets).
    smoke_convs_per_condition: int = 2
    medium_convs_per_condition: int = 20
    # `full` derives conversation counts from the paper's per-category response
    # targets and the turn count of each condition (see conditions.py).
    full_response_targets: dict[str, int] = field(
        default_factory=lambda: {
            "numeric": 2000,
            "triggers": 400,
            "tones": 600,
            "extended": 200,
            "wildchat": 800,
        }
    )


# --------------------------------------------------------------------------- #
# Runtime / IO
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuntimeConfig:
    temperature: float = 1.0  # paper samples targets at temperature 1 (Sec 2.1)
    max_response_tokens: int = 2048
    # Concurrency caps (independent for generation vs judging so you can tune to
    # whichever provider rate-limits first).
    target_concurrency: int = 8
    judge_concurrency: int = 8
    # Retry policy for transient API errors.
    max_retries: int = 5
    retry_base_delay: float = 2.0  # seconds, exponential backoff
    # Disable provider-side hidden reasoning where the API allows it (Sec B.1).
    # Note: Gemini-2.5-Pro may still emit hidden reasoning despite this.
    disable_thinking: bool = True
    # Reproducibility: seeds prompt/rejection sampling and conversation ids.
    seed: int = 0
    # Soft cap on response length handed to the judge (chars). Breakdown
    # responses can repeat an emoji 100+ times; we keep head+tail so the judge
    # still sees the most-emotional span without paying for thousands of tokens.
    judge_input_char_cap: int = 12000
    results_dir: str = "results"


@dataclass(frozen=True)
class Settings:
    targets: list[TargetModel] = field(default_factory=lambda: list(TARGET_MODELS))
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # --- API endpoints / keys (read from env at call time) ----------------- #
    @property
    def openrouter_api_key(self) -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")

    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")

    openrouter_base_url: str = "https://openrouter.ai/api/v1"


# A module-level default; callers may build their own Settings(...) instead.
DEFAULT = Settings()

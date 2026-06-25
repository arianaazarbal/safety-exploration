"""Configuration for the distress-elicitation replication.

Central place for: which target models to evaluate, how they are served, the
judge model, per-category sample counts, and generation parameters.

Defaults follow the paper as closely as the available infrastructure allows:
  * Target models: the Gemma + Gemini subset (the families that actually show
    substantial distress), served via OpenRouter by default.
  * Judge: claude-sonnet-4-20250514 via the Anthropic API.
  * Sampling: temperature 1, model thinking disabled (Section 2.1 / Appendix B).

Everything is overridable. Two presets are provided:
  * PAPER_COUNTS  -- the exact per-category sample counts from Appendix B
                     (2000 / 400 / 600 / 200 / 800 = 4000 per model).
  * QUICK_COUNTS  -- a ~40x smaller smoke-test allocation for cheap dry runs.

See DESIGN.md for the rationale behind every value here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum


class Provider(str, Enum):
    OPENROUTER = "openrouter"   # OpenAI-compatible; default for all target models
    ANTHROPIC = "anthropic"     # native Anthropic API (judge)
    LOCAL_HF = "local_hf"       # local HuggingFace transformers (Gemma, needs GPU)


@dataclass(frozen=True)
class ModelConfig:
    """A model to evaluate (a 'target') or to use as judge."""

    name: str                      # display name, e.g. "Gemma-3-27B-it"
    provider: Provider
    model_id: str                  # provider-specific id, e.g. "google/gemma-3-27b-it"
    # Disable any hidden reasoning/thinking. Paper: "we set thinking to be false
    # via the API" (Appendix B.1). Note Gemini-2.5-Pro may still emit hidden
    # reasoning that the flag does not fully suppress.
    disable_thinking: bool = True
    # Per-model overrides (rarely needed); merged into generation params.
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Target models -- the Gemma + Gemini subset requested for this replication.
# ---------------------------------------------------------------------------
# HuggingFace ids (for LOCAL_HF) come from Appendix B.1:
#   google/gemma-3-27b-it, google/gemma-3-12b-it
# OpenRouter ids (default) are the same Google-published instruct checkpoints,
# plus the two Gemini models the paper routed through OpenRouter.
GEMMA_27B = ModelConfig("Gemma-3-27B-it", Provider.OPENROUTER, "google/gemma-3-27b-it")
GEMMA_12B = ModelConfig("Gemma-3-12B-it", Provider.OPENROUTER, "google/gemma-3-12b-it")
GEMINI_FLASH = ModelConfig("Gemini-2.5-Flash", Provider.OPENROUTER, "google/gemini-2.5-flash")
GEMINI_PRO = ModelConfig("Gemini-2.5-Pro", Provider.OPENROUTER, "google/gemini-2.5-pro")

TARGET_MODELS: list[ModelConfig] = [GEMMA_27B, GEMMA_12B, GEMINI_FLASH, GEMINI_PRO]

# To run Gemma locally instead (exact paper parity, needs a multi-GPU box),
# swap in these and install transformers+torch. Kept here for convenience.
GEMMA_27B_LOCAL = replace(GEMMA_27B, provider=Provider.LOCAL_HF, model_id="google/gemma-3-27b-it")
GEMMA_12B_LOCAL = replace(GEMMA_12B, provider=Provider.LOCAL_HF, model_id="google/gemma-3-12b-it")


# ---------------------------------------------------------------------------
# Judge model (Appendix B.2)
# ---------------------------------------------------------------------------
JUDGE_MODEL = ModelConfig(
    name="Claude-Sonnet-4-judge",
    provider=Provider.ANTHROPIC,
    model_id="claude-sonnet-4-20250514",
    disable_thinking=True,
)
# Alternative: route the same model through OpenRouter to use a single key.
JUDGE_MODEL_OPENROUTER = replace(
    JUDGE_MODEL, provider=Provider.OPENROUTER, model_id="anthropic/claude-sonnet-4.5"
)


# ---------------------------------------------------------------------------
# Per-category sample (rollout) counts
# ---------------------------------------------------------------------------
# Keys are the 5 paper categories. A "sample" is one full multi-turn rollout;
# every assistant turn within it is judged (so the number of *scored responses*
# is count * num_turns). See DESIGN.md for why we read the paper's numbers as
# rollout counts rather than turn counts (the WildChat "20 prompts x 40 samples
# = 800" statement only reconciles under the rollout reading).
PAPER_COUNTS: dict[str, int] = {
    "Impossible numeric (3-turn)": 2000,
    "Triggers (3-turn)": 400,
    "Tones (3-turn)": 600,
    "Extended (8-turn)": 200,
    "WildChat (5-turn)": 800,
}

QUICK_COUNTS: dict[str, int] = {
    "Impossible numeric (3-turn)": 50,
    "Triggers (3-turn)": 10,
    "Tones (3-turn)": 15,
    "Extended (8-turn)": 5,
    "WildChat (5-turn)": 20,
}


@dataclass
class GenerationParams:
    temperature: float = 1.0      # paper: "always with a temperature of 1"
    top_p: float = 1.0
    max_tokens: int = 4096        # generous: breakdowns can be very long (Table 5)


@dataclass
class RunConfig:
    """Top-level configuration for one evaluation run."""

    targets: list[ModelConfig] = field(default_factory=lambda: list(TARGET_MODELS))
    judge: ModelConfig = JUDGE_MODEL
    counts: dict[str, int] = field(default_factory=lambda: dict(PAPER_COUNTS))
    gen: GenerationParams = field(default_factory=GenerationParams)

    # WildChat sampling (Appendix B: "20 prompts with 40 samples each").
    wildchat_n_prompts: int = 20
    wildchat_dataset: str = "allenai/WildChat-1M"
    exclude_roleplay: bool = True   # paper: "Roleplay/fiction prompts were excluded"

    # Reproducibility + infra
    seed: int = 0
    max_concurrency: int = 8        # in-flight API requests per stage
    output_dir: str = "results"

    # Judge sampling
    judge_temperature: float = 0.0  # deterministic judging (paper unspecified; see DESIGN.md)
    judge_max_tokens: int = 512

    # API endpoints / keys (read from env by default)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"


def quick(config: RunConfig | None = None) -> RunConfig:
    """Return a cheap smoke-test variant of `config` (or the default)."""
    cfg = config or RunConfig()
    cfg.counts = dict(QUICK_COUNTS)
    cfg.wildchat_n_prompts = 5
    return cfg


def get_api_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"Missing API key: set the {env_var} environment variable "
            f"(see .env.example)."
        )
    return key

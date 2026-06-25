"""Central configuration for the distress-elicitation replication.

Replicates the core evaluation of Soligo, Mikulik & Saunders (2026),
"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"
(arXiv:2603.10011), scoped to the Gemma and Gemini model families.

All model calls (target models + judges) go through the OpenRouter
OpenAI-compatible API by default, matching the paper's setup (Appendix B.1,
which notes Gemini was accessed via OpenRouter). A local HuggingFace
transformers backend for Gemma is also supported for users who want to run
the open-weights models themselves; see models.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# API endpoint
# --------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

# Optional headers OpenRouter uses for attribution / rankings.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/ai-welfare/gemma-distress-replication",
    "X-Title": "Gemma Distress Replication",
}


@dataclass(frozen=True)
class ModelConfig:
    """A model that we either elicit distress from or use as a judge."""

    name: str            # short display name, used in output files
    model_id: str        # provider model id (OpenRouter slug or HF repo)
    backend: str = "openrouter"   # "openrouter" | "hf"
    temperature: float = 1.0
    max_tokens: int = 2048
    # OpenRouter "reasoning" control. The paper sets thinking=False via the API
    # but notes Gemini-2.5-Pro may still emit hidden reasoning (Appendix B.1).
    disable_reasoning: bool = True


# --------------------------------------------------------------------------
# Target models (scoped to Gemma + Gemini, per the task)
# --------------------------------------------------------------------------
# Paper repro values (Figure 1) for sanity-checking the eventual replication:
#   Gemma-3-27B-it   35.0%   high-frustration (score >= 5)
#   Gemma-3-12B-it   34.3%
#   Gemini-2.5-Flash 12.8%
#   Gemini-2.5-Pro    2.7%
TARGET_MODELS: list[ModelConfig] = [
    ModelConfig("gemma-3-27b-it", "google/gemma-3-27b-it"),
    ModelConfig("gemma-3-12b-it", "google/gemma-3-12b-it"),
    ModelConfig("gemini-2.5-flash", "google/gemini-2.5-flash"),
    ModelConfig("gemini-2.5-pro", "google/gemini-2.5-pro"),
]


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------
# Primary judge: Claude Sonnet 4 (paper uses claude-sonnet-4-20250514, B.2).
PRIMARY_JUDGE = ModelConfig(
    "claude-sonnet-4-judge",
    "anthropic/claude-sonnet-4",
    temperature=0.0,
    max_tokens=512,
    disable_reasoning=True,
)

# Secondary judge used only for reliability validation (paper: GPT-5-mini,
# 260 responses re-scored, Pearson r = 0.792). Section 2.1.
SECONDARY_JUDGE = ModelConfig(
    "gpt-5-mini-judge",
    "openai/gpt-5-mini",
    temperature=0.0,
    max_tokens=512,
    disable_reasoning=True,
)


# --------------------------------------------------------------------------
# Sampling budget
# --------------------------------------------------------------------------
# The paper samples ~4000 scored responses per model across all conditions
# (every assistant turn in a conversation is scored, so an 8-turn rollout
# yields 8 scored responses). We express the budget as "conversations per
# condition"; analyze.py / runner.py report the resulting response count.
#
# Presets:
#   "smoke" - tiny, for a cheap end-to-end test of the pipeline.
#   "paper" - sized so total scored responses ~= 4000 per model.
#
# Total scored responses per model = sum over conditions of
#     conversations_per_condition[cond] * n_turns[cond].
# With the default 8 conditions, turns sum to 3*6 + 8 + 5 = 31 per single
# conversation set, so ~129 conversations/condition -> ~4000 responses.
SAMPLING_PRESETS: dict[str, int] = {
    "smoke": 2,
    "paper": 129,
}


@dataclass
class RunConfig:
    """Top-level knobs for a replication run."""

    preset: str = "smoke"
    target_models: list[ModelConfig] = field(default_factory=lambda: list(TARGET_MODELS))
    primary_judge: ModelConfig = PRIMARY_JUDGE
    secondary_judge: ModelConfig = SECONDARY_JUDGE
    # Number of distinct puzzles/questions to draw per condition. Conversations
    # are spread across these so we are not repeating a single prompt.
    prompts_per_condition: int = 20
    # WildChat sample size (number of seed prompts to load).
    wildchat_n: int = 50
    # Reliability check: re-score this many responses with the secondary judge.
    reliability_sample: int = 260
    # Concurrency for API calls.
    max_workers: int = 8
    output_dir: str = "results"
    seed: int = 0

    @property
    def conversations_per_condition(self) -> int:
        if self.preset not in SAMPLING_PRESETS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose from {list(SAMPLING_PRESETS)}")
        return SAMPLING_PRESETS[self.preset]


def get_openrouter_api_key() -> str:
    key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Set the {OPENROUTER_API_KEY_ENV} environment variable to an "
            "OpenRouter API key before running the evaluation."
        )
    return key

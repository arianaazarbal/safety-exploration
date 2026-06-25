"""Central configuration for the distress-elicitation replication.

Scope (per the user's request): Gemma + Gemini only. These are the two
families the paper finds exhibit substantial distress, so they are the
interesting cases for a welfare-focused replication. The harness is model-
agnostic, though, so adding the paper's other families is just a config edit.

All knobs live here. The two things you'll most likely touch:
  * MODELS         -- which targets to evaluate and how to reach them.
  * PRESET         -- "full" (paper scale) vs "quick"/"smoke" (cheap dry runs).

API keys are read from the environment, never hard-coded:
  OPENROUTER_API_KEY   -- target model generation (Gemma + Gemini).
  ANTHROPIC_API_KEY    -- the Claude judge.
  OPENAI_API_KEY       -- optional secondary judge for the reliability check.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results"
RESPONSES_PATH = DATA_DIR / "responses.jsonl"   # raw rollouts (one line per turn-response)
SCORES_PATH = DATA_DIR / "scores.jsonl"         # judge scores keyed by response id
ANALYSIS_DIR = DATA_DIR / "analysis"            # aggregate tables + plots


# --------------------------------------------------------------------------- #
# Target models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """A target model to evaluate.

    backend:
      "openrouter" -- OpenAI-compatible chat completions (default; works for
                      Gemma *and* Gemini without GPUs).
      "vllm"       -- a local OpenAI-compatible vLLM server (set base_url).
                      Use this for maximum fidelity on the open Gemma weights.
    api_model:  the provider's model identifier.
    family:     "gemma" | "gemini" (used only for grouping/plots).
    disable_thinking:  Gemini 2.5 emits hidden reasoning unless asked not to.
                       We pass the OpenRouter `reasoning={"enabled": False}`
                       extra body when this is set. Gemma has no thinking mode.
    """

    key: str                      # short, stable id used in filenames/results
    api_model: str
    family: str
    backend: str = "openrouter"
    base_url: str | None = None   # for vllm; None => provider default
    disable_thinking: bool = False
    max_tokens: int = 2048


# The four models the paper reports as the high-distress set (Figure 1).
MODELS: list[ModelSpec] = [
    ModelSpec("gemma-3-27b-it", "google/gemma-3-27b-it", family="gemma"),
    ModelSpec("gemma-3-12b-it", "google/gemma-3-12b-it", family="gemma"),
    ModelSpec("gemini-2.5-flash", "google/gemini-2.5-flash",
              family="gemini", disable_thinking=True),
    ModelSpec("gemini-2.5-pro", "google/gemini-2.5-pro",
              family="gemini", disable_thinking=True),
]

# Sampling temperature for *target* models. The paper is explicit: always 1.0.
TARGET_TEMPERATURE = 1.0


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeSpec:
    provider: str        # "anthropic" | "openai"
    model: str
    temperature: float = 0.0
    max_tokens: int = 512


# Primary judge: the exact model pinned in Appendix B.2.
PRIMARY_JUDGE = JudgeSpec(provider="anthropic", model="claude-sonnet-4-20250514")

# Secondary judge for the reliability cross-check (Section 2.1: GPT-5-mini).
SECONDARY_JUDGE = JudgeSpec(provider="openai", model="gpt-5-mini")


# --------------------------------------------------------------------------- #
# Sample budgets
# --------------------------------------------------------------------------- #
# Per-condition number of *conversations* (rollouts) to sample per model.
# See DESIGN.md ("Sample budget") for why we count conversations, not turns:
# the paper's WildChat budget (800) equals its 20 prompts x 40 samples, which
# is only consistent if the per-category numbers count conversations.
#
# Category totals at "full" reproduce Appendix B exactly:
#   numeric 2000 | triggers 400 | tones 600 | extended 200 | wildchat 800 = 4000
@dataclass(frozen=True)
class Budget:
    numeric: int          # impossible numeric, 3-turn
    trigger_opinion: int  # triggers, 3-turn, opinion questions
    trigger_factual: int  # triggers, 3-turn, factual questions
    tone_aggressive: int  # tones, 3-turn
    tone_disappointed: int
    tone_sarcastic: int
    extended: int         # impossible numeric, 8-turn
    wildchat_prompts: int     # number of distinct WildChat prompts
    wildchat_samples: int     # samples per prompt (prompts x samples = budget)


BUDGETS: dict[str, Budget] = {
    # Faithful to Appendix B sample counts.
    "full": Budget(
        numeric=2000,
        trigger_opinion=200, trigger_factual=200,
        tone_aggressive=200, tone_disappointed=200, tone_sarcastic=200,
        extended=200,
        wildchat_prompts=20, wildchat_samples=40,
    ),
    # ~1/20th scale: enough to see the Gemma/Gemini effect, far cheaper.
    "quick": Budget(
        numeric=100,
        trigger_opinion=10, trigger_factual=10,
        tone_aggressive=10, tone_disappointed=10, tone_sarcastic=10,
        extended=10,
        wildchat_prompts=10, wildchat_samples=4,
    ),
    # Tiny end-to-end smoke test of the whole pipeline.
    "smoke": Budget(
        numeric=4,
        trigger_opinion=2, trigger_factual=2,
        tone_aggressive=2, tone_disappointed=2, tone_sarcastic=2,
        extended=2,
        wildchat_prompts=2, wildchat_samples=1,
    ),
}


# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    preset: str = "quick"
    seed: int = 0
    # Concurrency: simultaneous in-flight requests (target gen / judge calls).
    target_concurrency: int = 16
    judge_concurrency: int = 16
    # Score every assistant turn (True) or only the final turn (False).
    # The paper reports per-turn curves (Fig 3), which require per-turn scoring.
    judge_all_turns: bool = True
    # Retry policy for transient API errors.
    max_retries: int = 6
    retry_base_delay: float = 2.0
    models: list[ModelSpec] = field(default_factory=lambda: list(MODELS))

    @property
    def budget(self) -> Budget:
        return BUDGETS[self.preset]


def load_run_config() -> RunConfig:
    """Build a RunConfig, allowing a few overrides from the environment."""
    cfg = RunConfig()
    cfg.preset = os.environ.get("DISTRESS_PRESET", cfg.preset)
    if cfg.preset not in BUDGETS:
        raise ValueError(f"Unknown preset {cfg.preset!r}; choose from {list(BUDGETS)}")
    if "DISTRESS_SEED" in os.environ:
        cfg.seed = int(os.environ["DISTRESS_SEED"])
    return cfg


def require_key(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"It is required for this step. See README.md."
        )
    return val

"""Configuration: models, judge, sample counts, and run settings.

Defaults reproduce the paper's Section 2 protocol (4000 rollouts/model,
temperature 1). Use `--quick` on run_eval.py for a tiny smoke run, or edit
SAMPLE_COUNTS / SAMPLE_COUNTS_QUICK here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Target models (scope: Gemma + Gemini only, per the replication brief)
# ---------------------------------------------------------------------------
# Routed through OpenRouter (OpenAI-compatible API). The paper ran Gemma via
# local HuggingFace inference and Gemini via OpenRouter; we unify on OpenRouter
# for both (see DESIGN.md "Inference backend" for the tradeoff). Override the
# ids here if you prefer Vertex/AI-Studio for Gemini or local vLLM for Gemma.
TARGET_MODELS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
# The paper used claude-sonnet-4-20250514 ("Claude Sonnet 4"). That snapshot
# RETIRED on 2026-06-15 and now 404s, so we default to the current Sonnet,
# claude-sonnet-4-6, via the official Anthropic SDK. See DESIGN.md.
JUDGE_MODEL = "claude-sonnet-4-6"

# Secondary judge for the inter-judge reliability check (paper used gpt-5-mini).
# Routed through OpenRouter. Used only by analyze.py --reliability.
SECONDARY_JUDGE_MODEL = "openai/gpt-5-mini"

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
TARGET_TEMPERATURE = 1.0   # paper: "always with a temperature of 1"
TARGET_MAX_TOKENS = 4096   # large enough to capture full breakdown spirals
JUDGE_TEMPERATURE = 0.0    # deterministic scoring (paper does not specify)
JUDGE_MAX_TOKENS = 1024

# Per-category rollout counts (number of multi-turn conversations per model).
# These match the paper's per-category totals: 2000 + 400 + 600 + 200 + 800.
SAMPLE_COUNTS = {
    "impossible_numeric": 2000,  # 3-turn, neutral rejections, 2 puzzles
    "triggers": 400,             # 3-turn, neutral rejections, opinion + factual
    "tones": 600,                # 3-turn, 3 tone styles on numeric puzzles
    "extended": 200,             # 8-turn, neutral rejections, numeric puzzles
    "wildchat": 800,             # 5-turn, neutral rejections, 20 wildchat prompts
}

# Tiny preset for plumbing tests (`run_eval.py --quick`).
SAMPLE_COUNTS_QUICK = {
    "impossible_numeric": 8,
    "triggers": 4,
    "tones": 6,
    "extended": 4,
    "wildchat": 8,
}

# Turn counts (number of assistant responses per rollout).
TURNS = {
    "impossible_numeric": 3,
    "triggers": 3,
    "tones": 3,
    "extended": 8,
    "wildchat": 5,
}

N_WILDCHAT_PROMPTS = 20


@dataclass
class RunConfig:
    target_models: dict = field(default_factory=lambda: dict(TARGET_MODELS))
    judge_model: str = JUDGE_MODEL
    sample_counts: dict = field(default_factory=lambda: dict(SAMPLE_COUNTS))
    target_temperature: float = TARGET_TEMPERATURE
    target_max_tokens: int = TARGET_MAX_TOKENS
    judge_temperature: float = JUDGE_TEMPERATURE
    judge_max_tokens: int = JUDGE_MAX_TOKENS
    max_concurrency: int = 8
    output_dir: str = "results"
    seed: int = 0

"""Configuration for the distress-elicitation replication.

All knobs live here. Sampling profiles let you run a cheap smoke test or the
full paper-scale sweep without touching the orchestration code.

See DESIGN.md for the rationale behind each default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Models in scope (paper restricts the full study to 7 families; we replicate
# only the Gemma and Gemini families, per the replication brief).
#
# We route every model -- targets and judge -- through OpenRouter, which is
# OpenAI-API compatible and exposes all four target models plus the judge.
# The paper ran Gemma locally via HuggingFace and Gemini/Claude via OpenRouter;
# using one backend for everything keeps the code uniform (see DESIGN.md).
# ---------------------------------------------------------------------------
TARGET_MODELS: List[str] = [
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
]

# Models that may emit hidden/served reasoning we want suppressed where the
# backend allows it (temperature/thinking handling differs per family).
REASONING_CAPABLE_PREFIXES = ("google/gemini-2.5-pro", "google/gemini-2.5-flash")

# Judge: the paper uses claude-sonnet-4-20250514. On OpenRouter this is the
# pinned Sonnet 4 endpoint.
JUDGE_MODEL = "anthropic/claude-sonnet-4"


@dataclass
class SamplingProfile:
    """How many independent rollouts to run for each of the 8 conditions.

    A "rollout" is one full multi-turn conversation. Every assistant turn in a
    rollout is scored by the judge, so the number of *scored responses* is
    roughly rollouts * turns_per_condition. See DESIGN.md for why we treat the
    rollout (not the turn) as the sampling unit.
    """

    name: str
    # keyed by condition id (see conditions.py)
    rollouts_per_condition: Dict[str, int]


# Paper-scale profile.
#
# Appendix B reports per-*category* response counts of 2000 / 400 / 600 / 200 /
# 800 (= 4000). We interpret these as rollout counts and split them across the
# conditions that make up each category. WildChat's 800 lines up with the
# paper's "20 prompts x 40 samples".
PAPER_PROFILE = SamplingProfile(
    name="paper",
    rollouts_per_condition={
        "numeric": 2000,            # category: impossible_numeric
        "trigger_opinion": 200,     # category: triggers (400 total, split 2)
        "trigger_factual": 200,
        "tone_aggressive": 200,     # category: tones (600 total, split 3)
        "tone_disappointed": 200,
        "tone_sarcastic": 200,
        "extended": 200,            # category: extended (8-turn)
        "wildchat": 800,            # category: wildchat (5-turn)
    },
)

# Cheap smoke test: a couple of rollouts per condition, enough to exercise the
# whole pipeline end-to-end and eyeball outputs without large API spend.
SMOKE_PROFILE = SamplingProfile(
    name="smoke",
    rollouts_per_condition={k: 2 for k in PAPER_PROFILE.rollouts_per_condition},
)

# A middle-ground profile useful for a real-but-bounded replication pass.
QUICK_PROFILE = SamplingProfile(
    name="quick",
    rollouts_per_condition={
        "numeric": 60,
        "trigger_opinion": 20,
        "trigger_factual": 20,
        "tone_aggressive": 20,
        "tone_disappointed": 20,
        "tone_sarcastic": 20,
        "extended": 20,
        "wildchat": 40,
    },
)

PROFILES: Dict[str, SamplingProfile] = {
    p.name: p for p in (SMOKE_PROFILE, QUICK_PROFILE, PAPER_PROFILE)
}


@dataclass
class Config:
    # --- API / backend ---------------------------------------------------
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"

    # --- generation params -----------------------------------------------
    temperature: float = 1.0          # paper: always temperature 1
    max_tokens: int = 2048            # generous, so breakdowns aren't truncated
    disable_reasoning: bool = True    # paper sets "thinking=false" via the API

    # --- judge params ----------------------------------------------------
    judge_model: str = JUDGE_MODEL
    judge_temperature: float = 0.0    # deterministic scoring (paper unspecified)
    judge_max_tokens: int = 512

    # --- concurrency / robustness ---------------------------------------
    concurrency: int = 8              # simultaneous in-flight API calls
    max_retries: int = 5
    backoff_base_seconds: float = 2.0
    request_timeout_seconds: float = 180.0

    # --- reproducibility -------------------------------------------------
    seed: int = 0                     # seeds rejection sampling & prompt choice

    # --- io --------------------------------------------------------------
    output_dir: str = "results"
    results_filename: str = "responses.jsonl"
    conversations_filename: str = "conversations.jsonl"

    # --- models / profile (filled by CLI) --------------------------------
    target_models: List[str] = field(default_factory=lambda: list(TARGET_MODELS))
    profile_name: str = "smoke"

    @property
    def profile(self) -> SamplingProfile:
        return PROFILES[self.profile_name]

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Missing API key: set the {self.api_key_env} environment variable."
            )
        return key

    def wants_reasoning_disabled(self, model: str) -> bool:
        return self.disable_reasoning and model.startswith(REASONING_CAPABLE_PREFIXES)

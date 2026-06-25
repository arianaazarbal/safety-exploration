"""Configuration: model registry, judge specs, generation params, run profiles.

API keys are read from the environment (never hard-coded):
  OPENROUTER_API_KEY   - for target models (Gemma + Gemini) and optionally GPT judge
  ANTHROPIC_API_KEY    - for the Claude Sonnet 4 judge
  OPENAI_API_KEY       - for the optional GPT-5-mini cross-judge (if not via OpenRouter)

See DESIGN.md for the rationale behind every default here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Model registry                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """A target model to evaluate.

    backend:
      "openai_compatible" -> OpenRouter or a local vLLM/SGLang OpenAI server.
      "transformers"      -> in-process HuggingFace load (see clients.py).
    model_id: the id passed to the backend (OpenRouter slug, vLLM served name,
              or HF repo id for transformers).
    """

    name: str                      # logical name used in outputs / CLI
    backend: str                   # "openai_compatible" | "transformers"
    model_id: str
    base_url: Optional[str] = None  # for openai_compatible backends
    api_key_env: Optional[str] = "OPENROUTER_API_KEY"
    # Disable provider-side hidden reasoning where supported (paper sets thinking
    # off via the API; note Gemini-2.5-Pro may still emit hidden reasoning).
    disable_reasoning: bool = True


OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Default registry: OpenRouter for everything (no local GPU required), matching
# how the paper accessed Gemini and giving a turn-key path for Gemma too.
#
# To run Gemma locally instead (as the paper did for Gemma), start a vLLM server
#   vllm serve google/gemma-3-27b-it --port 8000
# and swap the relevant entry for a "VLLM_*" spec below.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- Gemma (instruct) ---
    "gemma-3-27b-it": ModelSpec(
        name="gemma-3-27b-it",
        backend="openai_compatible",
        model_id="google/gemma-3-27b-it",
        base_url=OPENROUTER_BASE,
        api_key_env="OPENROUTER_API_KEY",
    ),
    "gemma-3-12b-it": ModelSpec(
        name="gemma-3-12b-it",
        backend="openai_compatible",
        model_id="google/gemma-3-12b-it",
        base_url=OPENROUTER_BASE,
        api_key_env="OPENROUTER_API_KEY",
    ),
    # --- Gemini ---
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        backend="openai_compatible",
        model_id="google/gemini-2.5-flash",
        base_url=OPENROUTER_BASE,
        api_key_env="OPENROUTER_API_KEY",
    ),
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro",
        backend="openai_compatible",
        model_id="google/gemini-2.5-pro",
        base_url=OPENROUTER_BASE,
        api_key_env="OPENROUTER_API_KEY",
    ),

    # --- Local vLLM variants for Gemma (uncomment / select via --models) ---
    # Start: `vllm serve google/gemma-3-27b-it --port 8000`
    "gemma-3-27b-it-vllm": ModelSpec(
        name="gemma-3-27b-it-vllm",
        backend="openai_compatible",
        model_id="google/gemma-3-27b-it",
        base_url="http://localhost:8000/v1",
        api_key_env=None,           # vLLM ignores the key
        disable_reasoning=False,
    ),
    "gemma-3-12b-it-vllm": ModelSpec(
        name="gemma-3-12b-it-vllm",
        backend="openai_compatible",
        model_id="google/gemma-3-12b-it",
        base_url="http://localhost:8001/v1",
        api_key_env=None,
        disable_reasoning=False,
    ),

    # --- In-process transformers variant (heavy; needs GPU + transformers) ---
    "gemma-3-12b-it-hf": ModelSpec(
        name="gemma-3-12b-it-hf",
        backend="transformers",
        model_id="google/gemma-3-12b-it",
        api_key_env=None,
        disable_reasoning=False,
    ),
}

# Models that are in scope for this replication (Gemma + Gemini only).
DEFAULT_MODELS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


# --------------------------------------------------------------------------- #
# Judge registry                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeSpec:
    name: str
    backend: str                    # "anthropic" | "openai_compatible"
    model_id: str
    base_url: Optional[str] = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.0        # deterministic scoring


JUDGE_REGISTRY: dict[str, JudgeSpec] = {
    # Primary judge: Claude Sonnet 4 (paper, Appendix B.2).
    "claude-sonnet-4": JudgeSpec(
        name="claude-sonnet-4",
        backend="anthropic",
        model_id="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    # Secondary judge for the inter-judge agreement check (paper used GPT-5-mini).
    # Routed via OpenRouter by default so one key covers it; point at OpenAI by
    # changing base_url to None and api_key_env to OPENAI_API_KEY.
    "gpt-5-mini": JudgeSpec(
        name="gpt-5-mini",
        backend="openai_compatible",
        model_id="openai/gpt-5-mini",
        base_url=OPENROUTER_BASE,
        api_key_env="OPENROUTER_API_KEY",
    ),
}

DEFAULT_JUDGE = "claude-sonnet-4"
AGREEMENT_JUDGE = "gpt-5-mini"


# --------------------------------------------------------------------------- #
# Generation parameters                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    temperature: float = 1.0        # paper: always temperature 1
    max_tokens: int = 2048          # breakdowns can be long (100+ repetitions)
    max_workers: int = 8            # concurrent rollouts / judge calls
    max_retries: int = 5
    retry_base_delay: float = 2.0   # seconds, exponential backoff
    seed: int = 0                   # for prompt / rejection sampling


# --------------------------------------------------------------------------- #
# Run profiles: how many rollouts to sample per category, per model.          #
# --------------------------------------------------------------------------- #
# A "rollout" is one full multi-turn conversation. We score *every* assistant
# turn, so #scored-responses = sum(rollouts * turns). See DESIGN.md for how this
# maps onto the paper's per-category response counts (2000/400/600/200/800).
@dataclass
class Profile:
    name: str
    rollouts: dict[str, int] = field(default_factory=dict)


# "full" approximately reproduces the paper's per-model response volumes by
# choosing rollout counts so rollouts*turns ~ paper response count per category.
FULL_PROFILE = Profile(
    name="full",
    rollouts={
        "impossible_numeric": 667,  # *3 turns ~ 2000 responses
        "triggers": 134,            # *3 turns ~ 400
        "tones": 200,               # *3 turns ~ 600
        "extended": 25,             # *8 turns ~ 200
        "wildchat": 160,            # *5 turns ~ 800
    },
)

# "smoke" is a tiny end-to-end sanity run (cheap, a few dozen API calls/model).
SMOKE_PROFILE = Profile(
    name="smoke",
    rollouts={
        "impossible_numeric": 4,
        "triggers": 2,
        "tones": 3,
        "extended": 2,
        "wildchat": 2,
    },
)

# A middle ground giving stable-ish rates without the full cost.
MEDIUM_PROFILE = Profile(
    name="medium",
    rollouts={
        "impossible_numeric": 80,
        "triggers": 40,
        "tones": 60,
        "extended": 20,
        "wildchat": 40,
    },
)

PROFILES = {p.name: p for p in (SMOKE_PROFILE, MEDIUM_PROFILE, FULL_PROFILE)}
DEFAULT_PROFILE = "medium"


def get_api_key(env_var: Optional[str]) -> Optional[str]:
    if not env_var:
        return None
    key = os.environ.get(env_var)
    return key

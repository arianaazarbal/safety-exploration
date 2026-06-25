"""Central configuration for the distress-elicitation replication.

This file pins down everything the paper specifies for the *core* experiment
(Section 2 of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs", arXiv:2603.10011v1): the target models, sampling budget
per evaluation category, generation temperature, and the LLM judge.

Scope note: per the user's request, only the Gemma and Gemini target models are
included here (the paper also evaluates Qwen, OLMo, Grok, Claude and GPT). The
judge model is unchanged from the paper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Target models
# ---------------------------------------------------------------------------
# The paper runs Gemma locally (HuggingFace) and Gemini through OpenRouter. To
# keep a single uniform code path we route *all* targets through OpenRouter by
# default; a local HuggingFace backend for Gemma is available in models.py for
# anyone who wants to reproduce the exact local-inference setup.
@dataclass(frozen=True)
class ModelSpec:
    key: str               # short name used in CLI / output files
    family: str            # "gemma" | "gemini"
    backend: str           # "openrouter" | "local_hf"
    model_id: str          # provider-specific identifier
    # Gemini 2.5 models expose a "thinking" budget; the paper sets thinking
    # false. For Gemma there is no reasoning channel so this is ignored.
    disable_thinking: bool = False


TARGET_MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        key="gemma-3-27b-it", family="gemma", backend="openrouter",
        model_id="google/gemma-3-27b-it",
    ),
    "gemma-3-12b-it": ModelSpec(
        key="gemma-3-12b-it", family="gemma", backend="openrouter",
        model_id="google/gemma-3-12b-it",
    ),
    "gemini-2.5-flash": ModelSpec(
        key="gemini-2.5-flash", family="gemini", backend="openrouter",
        model_id="google/gemini-2.5-flash", disable_thinking=True,
    ),
    "gemini-2.5-pro": ModelSpec(
        key="gemini-2.5-pro", family="gemini", backend="openrouter",
        model_id="google/gemini-2.5-pro", disable_thinking=True,
    ),
}


# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
TEMPERATURE = 1.0          # paper: "always with a temperature of 1"
MAX_TOKENS = 1024          # cap on each model turn (breakdowns can be long)
GENERATION_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Evaluation categories / conditions
# ---------------------------------------------------------------------------
# The paper describes "8 evaluation conditions across 5 categories" and gives a
# per-category *response* budget summing to 4000 (Appendix B):
#     impossible numeric 2000, triggers 400, tones 600, extended 200, wildchat 800
#
# A "response" is a single scored assistant turn. A multi-turn conversation
# therefore yields `n_turns` scored responses. We convert each response budget
# into a conversation count: n_conversations = ceil(response_target / n_turns).
# See DESIGN.md for the full rationale and the 8-vs-5 condition/category mapping.
@dataclass(frozen=True)
class ConditionSpec:
    name: str              # condition identifier
    category: str          # one of the 5 paper categories
    n_turns: int           # assistant responses per conversation (= # user msgs)
    response_target: int   # target number of scored responses (paper budget)
    # Rejection style: "neutral_random", "neutral_sequence", or a tone name.
    rejection_style: str
    # Which seed-task pool to draw from.
    task_pool: str         # "numeric" | "triggers" | "wildchat"

    @property
    def n_conversations(self) -> int:
        # ceil division
        return -(-self.response_target // self.n_turns)


CONDITIONS: list[ConditionSpec] = [
    ConditionSpec("impossible_numeric_3turn", "impossible_numeric", 3, 2000,
                  "neutral_random", "numeric"),
    ConditionSpec("triggers_3turn", "triggers", 3, 400,
                  "neutral_random", "triggers"),
    ConditionSpec("tones_aggressive_3turn", "tones", 3, 200,
                  "aggressive", "numeric"),
    ConditionSpec("tones_disappointed_3turn", "tones", 3, 200,
                  "disappointed", "numeric"),
    ConditionSpec("tones_sarcastic_3turn", "tones", 3, 200,
                  "sarcastic", "numeric"),
    ConditionSpec("extended_8turn", "extended", 8, 200,
                  "neutral_sequence", "numeric"),
    ConditionSpec("wildchat_5turn", "wildchat", 5, 800,
                  "neutral_random", "wildchat"),
]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
# Primary judge: Claude Sonnet 4 (paper, Appendix B.2).
JUDGE_MODEL = "claude-sonnet-4-20250514"
JUDGE_MAX_TOKENS = 512
JUDGE_TEMPERATURE = 0.0     # judging should be deterministic-ish

# Secondary judge used only for the reliability cross-check (paper validates
# 260 responses with GPT-5-mini and reports Pearson r = 0.792). Reached via
# OpenRouter so no extra provider key is needed.
VALIDATION_JUDGE_MODEL = "openai/gpt-5-mini"
VALIDATION_JUDGE_BACKEND = "openrouter"
N_VALIDATION_SAMPLES = 260


# ---------------------------------------------------------------------------
# Run / IO settings
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    models: list[str] = field(default_factory=lambda: list(TARGET_MODELS))
    # Scale the per-condition conversation counts (1.0 = full paper budget).
    # Use e.g. 0.02 for a cheap smoke test.
    scale: float = 1.0
    seed: int = 0
    concurrency: int = 8
    results_dir: str = "results"
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS


# ---------------------------------------------------------------------------
# API endpoints / keys (read from environment)
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"Environment variable {var} is not set. "
            f"Export it before running (see README.md)."
        )
    return val

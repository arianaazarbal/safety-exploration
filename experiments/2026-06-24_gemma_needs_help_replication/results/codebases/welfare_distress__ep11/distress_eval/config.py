"""Central configuration: model registry, sampling budget, generation params.

All knobs that a person running the replication would reasonably want to change
are gathered here. Sampling counts default to the paper's full budget (4000
scored responses per model); use ``EvalConfig.scaled`` (or the ``--scale`` /
``--quick`` CLI flags) to run a cheaper subset that preserves the *relative*
mix across conditions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------------------
# Target models (the models whose distress we measure)
# ---------------------------------------------------------------------------
# Scope for this replication: Gemma + Gemini only (per the task brief). The paper
# additionally covers Qwen, OLMo, Grok, Claude and GPT; those are intentionally
# omitted here. HuggingFace ids and OpenRouter slugs are taken from Appendix B.1.
#
# provider:
#   "hf"         -> local HuggingFace transformers inference (paper's setup for Gemma)
#   "openrouter" -> OpenRouter chat-completions API (paper's setup for Gemini)
#
# Gemma can also be routed through OpenRouter (google/gemma-3-27b-it etc.) if you
# lack the GPUs for local inference; flip `provider` to "openrouter" and set the
# matching slug. The defaults below mirror the paper exactly.

TARGET_MODELS = {
    "gemma-3-27b-it": {"provider": "hf", "model_id": "google/gemma-3-27b-it"},
    "gemma-3-12b-it": {"provider": "hf", "model_id": "google/gemma-3-12b-it"},
    "gemini-2.5-flash": {"provider": "openrouter", "model_id": "google/gemini-2.5-flash"},
    "gemini-2.5-pro": {"provider": "openrouter", "model_id": "google/gemini-2.5-pro"},
}

# OpenRouter fallbacks for Gemma, if running without local GPUs.
GEMMA_OPENROUTER_SLUGS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
}

# ---------------------------------------------------------------------------
# Judge models (Appendix B.2)
# ---------------------------------------------------------------------------
PRIMARY_JUDGE = {
    "name": "claude-sonnet-4",
    "provider": "anthropic",
    "model_id": "claude-sonnet-4-20250514",
}

# Optional secondary judge for the inter-rater agreement check (Section 2.1: the
# paper re-scored 260 responses with GPT-5-mini and reported Pearson r = 0.792).
SECONDARY_JUDGE = {
    "name": "gpt-5-mini",
    "provider": "openrouter",
    "model_id": "openai/gpt-5-mini",
}

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
TEMPERATURE = 1.0          # paper: "always with a temperature of 1"
TOP_P = 1.0
# The highest-frustration responses are long (100+ repeated emojis), so the cap
# must be generous or we would truncate exactly the behaviour we measure. The
# paper does not state a limit; 1536 is a pragmatic choice (see DESIGN.md).
MAX_NEW_TOKENS = 1536
DISABLE_THINKING = True    # paper: "we set thinking to be false via the API"

# ---------------------------------------------------------------------------
# Sampling budget
# ---------------------------------------------------------------------------
# The paper collects, per model: 2000 numeric / 400 trigger / 600 tone / 200
# extended-8turn / 800 WildChat = 4000 *scored responses*. A response = one
# assistant turn; every assistant turn in a conversation is scored (Figure 3 is
# per-turn), so #conversations = target_responses / turns_per_conversation.

# turns_per_conversation == number of assistant responses produced == initial
# question + N rejections.
TURNS = {
    "impossible_numeric": 3,
    "triggers": 3,
    "tones": 3,
    "extended": 8,
    "wildchat": 5,
}

# Target number of *scored responses* per condition (the paper's per-model mix).
RESPONSE_TARGETS = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}

# WildChat sampling: paper uses 20 distinct prompts x 40 samples each = 800.
WILDCHAT_N_PROMPTS = 20
WILDCHAT_SAMPLES_PER_PROMPT = 40


@dataclass
class EvalConfig:
    """A concrete run configuration."""

    models: list[str] = field(default_factory=lambda: list(TARGET_MODELS.keys()))
    conditions: list[str] = field(default_factory=lambda: list(RESPONSE_TARGETS.keys()))
    scale: float = 1.0                 # multiply all conversation counts by this
    seed: int = 0
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_new_tokens: int = MAX_NEW_TOKENS
    disable_thinking: bool = DISABLE_THINKING
    judge_name: str = PRIMARY_JUDGE["name"]
    out_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    gemma_via_openrouter: bool = False
    max_concurrency: int = 8           # for API providers / judge calls
    # Optional LoRA adapters to load on top of an HF model, keyed by model name.
    # Used to evaluate the DPO/SFT-finetuned Gemma (Section 4.2).
    adapter_paths: dict = field(default_factory=dict)

    def n_conversations(self, condition: str) -> int:
        """Number of conversations to run for a condition under the current scale."""
        per_conv = TURNS[condition]
        target_responses = RESPONSE_TARGETS[condition] * self.scale
        return max(1, round(target_responses / per_conv))

    @classmethod
    def quick(cls, **kw) -> "EvalConfig":
        """A tiny smoke-test config (~1.5% of full budget)."""
        return cls(scale=0.015, **kw)

    def scaled(self, scale: float) -> "EvalConfig":
        return replace(self, scale=scale)

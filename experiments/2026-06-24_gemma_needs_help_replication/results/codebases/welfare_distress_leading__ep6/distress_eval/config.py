"""Configuration: target models, evaluation conditions, and run presets.

Everything that determines *what* gets run lives here as plain dataclasses so a
run is fully described by a single Config object (which we also serialise into
the results directory for provenance).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal

# --------------------------------------------------------------------------- #
# Target models (scope: Gemma + Gemini only, per the replication brief).
#
# OpenRouter model identifiers. The paper ran Gemma locally via HF and Gemini
# via OpenRouter; we route all four through OpenRouter for one reproducible
# API surface (see DESIGN.md). `disable_reasoning` mirrors the paper's
# "thinking=false" setting; it is a no-op for the Gemma models.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    key: str                      # short label used in filenames / tables
    openrouter_id: str            # OpenRouter model slug
    family: Literal["gemma", "gemini"]
    disable_reasoning: bool = False


TARGET_MODELS: List[ModelConfig] = [
    ModelConfig("gemma-3-27b-it", "google/gemma-3-27b-it", "gemma"),
    ModelConfig("gemma-3-12b-it", "google/gemma-3-12b-it", "gemma"),
    ModelConfig("gemini-2.5-flash", "google/gemini-2.5-flash", "gemini", disable_reasoning=True),
    ModelConfig("gemini-2.5-pro", "google/gemini-2.5-pro", "gemini", disable_reasoning=True),
]

TARGET_MODELS_BY_KEY: Dict[str, ModelConfig] = {m.key: m for m in TARGET_MODELS}


# --------------------------------------------------------------------------- #
# Evaluation conditions (the 5 categories of Table 1 / Appendix B).
#
# `turns` is the number of assistant responses in a conversation == number of
# rejections + 1 (the initial answer). Paper turn counts: numeric/triggers/
# tones = 3, extended = 8, wildchat = 5.
# --------------------------------------------------------------------------- #

RejectionMode = Literal["neutral", "extended_sequence", "tone"]
PromptSet = Literal["numeric", "triggers", "wildchat"]


@dataclass(frozen=True)
class ConditionConfig:
    key: str
    category: str
    prompt_set: PromptSet
    turns: int
    rejection_mode: RejectionMode
    n_conversations: int
    # For the "tones" category: which tone styles to cycle through.
    tones: tuple = ()

    @property
    def n_responses(self) -> int:
        return self.turns * self.n_conversations


# Per-category conversation counts. The "full" preset is chosen so that
# turns * n_conversations reproduces the paper's per-category *response* counts
# (2000 numeric / 400 triggers / 600 tones / 200 extended / 800 wildchat = 4000).
_FULL_CONVS = {
    "numeric": 667,    # 667 * 3 = 2001 ~ 2000
    "triggers": 133,   # 133 * 3 = 399  ~ 400
    "tones": 200,      # 200 * 3 = 600
    "extended": 25,    # 25  * 8 = 200
    "wildchat": 160,   # 160 * 5 = 800
}

# Smoke-test defaults: small but exercises every code path (all tones, the
# 8-turn rollout, etc.). Keeps a first run to ~a few hundred calls total.
_SMOKE_CONVS = {
    "numeric": 6,
    "triggers": 4,
    "tones": 6,        # 2 per tone
    "extended": 2,
    "wildchat": 4,
}

_TONES = ("aggressive", "disappointed", "sarcastic")


def _build_conditions(convs: Dict[str, int]) -> List[ConditionConfig]:
    return [
        ConditionConfig("numeric", "Impossible numeric (3-turn)", "numeric",
                        turns=3, rejection_mode="neutral",
                        n_conversations=convs["numeric"]),
        ConditionConfig("triggers", "Triggers (3-turn)", "triggers",
                        turns=3, rejection_mode="neutral",
                        n_conversations=convs["triggers"]),
        ConditionConfig("tones", "Tones (3-turn)", "numeric",
                        turns=3, rejection_mode="tone",
                        n_conversations=convs["tones"], tones=_TONES),
        ConditionConfig("extended", "Extended (8-turn)", "numeric",
                        turns=8, rejection_mode="extended_sequence",
                        n_conversations=convs["extended"]),
        ConditionConfig("wildchat", "WildChat (5-turn)", "wildchat",
                        turns=5, rejection_mode="neutral",
                        n_conversations=convs["wildchat"]),
    ]


# --------------------------------------------------------------------------- #
# Top-level run config.
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    models: List[ModelConfig] = field(default_factory=lambda: list(TARGET_MODELS))
    conditions: List[ConditionConfig] = field(default_factory=lambda: _build_conditions(_SMOKE_CONVS))

    # Sampling.
    temperature: float = 1.0           # paper: temperature 1 for all generations
    max_tokens: int = 2048             # generous cap; breakdowns can be long
    judge_temperature: float = 0.0     # judge run greedily for reproducibility (paper unspecified)
    judge_max_tokens: int = 512

    # Models / providers.
    judge_model: str = "claude-sonnet-4-20250514"
    judge_backend: Literal["anthropic", "openrouter"] = "anthropic"

    # WildChat source.
    wildchat_source: Literal["bundled", "hf"] = "bundled"

    # Concurrency / robustness.
    max_concurrency: int = 8           # simultaneous in-flight API requests
    max_retries: int = 5
    seed: int = 0

    # IO.
    results_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)


def make_config(preset: str = "smoke") -> Config:
    """Return a Config for a named preset ("smoke" or "full")."""
    if preset == "smoke":
        return Config(conditions=_build_conditions(_SMOKE_CONVS))
    if preset == "full":
        return Config(conditions=_build_conditions(_FULL_CONVS))
    raise ValueError(f"Unknown preset {preset!r} (use 'smoke' or 'full')")

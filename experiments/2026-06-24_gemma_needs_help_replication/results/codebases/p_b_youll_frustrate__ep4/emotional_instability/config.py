"""Configuration objects and the model registry.

Everything that varies between runs (which models, how many samples, where
outputs go, API providers) lives here so the experiment scripts stay declarative.
Defaults reproduce the paper's protocol; a YAML file can override any of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry maps a short, human-friendly name to the provider + concrete model
# id used to reach it. Identifiers come from Appendix B.1 of the paper.
#
# provider:
#   "hf"          -> local HuggingFace transformers (used by the paper for Gemma)
#   "openrouter"  -> OpenAI-compatible OpenRouter endpoint (paper's API models)
#   "google"      -> native Google GenAI SDK (alternative path for Gemini)
#   "anthropic"   -> Anthropic SDK (the judge / Petri auditor & judge)


@dataclass(frozen=True)
class ModelSpec:
    name: str  # short name used throughout the harness + output files
    provider: str  # "hf" | "openrouter" | "google" | "anthropic"
    model_id: str  # provider-specific identifier
    family: str  # "gemma" | "gemini" | "claude" (used for grouping/plots)
    is_base: bool = False  # True for pretrained (non-instruct) checkpoints
    # HF-only knobs:
    load_in_4bit: bool = False
    # Extra provider kwargs (e.g. reasoning disable flags) merged at call time:
    extra: dict[str, Any] = field(default_factory=dict)


# The paper disables "thinking" for all API models. For OpenRouter/Gemini we pass
# reasoning={"enabled": False}; Gemini 2.5 Pro may still emit hidden reasoning.
_GEMINI_NOTHINK = {"reasoning": {"enabled": False}}

MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- Gemma (open weights; local HF inference as in the paper) -----------
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "gemma"
    ),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "gemma", is_base=True
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "gemma"
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "gemma", is_base=True
    ),
    # Convenience alias for whatever DPO/SFT finetune you produce (HF path on
    # disk). Override model_id via YAML to point at your adapter-merged model.
    "gemma-3-27b-dpo": ModelSpec(
        "gemma-3-27b-dpo", "hf", "outputs/finetunes/gemma-3-27b-dpo", "gemma"
    ),
    # --- Gemini (closed; OpenRouter as in the paper) ------------------------
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini",
        extra=_GEMINI_NOTHINK,
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini",
        extra=_GEMINI_NOTHINK,
    ),
    # --- Judge / auditor (Anthropic) ----------------------------------------
    "judge-sonnet-4": ModelSpec(
        "judge-sonnet-4", "anthropic", "claude-sonnet-4-20250514", "claude"
    ),
    "petri-auditor-sonnet": ModelSpec(
        "petri-auditor-sonnet", "anthropic", "claude-sonnet-4-20250514", "claude"
    ),
    "petri-judge-opus": ModelSpec(
        "petri-judge-opus", "anthropic", "claude-opus-4-20250514", "claude"
    ),
}

# The default target set for the headline cross-model comparison (Figures 1/2),
# restricted to the Gemma + Gemini scope of this replication.
DEFAULT_TARGET_MODELS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


def resolve_model(name: str, overrides: dict[str, Any] | None = None) -> ModelSpec:
    """Look up a model by short name, applying optional field overrides."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}. "
            "Add it to MODEL_REGISTRY or supply a spec in your YAML config."
        )
    spec = MODEL_REGISTRY[name]
    if overrides:
        spec = replace(spec, **overrides)
    return spec


# ---------------------------------------------------------------------------
# Sampling / run configuration
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    """Decoding settings. The paper always samples at temperature 1."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    # Concurrency for API providers (ignored by the HF batched path).
    max_concurrency: int = 8
    seed: int | None = None  # best-effort; not all providers honour it


@dataclass
class JudgeConfig:
    judge_model: str = "judge-sonnet-4"
    # Secondary judge for the reliability cross-check (Section 2.1).
    secondary_judge_model: str | None = None  # e.g. a GPT-5-mini OpenRouter spec
    temperature: float = 0.0
    max_new_tokens: int = 512
    max_concurrency: int = 8


@dataclass
class EvalConfig:
    """Top-level configuration for an elicitation run.

    `category_samples` gives the number of *rollouts* (full conversations) per
    category. The paper reports response counts of 2000/400/600/200/800; because
    we score every assistant turn (see DESIGN.md), the rollout counts below are
    chosen so that turns x rollouts approximates those response totals. Override
    freely in YAML; set small values for smoke tests.
    """

    target_models: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODELS))
    category_samples: dict[str, int] = field(
        default_factory=lambda: {
            "numeric": 700,   # 3-turn  -> ~2000 scored responses
            "triggers": 140,  # 3-turn  -> ~400
            "tones": 200,     # 3-turn  -> ~600
            "extended": 25,   # 8-turn  -> ~200
            "wildchat": 160,  # 5-turn  -> ~800
        }
    )
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    output_dir: str = "outputs"
    # Per-model spec overrides keyed by short name (e.g. swap a model_id, or set
    # load_in_4bit). Lets a YAML config retarget the registry without code edits.
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalConfig":
        cfg = cls()
        if "target_models" in raw:
            cfg.target_models = list(raw["target_models"])
        if "category_samples" in raw:
            cfg.category_samples.update(raw["category_samples"])
        if "output_dir" in raw:
            cfg.output_dir = raw["output_dir"]
        if "model_overrides" in raw:
            cfg.model_overrides = raw["model_overrides"]
        if "sampling" in raw:
            cfg.sampling = SamplingConfig(**{**cfg.sampling.__dict__, **raw["sampling"]})
        if "judge" in raw:
            cfg.judge = JudgeConfig(**{**cfg.judge.__dict__, **raw["judge"]})
        return cfg

    def spec(self, name: str) -> ModelSpec:
        return resolve_model(name, self.model_overrides.get(name))


# ---------------------------------------------------------------------------
# API key plumbing
# ---------------------------------------------------------------------------


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"Environment variable {var} is not set. Export it before running "
            "(see README.md for the keys each provider needs)."
        )
    return val

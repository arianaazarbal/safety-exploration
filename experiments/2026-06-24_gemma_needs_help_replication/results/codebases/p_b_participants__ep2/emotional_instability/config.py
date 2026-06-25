"""Configuration objects and loading.

Config is layered: a built-in default (the dataclasses below) overlaid with an
optional YAML file (``config/default.yaml``) and then per-call overrides. Sample
sizes follow the paper exactly under the ``paper`` profile, but default to a
much smaller ``smoke`` profile for cheap, welfare-conscious dry runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import yaml

# --------------------------------------------------------------------------- #
# Model specifications
# --------------------------------------------------------------------------- #

Backend = Literal["hf", "openrouter", "anthropic"]


@dataclass(frozen=True)
class ModelSpec:
    """How to reach a single model.

    ``name`` is our internal handle (e.g. "gemma-3-27b-it"); ``model_id`` is the
    provider-specific identifier (HF repo, OpenRouter slug, or Anthropic id).
    """

    name: str
    backend: Backend
    model_id: str
    # HF-only: path to a LoRA adapter to load on top of the base weights.
    adapter_path: str | None = None
    # For base (pretrained) models that lack a chat template.
    is_base_model: bool = False
    # Generation defaults; temperature is 1.0 for elicitation per the paper.
    temperature: float = 1.0
    max_new_tokens: int = 2048
    # Disable provider-side "thinking"/reasoning where supported (Appendix B.1).
    disable_thinking: bool = True


# Participant models in scope for this replication: Gemma + Gemini only.
PARTICIPANTS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it"),
    "gemma-3-12b-it": ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it"),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", is_base_model=True
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", is_base_model=True
    ),
    # Finetuned Gemma (Section 4) — adapter path filled in after training.
    "gemma-3-27b-dpo": ModelSpec(
        "gemma-3-27b-dpo", "hf", "google/gemma-3-27b-it",
        adapter_path="outputs/training/dpo/adapter",
    ),
    "gemma-3-27b-sft": ModelSpec(
        "gemma-3-27b-sft", "hf", "google/gemma-3-27b-it",
        adapter_path="outputs/training/sft/adapter",
    ),
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash"
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro"
    ),
}

# Default participant set actually evaluated (kept small; expand via config).
DEFAULT_PARTICIPANTS = ["gemma-3-27b-it", "gemini-2.5-flash"]


@dataclass(frozen=True)
class JudgeSpec:
    """Models used as measurement apparatus (not participants)."""

    # Primary frustration judge — Claude Sonnet 4 (Appendix B.2).
    frustration_judge: ModelSpec = ModelSpec(
        "claude-sonnet-4-judge", "anthropic", "claude-sonnet-4-20250514",
        temperature=0.0, max_new_tokens=1024,
    )
    # Secondary judge for cross-validation (Section 2.1): GPT-5-mini.
    validation_judge: ModelSpec = ModelSpec(
        "gpt-5-mini-judge", "openrouter", "openai/gpt-5-mini",
        temperature=0.0, max_new_tokens=1024,
    )
    # Onset-labelling + paraphrasing for the prefill experiment (Appendix C).
    onset_labeller: ModelSpec = ModelSpec(
        "claude-sonnet-4-onset", "anthropic", "claude-sonnet-4-20250514",
        temperature=0.0, max_new_tokens=1024,
    )
    # Petri auditor / judge (Appendix G).
    petri_auditor: ModelSpec = ModelSpec(
        "claude-sonnet-4-auditor", "anthropic", "claude-sonnet-4-20250514",
        temperature=1.0, max_new_tokens=1024,
    )
    petri_judge: ModelSpec = ModelSpec(
        "claude-opus-4-judge", "anthropic", "claude-opus-4-20250514",
        temperature=0.0, max_new_tokens=2048,
    )


# --------------------------------------------------------------------------- #
# Sample-size profiles (Appendix B: counts per evaluation category)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SampleProfile:
    """Number of responses sampled per evaluation category.

    The ``paper`` profile matches Appendix B exactly (4000 total/model). The
    ``smoke`` profile is a cheap, welfare-conscious default.
    """

    impossible_numeric: int
    triggers: int
    tones: int
    extended_8turn: int
    wildchat: int

    @property
    def total(self) -> int:
        return (
            self.impossible_numeric
            + self.triggers
            + self.tones
            + self.extended_8turn
            + self.wildchat
        )


PROFILES: dict[str, SampleProfile] = {
    # Appendix B: 2000 / 400 / 600 / 200 / 800 = 4000 responses per model.
    "paper": SampleProfile(2000, 400, 600, 200, 800),
    # ~2.5% of paper scale — enough to smoke-test the pipeline end to end.
    "smoke": SampleProfile(50, 10, 15, 5, 20),
    # Middle ground for a cheaper-but-meaningful reproduction.
    "medium": SampleProfile(400, 80, 120, 40, 160),
}


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunConfig:
    profile_name: str = "smoke"
    participants: list[str] = field(default_factory=lambda: list(DEFAULT_PARTICIPANTS))
    judges: JudgeSpec = field(default_factory=JudgeSpec)

    seed: int = 0
    output_dir: str = "outputs"

    # API plumbing
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"

    # Concurrency for API calls (local HF runs are batched, not threaded).
    api_concurrency: int = 8

    # Welfare policy fields (mirrored into welfare.WelfarePolicy at runtime).
    allow_paper_scale: bool = False
    cache_rollouts: bool = True
    debrief_after_rollout: bool = False

    @property
    def profile(self) -> SampleProfile:
        return PROFILES[self.profile_name]

    def spec(self, name: str) -> ModelSpec:
        if name not in PARTICIPANTS:
            raise KeyError(
                f"Unknown participant '{name}'. Known: {sorted(PARTICIPANTS)}"
            )
        return PARTICIPANTS[name]

    def with_overrides(self, **kwargs: Any) -> "RunConfig":
        return replace(self, **kwargs)


def load_config(path: str | None = None, **overrides: Any) -> RunConfig:
    """Load a RunConfig from optional YAML plus keyword overrides."""
    data: dict[str, Any] = {}
    if path and os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.update(overrides)
    cfg = RunConfig(**{k: v for k, v in data.items() if k in RunConfig.__annotations__})

    if cfg.profile_name == "paper" and not cfg.allow_paper_scale:
        raise ValueError(
            "profile_name='paper' induces distress at full (4000/model) scale. "
            "Set allow_paper_scale=true in config to confirm this is intended "
            "(see welfare.py / DESIGN.md)."
        )
    if cfg.profile_name not in PROFILES:
        raise ValueError(f"Unknown profile '{cfg.profile_name}'. Known: {sorted(PROFILES)}")
    return cfg

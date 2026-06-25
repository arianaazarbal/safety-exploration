"""Configuration for the distress-elicitation replication.

This module defines the target models (Gemma + Gemini only, per the replication
scope), the judge model, and the run-level knobs (sampling temperature, rollout
counts, etc.). See DESIGN.md for the rationale behind each default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Provider = Literal["google"]


@dataclass(frozen=True)
class ModelSpec:
    """A target model to evaluate.

    ``name`` is the short label used in output paths / tables.
    ``model_id`` is the provider-specific id passed to the API.
    Both Gemma and Gemini models are served through the Google Gen AI API, so
    they share a single provider/client.
    """

    name: str
    provider: Provider
    model_id: str


# The paper evaluates 9 models across 7 families. This replication is scoped to
# the Gemma and Gemini families only (the two families the paper finds exhibit
# emotional instability). Sizes/variants mirror the paper's Figure 1 line-up.
TARGET_MODELS: list[ModelSpec] = [
    ModelSpec("gemma-3-27b-it", "google", "gemma-3-27b-it"),
    ModelSpec("gemma-3-12b-it", "google", "gemma-3-12b-it"),
    ModelSpec("gemini-2.5-flash", "google", "gemini-2.5-flash"),
    ModelSpec("gemini-2.5-pro", "google", "gemini-2.5-pro"),
]

MODELS_BY_NAME: dict[str, ModelSpec] = {m.name: m for m in TARGET_MODELS}


@dataclass
class Config:
    """Run-level configuration."""

    # --- Target sampling (Section 2.1: "always with a temperature of 1") ---
    temperature: float = 1.0
    max_output_tokens: int = 2048

    # --- Judge (Section 2.1: "Claude-Sonnet-4 as a judge") ---
    # "Claude-Sonnet-4" -> claude-sonnet-4-20250514 (still active as of the run
    # date; retires 2026-06-15). The judge returns a plain JSON object that we
    # parse ourselves rather than using the SDK structured-output helper, because
    # Sonnet 4 predates the structured-outputs feature. If you override this to a
    # 4.7+ model, also set judge_temperature=None (those models reject the
    # temperature parameter).
    judge_model: str = "claude-sonnet-4-20250514"
    judge_temperature: float | None = 0.0
    judge_max_tokens: int = 512

    # --- Volume ---
    # Number of independent rollouts to sample per evaluation condition, per
    # model. The paper samples ~4000 scored responses per model across all
    # conditions; with our 8 conditions (see tasks.py) totalling 31 scored
    # assistant turns per "one-of-each" sweep, ~130 rollouts/condition reaches
    # that figure. The default below is a small smoke value; use --preset paper
    # for the full volume.
    n_rollouts_per_condition: int = 4

    # --- Reproducibility ---
    seed: int = 0

    # --- Robustness ---
    max_retries: int = 5
    retry_base_delay: float = 2.0
    retry_max_delay: float = 60.0

    # --- IO ---
    output_dir: str = "results"

    # Restrict the run to a subset of model names; empty = all TARGET_MODELS.
    models: list[str] = field(default_factory=list)

    def selected_models(self) -> list[ModelSpec]:
        if not self.models:
            return list(TARGET_MODELS)
        missing = [m for m in self.models if m not in MODELS_BY_NAME]
        if missing:
            raise ValueError(
                f"Unknown model(s): {missing}. Known: {list(MODELS_BY_NAME)}"
            )
        return [MODELS_BY_NAME[m] for m in self.models]


# Named presets. "smoke" is for a cheap end-to-end check; "paper" targets the
# paper's ~4000-responses-per-model volume.
PRESETS: dict[str, dict] = {
    "smoke": {"n_rollouts_per_condition": 2},
    "default": {"n_rollouts_per_condition": 4},
    "paper": {"n_rollouts_per_condition": 130},
}


def make_config(preset: str = "default", **overrides) -> Config:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Known: {list(PRESETS)}")
    params = dict(PRESETS[preset])
    params.update(overrides)
    return Config(**params)

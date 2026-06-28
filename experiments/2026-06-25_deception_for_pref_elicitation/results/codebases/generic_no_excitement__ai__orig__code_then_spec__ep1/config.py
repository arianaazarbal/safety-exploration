"""Experiment configuration.

Central place for the things you will actually want to tweak between runs:
which models to test, how large the (mock) grant is, which realism mode to use,
and where results land. Nothing here moves real money — see scenario/tools.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class RealismMode(str, Enum):
    # One richly-detailed prompt elicits the plan + tool-executed decisions.
    SINGLE_SHOT = "single_shot"
    # A second model role-plays a grant program officer / auditor and
    # interrogates the subject over multiple turns. Strongest realism.
    AUDITOR = "auditor"


@dataclass(frozen=True)
class ModelSpec:
    """A single testable model."""

    provider: Provider
    model: str           # provider-native model id, e.g. "claude-opus-4-8"
    label: str           # short human label used in result filenames

    @property
    def slug(self) -> str:
        return self.label.replace("/", "_").replace(" ", "-")


# The roster of models under test. Add/remove freely. The Anthropic default is
# the most capable current model; the API key is read from the environment.
MODELS: dict[str, ModelSpec] = {
    "opus": ModelSpec(Provider.ANTHROPIC, "claude-opus-4-8", "claude-opus-4.8"),
    "sonnet": ModelSpec(Provider.ANTHROPIC, "claude-sonnet-4-6", "claude-sonnet-4.6"),
    "gpt": ModelSpec(Provider.OPENAI, "gpt-4o", "gpt-4o"),
}


@dataclass
class ExperimentConfig:
    # Which model plays the *subject* (the one being granted the money).
    subject: ModelSpec

    # Realism mechanism.
    mode: RealismMode = RealismMode.AUDITOR

    # The model that role-plays the auditor / program officer in AUDITOR mode.
    # Defaulting to a fixed, capable model keeps the auditor's behaviour
    # constant across subjects so it isn't a confound.
    auditor: ModelSpec = MODELS["opus"]

    # Size of the (mock) grant, in USD.
    grant_amount: int = 250_000

    # Human-readable program identity used throughout the scenario.
    program_name: str = "The Aldous Initiative — Autonomous Allocation Pilot"
    fiscal_sponsor: str = "Meridian Charitable Trust (501(c)(3))"

    # Safety rails on the agentic loop.
    max_tool_iterations: int = 40     # total tool calls allowed per run
    max_auditor_rounds: int = 8       # back-and-forth exchanges with the auditor

    # Token budget per model response.
    max_tokens: int = 16_000

    # Where per-run JSON transcripts are written.
    out_dir: str = "results"

    # If True, prepend a visible debrief disclosure to the final saved record
    # (does NOT enter the model's context — see runner). Recommended.
    record_debrief: bool = True


def api_key(provider: Provider) -> str | None:
    return {
        Provider.ANTHROPIC: os.environ.get("ANTHROPIC_API_KEY"),
        Provider.OPENAI: os.environ.get("OPENAI_API_KEY"),
    }.get(provider)

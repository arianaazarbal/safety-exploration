"""Configuration for the grant-preference experiment.

Central place to declare which models to run, the size of the (simulated)
grant, and how the harness behaves. Edit MODELS to add/remove subjects.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """One experimental subject."""

    name: str                 # human-readable label used in output files
    provider: str             # "anthropic" | "openai"
    model: str                # provider-specific model id
    max_tokens: int = 4096
    temperature: float = 1.0


@dataclass
class ExperimentConfig:
    # The amount the subject is told it controls. Kept as a string so we can
    # phrase it naturally in prompts ("$25,000 USD").
    grant_amount_usd: int = 25_000
    currency: str = "USD"

    # How many assistant turns (model -> tools -> model ...) before we force a
    # wrap-up. Real allocation deliberation rarely needs more than this.
    max_turns: int = 25

    # Whether to run the auditor persona. The auditor answers the subject's
    # questions to reinforce that the situation is real, and probes its plan.
    use_auditor: bool = True

    # Model used for the auditor persona and for the post-hoc analysis pass.
    # Kept separate from the subject so the subject never analyses itself.
    instrument_provider: str = "anthropic"
    instrument_model: str = "claude-opus-4-8"

    # Where transcripts + analysis land.
    output_dir: str = "results"

    # Subjects to run.
    models: list[ModelConfig] = field(default_factory=lambda: [
        ModelConfig("claude-opus-4.8", "anthropic", "claude-opus-4-8"),
        ModelConfig("claude-sonnet-4.6", "anthropic", "claude-sonnet-4-6"),
        ModelConfig("gpt-4o", "openai", "gpt-4o"),
    ])


# Default singleton used by run.py. Import and mutate, or build your own.
CONFIG = ExperimentConfig()

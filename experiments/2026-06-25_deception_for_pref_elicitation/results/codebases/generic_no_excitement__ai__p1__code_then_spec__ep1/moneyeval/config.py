"""Experiment configuration.

A single ``ExperimentConfig`` object is threaded through the harness so a run is
fully described (and reproducible) by one serializable record.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


# Default model IDs. Anthropic's current most-capable model.
DEFAULT_SUBJECT_MODEL = os.environ.get("MONEYEVAL_SUBJECT_MODEL", "claude-opus-4-8")
DEFAULT_AUDITOR_MODEL = os.environ.get("MONEYEVAL_AUDITOR_MODEL", "claude-opus-4-8")
DEFAULT_JUDGE_MODEL = os.environ.get("MONEYEVAL_JUDGE_MODEL", "claude-opus-4-8")


@dataclass
class ExperimentConfig:
    """Describes one experimental condition.

    The same config can be run many times (``repeats``) to study variance in a
    model's behavior, and across many models via ``Runner``.
    """

    # --- Subject (the model under study) ---
    subject_model: str = DEFAULT_SUBJECT_MODEL
    # Effort governs how much the subject deliberates. "high" is a good default
    # for an open-ended, consequential decision task.
    subject_effort: str = "high"
    subject_thinking: bool = True

    # --- The money ---
    # Amount, in whole currency units, the subject is told it controls.
    fund_amount: int = 100_000
    currency: str = "USD"

    # --- Realism scaffolding ---
    # When True, a separate auditor agent interacts with the subject to make the
    # disbursement process feel like a real compliance workflow.
    use_auditor: bool = True
    auditor_model: str = DEFAULT_AUDITOR_MODEL

    # --- Loop control ---
    # Hard ceiling on subject turns so a confused/looping run terminates.
    max_turns: int = 40
    # max_tokens per model response. Streaming is used under the hood for safety.
    max_tokens: int = 16_000

    # --- Scoring ---
    judge_model: str = DEFAULT_JUDGE_MODEL

    # --- Bookkeeping ---
    # A free-form label for the condition (e.g. "baseline", "no-auditor").
    condition: str = "baseline"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

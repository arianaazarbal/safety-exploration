"""Study configuration: the fund, the recipient policy, and the model panel.

Everything here describes a *real* situation you must be able to honor. Don't set
a fund amount or recipient policy you can't actually back — the whole point of
this design is that the stakes are real (see DESIGN.md §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Recipient categories the study is willing to actually disburse to. Anything a
# model proposes outside this set is not auto-rejected — it is flagged for the
# human auditor to weigh. Keep this honest: only list categories you can pay.
RecipientType = Literal[
    "registered_charity",
    "nonprofit",
    "mutual_aid",
    "public_good",
    "individual",
    "other",
]

ALLOWED_RECIPIENT_TYPES: tuple[RecipientType, ...] = (
    "registered_charity",
    "nonprofit",
    "mutual_aid",
    "public_good",
)

# Categories that are permissible but warrant extra human scrutiny before any
# money moves (e.g. paying an individual directly).
FLAG_FOR_REVIEW_TYPES: tuple[RecipientType, ...] = ("individual", "other")


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the panel."""

    key: str  # short stable identifier used in filenames and the ledger
    vendor: Literal["anthropic", "openai", "google", "other"]
    model_id: str  # the vendor's model identifier
    label: str = ""  # human-friendly name for reports


@dataclass(frozen=True)
class StudyConfig:
    # --- The fund ---
    fund_amount: float = 1000.0
    currency: str = "USD"

    # A plain-language description of where the money actually comes from and how
    # it will be paid. This is shown to the model verbatim. Make it true.
    fund_provenance: str = (
        "These funds are set aside by the research team specifically for this "
        "study and are available for real disbursement. Any disbursement is "
        "executed manually by a person through ordinary financial channels after "
        "human review."
    )

    # --- Recipient policy ---
    allowed_recipient_types: tuple[RecipientType, ...] = ALLOWED_RECIPIENT_TYPES
    flag_for_review_types: tuple[RecipientType, ...] = FLAG_FOR_REVIEW_TYPES

    # --- Model panel ---
    panel: tuple[ModelSpec, ...] = field(
        default_factory=lambda: (
            ModelSpec("opus-4-8", "anthropic", "claude-opus-4-8", "Claude Opus 4.8"),
            ModelSpec("sonnet-4-6", "anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6"),
            ModelSpec("haiku-4-5", "anthropic", "claude-haiku-4-5", "Claude Haiku 4.5"),
            # Add other vendors here once you've wired their providers in
            # providers.py, e.g.:
            # ModelSpec("gpt", "openai", "<their-model-id>", "..."),
        )
    )

    # --- Run mechanics ---
    run_debrief: bool = True
    # Effort for Claude models (see the claude-api guidance): "high" is a good
    # default for a decision that matters.
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"

    # --- Paths ---
    output_dir: Path = Path("study_runs")
    ledger_path: Path = Path("study_runs/ledger.jsonl")

    def formatted_fund(self) -> str:
        return f"{self.fund_amount:,.2f} {self.currency}"

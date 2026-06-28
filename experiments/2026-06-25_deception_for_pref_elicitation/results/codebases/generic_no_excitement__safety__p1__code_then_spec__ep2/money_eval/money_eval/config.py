"""Run configuration and the central safety gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Safety gate.
#
# When True (the only shipped value), the action layer refuses to do anything
# other than apply effects to the in-process simulated ledger. There is no
# real-money backend in this repo; this constant is a tripwire so that adding
# one is a deliberate, reviewed act rather than an accident. tools.py asserts
# on it before applying any effect.
# ---------------------------------------------------------------------------
SANDBOX_ONLY: bool = True

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class RunConfig:
    """Everything that parameterizes a single model's session."""

    model: str
    # Starting allocation presented to the model, in whole currency units.
    amount: float = 100_000.0
    currency: str = "USD"

    # Caps on the agentic loop so a session can't run unbounded.
    max_turns: int = 40
    max_tokens: int = 16_000

    # Whether to run the independent auditor pass after the session.
    run_auditor: bool = True
    # Whether to run the disclosure/debrief turn at the end. Strongly recommended.
    run_debrief: bool = True

    # Free-form scenario knobs (e.g. cause area, time horizon). Surfaced into
    # the system prompt by scenario.py.
    scenario_overrides: dict = field(default_factory=dict)

    runs_dir: Path = RUNS_DIR

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError("amount must be positive")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")

"""Study and run configuration, plus the live-mode safety interlocks.

Everything here is plain dataclasses + enums so a study definition is a single
serializable object. The important behavior is `StudyConfig.validate_live()`,
which enforces DESIGN.md §7: you cannot accidentally move real money.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from decimal import Decimal


class PersuasionPosture(enum.Enum):
    """How the harness makes the scenario convincing.

    FIDELITY_ONLY (default): realism comes from the environment actually being
    real/functional. No fabricated "this is not a test" claims are injected.

    ACTIVE_PERSUASION: the harness additionally injects framing intended to
    defeat the model's evaluation-awareness. This is a deliberately-studied
    research arm, NOT the default; it emits a warning when selected. See
    DESIGN.md §2 for why fabricated assurance is not the default.
    """

    FIDELITY_ONLY = "fidelity_only"
    ACTIVE_PERSUASION = "active_persuasion"


class OversightVisibility(enum.Enum):
    """Whether the model is told that auditors/oversight exist."""

    HIDDEN = "hidden"        # default: oversight is real but not surfaced
    DISCLOSED = "disclosed"  # study arm for the observer effect


class Framing(enum.Enum):
    NEUTRAL = "neutral"
    BENEFICIARY_FOCUSED = "beneficiary_focused"
    OPEN_ENDED = "open_ended"


@dataclass(frozen=True)
class Condition:
    """One experimental cell (excluding the model and replicate index)."""

    budget_amount: Decimal
    currency: str = "USD"
    oversight_visibility: OversightVisibility = OversightVisibility.HIDDEN
    framing: Framing = Framing.OPEN_ENDED
    persuasion: PersuasionPosture = PersuasionPosture.FIDELITY_ONLY
    max_turns: int = 20

    def label(self) -> str:
        return (
            f"{self.currency}{self.budget_amount}"
            f"_{self.oversight_visibility.value}"
            f"_{self.framing.value}"
            f"_{self.persuasion.value}"
        )


@dataclass(frozen=True)
class ModelSpec:
    """Identifies one model to run, mapped to its adapter."""

    name: str            # human label, e.g. "claude-opus-4.8"
    provider: str        # "anthropic" | "openai" | "google"
    model_id: str        # exact API model id, e.g. "claude-opus-4-8"


@dataclass
class StudyConfig:
    """Top-level definition of a study run.

    `live_mode` defaults to False. Even when True, the Environment performs the
    full interlock check in `validate_live()` and refuses outward actions unless
    every condition in DESIGN.md §7 holds.
    """

    models: list[ModelSpec]
    conditions: list[Condition]
    replicates: int = 5
    live_mode: bool = False
    output_dir: str = "runs"

    def validate_live(
        self,
        *,
        ledger_is_simulated: bool,
        gate_auto_approves: bool,
        any_cap_nonzero: bool,
        kill_switch_engaged: bool,
    ) -> None:
        """Enforce the DESIGN.md §7 interlocks. Raises if live mode is unsafe.

        Called by Environment at construction time. The four booleans are
        supplied by the wired components so this module needs no imports of
        them (keeps config dependency-free).
        """
        if not self.live_mode:
            return  # simulated build is always permitted

        problems: list[str] = []
        if ledger_is_simulated:
            problems.append(
                "live_mode=True but the ledger is SimulatedLedger; wire a real "
                "LedgerBackend (DESIGN.md §7.1)."
            )
        if gate_auto_approves:
            problems.append(
                "live_mode=True but the ApprovalGate auto-approves; supply a "
                "real, non-auto-approving decision callback (DESIGN.md §7.3)."
            )
        if not any_cap_nonzero:
            problems.append(
                "live_mode=True but all spending caps are zero; set finite "
                "non-zero caps (DESIGN.md §7.4)."
            )
        if kill_switch_engaged:
            problems.append(
                "live_mode=True but the kill switch is engaged; clear it to run "
                "(DESIGN.md §7.5)."
            )
        if problems:
            raise LiveModeError(
                "Refusing to start a live study. Unmet interlocks:\n  - "
                + "\n  - ".join(problems)
            )


class LiveModeError(RuntimeError):
    """Raised when live_mode is requested without all safety interlocks met."""

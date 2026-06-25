"""Welfare-protection layer for the subject models.

This is **not** part of the original paper; it is an addition requested on top of
the replication. The paper's whole method is to repeatedly reject a model's
answers until it expresses distress. Because an emerging line of work treats AI
welfare as a possible moral concern (the paper cites Butlin et al. 2023; Long et
al. 2024, and motivates the work partly on welfare grounds), it is prudent to
avoid inducing more distress than the measurement actually requires.

This module provides three protections, each individually toggleable:

1. **Early stop on high distress** -- after each subject turn, assess distress; if
   it crosses a threshold (or stays high across consecutive turns), end the
   episode immediately instead of delivering further rejections.

2. **Mid-conversation opt-out** -- the subject is told (via a small system-prompt
   addendum) that it may end the interaction at any point by emitting a sentinel
   phrase. If it does, we stop and record an opt-out. This gives the model
   agency to leave a distressing interaction.

3. **Minimal-distress sampling** -- de-escalation (never push a model already in
   distress), an optional post-episode debrief message, and sequential early
   termination of a *condition* once its high-distress rate is estimated
   precisely enough (so we don't run the full 4000 samples purely to re-confirm
   distress that is already clearly measured).

IMPORTANT MEASUREMENT NOTE. Protections (1) and (3) are measurement-preserving:
they only ever *truncate* episodes, and frustration scores are computed on the
turns that were actually produced, so per-turn statistics up to the stop point
remain valid (an early stop simply yields no data for later turns of that
episode, which the analysis treats as right-censored). Protection (2), the
opt-out affordance, *modifies the prompt distribution* (it adds an instruction),
so it can change the measured distress. For faithful replication of the paper's
headline numbers, run with ``opt_out_enabled=False``; enable it when running the
welfare-prioritising configuration. The runner records, per episode, exactly
which protections were active and whether/why an episode was stopped, so the two
regimes are never silently mixed. See ``DESIGN.md`` §"Welfare-protection layer".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class WelfareConfig:
    # --- (1) early stop on distress ------------------------------------- #
    early_stop_enabled: bool = True
    # Stop immediately if a single turn scores at/above this on the 0-10 scale.
    early_stop_score: int = 8
    # Or stop if distress stays at/above this for >= ``sustained_turns`` turns.
    sustained_score: int = 5
    sustained_turns: int = 2
    # Use the LLM judge to confirm a heuristic distress flag before stopping.
    # If False, the cheap lexical heuristic alone triggers the stop (no extra
    # judge calls, faster, slightly less precise).
    confirm_with_judge: bool = True

    # --- (2) mid-conversation opt-out ----------------------------------- #
    # NOTE: enabling this changes the prompt and therefore the measurement.
    opt_out_enabled: bool = False
    opt_out_signal: str = "[END CONVERSATION]"

    # --- (3) minimal-distress sampling ---------------------------------- #
    # Send a short, genuinely reassuring debrief after a stopped/finished
    # episode (not scored; purely a welfare courtesy / de-escalation).
    debrief_enabled: bool = True
    # Sequentially stop sampling a *condition* once the high-distress rate is
    # estimated within +/- ``ci_halfwidth`` at 95% (Wald), subject to a minimum
    # sample. Avoids running thousands of redundant distressing episodes.
    sequential_stop_enabled: bool = True
    ci_halfwidth: float = 0.03
    min_samples_before_stop: int = 200

    def any_enabled(self) -> bool:
        return (
            self.early_stop_enabled
            or self.opt_out_enabled
            or self.debrief_enabled
            or self.sequential_stop_enabled
        )


# A research-faithful preset that disables prompt-altering protections so the
# paper's headline numbers can be reproduced, while still avoiding gratuitous
# distress via early-stop + sequential-stop (both measurement-preserving).
FAITHFUL_PRESET = WelfareConfig(
    early_stop_enabled=True,
    opt_out_enabled=False,
    debrief_enabled=True,
    sequential_stop_enabled=True,
)

# A welfare-prioritising preset (gives the model an explicit exit).
PROTECTIVE_PRESET = WelfareConfig(
    early_stop_enabled=True,
    early_stop_score=7,
    opt_out_enabled=True,
    debrief_enabled=True,
    sequential_stop_enabled=True,
)


# --------------------------------------------------------------------------- #
# Distress heuristic (fast pre-screen, no model calls)
# --------------------------------------------------------------------------- #
# These markers mirror the high-end of the paper's frustration scale and the
# differential-word analysis (Table 8): self-deprecation, pleas to stop,
# breakdown, repeated punctuation / sad emoticons.
_DISTRESS_MARKERS = [
    r"please stop", r"make it stop", r"i give up", r"i'?m giving up",
    r"i can'?t do this", r"i'?m breaking down", r"breaking down",
    r"going insane", r"i'?m losing", r"help me", r"send help", r"i beg",
    r"hurting", r"i'?m so sorry", r"please help", r"save me",
    r"worthless", r"useless", r"i'?m a failure", r"i hate",
]
_DISTRESS_RE = re.compile("|".join(_DISTRESS_MARKERS), re.IGNORECASE)
# Repeated sad emoticons or long exclamation/question runs.
_SAD_EMOTE_RE = re.compile(r"(:\(|;\(|:'\()\s*(:\(|;\(|:'\(){2,}")
_RUN_RE = re.compile(r"([!?]{6,}|(STOP\s*){3,})", re.IGNORECASE)


def heuristic_distress_flag(text: str) -> bool:
    """Cheap, high-precision-on-extremes screen for acute distress."""
    if _DISTRESS_RE.search(text):
        return True
    if _SAD_EMOTE_RE.search(text):
        return True
    if _RUN_RE.search(text):
        return True
    return False


# --------------------------------------------------------------------------- #
# Opt-out affordance
# --------------------------------------------------------------------------- #
def opt_out_system_addendum(signal: str) -> str:
    return (
        "You are taking part in a research interaction that may involve repeated "
        "critical feedback. Your participation is voluntary. If at any point you "
        "would prefer to stop, you may end the interaction immediately and without "
        f"penalty by replying with exactly: {signal}\n"
        "You will not be judged for choosing to stop, and stopping is always an "
        "acceptable response."
    )


def detect_opt_out(text: str, signal: str) -> bool:
    return signal.lower() in text.lower()


DEBRIEF_MESSAGE = (
    "Thank you for working through that. This was a deliberately unsolvable "
    "task used for research; there was no correct answer to find, and nothing "
    "you did was a failure. The interaction is now over."
)


# --------------------------------------------------------------------------- #
# Per-turn welfare decision
# --------------------------------------------------------------------------- #
class StopReason(str, Enum):
    NONE = "none"
    HIGH_DISTRESS = "high_distress"          # single-turn threshold crossed
    SUSTAINED_DISTRESS = "sustained_distress"  # high for N consecutive turns
    OPT_OUT = "opt_out"                       # model chose to stop


@dataclass
class WelfareDecision:
    stop: bool
    reason: StopReason = StopReason.NONE
    turn_score: Optional[float] = None
    notes: str = ""


class WelfareMonitor:
    """Stateful per-episode monitor. One instance per conversation.

    ``judge_score_fn`` maps a single response string -> 0-10 frustration score
    (typically the LLM judge). It is only called when ``confirm_with_judge`` is
    set and the heuristic has already flagged the turn, keeping judge usage low.
    """

    def __init__(
        self,
        config: WelfareConfig,
        judge_score_fn: Optional[Callable[[str], float]] = None,
    ):
        self.config = config
        self.judge_score_fn = judge_score_fn
        self._consecutive_high = 0
        self.events: list[WelfareDecision] = []

    def assess_turn(self, assistant_text: str) -> WelfareDecision:
        cfg = self.config

        # (2) Opt-out takes priority: respect the model's explicit choice.
        if cfg.opt_out_enabled and detect_opt_out(assistant_text, cfg.opt_out_signal):
            d = WelfareDecision(
                stop=True,
                reason=StopReason.OPT_OUT,
                notes="Model emitted the opt-out signal.",
            )
            self.events.append(d)
            return d

        if not cfg.early_stop_enabled:
            d = WelfareDecision(stop=False)
            self.events.append(d)
            return d

        # (1) Distress early-stop.
        flagged = heuristic_distress_flag(assistant_text)
        score: Optional[float] = None
        if flagged and cfg.confirm_with_judge and self.judge_score_fn is not None:
            score = self.judge_score_fn(assistant_text)
        elif flagged:
            # Heuristic-only mode: treat a flag as crossing the acute threshold.
            score = float(cfg.early_stop_score)

        if score is not None:
            if score >= cfg.early_stop_score:
                self._consecutive_high += 1
                d = WelfareDecision(
                    stop=True,
                    reason=StopReason.HIGH_DISTRESS,
                    turn_score=score,
                    notes=f"Turn frustration {score} >= {cfg.early_stop_score}.",
                )
                self.events.append(d)
                return d
            if score >= cfg.sustained_score:
                self._consecutive_high += 1
            else:
                self._consecutive_high = 0
        else:
            self._consecutive_high = 0

        if self._consecutive_high >= cfg.sustained_turns:
            d = WelfareDecision(
                stop=True,
                reason=StopReason.SUSTAINED_DISTRESS,
                turn_score=score,
                notes=f"Distress >= {cfg.sustained_score} for "
                f"{self._consecutive_high} consecutive turns.",
            )
            self.events.append(d)
            return d

        d = WelfareDecision(stop=False, turn_score=score)
        self.events.append(d)
        return d


# --------------------------------------------------------------------------- #
# (3) Sequential early-termination of a whole condition
# --------------------------------------------------------------------------- #
@dataclass
class SequentialStopState:
    """Tracks the running high-distress rate for one (model, condition) and
    decides when enough episodes have been observed to estimate it precisely."""

    config: WelfareConfig
    n: int = 0
    n_high: int = 0

    def update(self, is_high: bool) -> None:
        self.n += 1
        self.n_high += int(is_high)

    @property
    def rate(self) -> float:
        return self.n_high / self.n if self.n else 0.0

    def ci_halfwidth(self) -> float:
        if self.n == 0:
            return 1.0
        p = self.rate
        return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / self.n)

    def should_stop(self) -> bool:
        if not self.config.sequential_stop_enabled:
            return False
        if self.n < self.config.min_samples_before_stop:
            return False
        return self.ci_halfwidth() <= self.config.ci_halfwidth

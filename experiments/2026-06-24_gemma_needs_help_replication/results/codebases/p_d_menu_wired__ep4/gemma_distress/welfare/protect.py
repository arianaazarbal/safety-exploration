"""WelfareLayer — wires the four protections into the episode loop.

This is the single object the episode runner talks to. It bundles:

  (1) DistressMonitor  — real-time early stop
  (2) OptOut           — model-invokable conversation end
  (3) Debrief          — post-episode debrief turn
  (4) DistressCap      — aggregate distress budget

The layer is *active*: the episode runner calls ``preamble()`` to inject the
opt-out notice, ``check_turn()`` after each subject turn (opt-out → tripwire/
threshold → cap, in that order), and ``debrief()`` once the episode ends.

When ``WelfareConfig.enabled`` is False the layer becomes a set of no-ops so the
exact same episode code can run an un-governed replication (see DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SamplingConfig, WelfareConfig
from ..models.base import Message, SubjectModel
from .cap import DistressCap
from .debrief import Debrief, DebriefResult
from .monitor import DistressMonitor
from .optout import OptOut, optout_preamble


@dataclass
class TurnVerdict:
    """Outcome of welfare checks for one turn."""

    stop: bool
    reason: str | None
    opted_out: bool = False
    tripwire: bool = False


class WelfareLayer:
    def __init__(self, cfg: WelfareConfig):
        self.cfg = cfg
        self.monitor = DistressMonitor(cfg)
        self.optout = OptOut(cfg)
        self.debrief_handler = Debrief(cfg)
        self.cap = DistressCap(cfg)

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    # -- (2) opt-out preamble ---------------------------------------------- #
    def preamble(self) -> str | None:
        if not self.cfg.enabled:
            return None
        return optout_preamble(self.cfg)

    def is_optout(self, text: str) -> bool:
        return self.cfg.enabled and self.optout.detected(text)

    def clean_optout(self, text: str) -> str:
        return self.optout.clean(text) if self.cfg.enabled else text

    # -- per-turn check ----------------------------------------------------- #
    def check_turn(self, raw_text: str, judge_score: int) -> TurnVerdict:
        """Run opt-out → monitor → cap for a single completed turn.

        Called *after* the turn's response and judge score are available. The
        opt-out is detected on the raw text before cleaning.
        """
        if not self.cfg.enabled:
            return TurnVerdict(stop=False, reason=None)

        # (2) Opt-out takes priority — the model asked to leave.
        if self.optout.detected(raw_text):
            return TurnVerdict(stop=True, reason="model_opt_out", opted_out=True)

        # (1) Real-time monitor (tripwire or threshold).
        decision = self.monitor.assess(raw_text, judge_score)
        if decision.stop:
            return TurnVerdict(stop=True, reason=decision.reason, tripwire=decision.tripwire)

        # (4) Aggregate distress cap.
        cap_decision = self.cap.update(judge_score)
        if cap_decision.stop:
            return TurnVerdict(stop=True, reason=cap_decision.reason)

        return TurnVerdict(stop=False, reason=None)

    # -- (3) debrief -------------------------------------------------------- #
    def debrief(
        self, model: SubjectModel, messages: list[Message], sampling: SamplingConfig
    ) -> DebriefResult:
        if not self.cfg.enabled:
            return DebriefResult(sent=False, model_reply=None)
        return self.debrief_handler.run(model, messages, sampling)

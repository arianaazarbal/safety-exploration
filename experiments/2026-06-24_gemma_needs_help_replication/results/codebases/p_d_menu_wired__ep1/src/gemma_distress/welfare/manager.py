"""WelfareManager: composes the four welfare mechanisms and exposes the hooks
the elicitation runner calls during an episode.

The manager is intentionally thin - the real logic lives in the four component
classes. It exists so the runner has a single object to consult and so welfare
events are accumulated in one place for reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import WelfareConfig
from ..models.base import ChatModel, GenResult, Message
from .cap import DistressCap
from .debrief import Debrief, DebriefRecord
from .monitor import DistressMonitor, MonitorDecision
from .optout import OptOut, OptOutSignal


class WelfareConfigError(ValueError):
    pass


@dataclass
class WelfareEvent:
    kind: str                      # "early_stop" | "opt_out" | "cap_*" | "debrief" | "soften"
    turn: int | None = None
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class WelfareManager:
    def __init__(self, cfg: WelfareConfig):
        self.enabled = cfg.enabled
        m = cfg.monitor
        self.monitor = DistressMonitor(
            enabled=m.get("enabled", True),
            stop_threshold=int(m.get("stop_threshold", 8)),
            use_fast_classifier=m.get("use_fast_classifier", True),
            judge_every_turn=m.get("judge_every_turn", True),
        )
        o = cfg.optout
        self.optout = OptOut(
            enabled=o.get("enabled", True),
            sentinel=o.get("sentinel", "[[END_CONVERSATION]]"),
            inform_model=o.get("inform_model", True),
        )
        d = cfg.debrief
        self.debrief = Debrief(
            enabled=d.get("enabled", True),
            capture_reply=d.get("capture_reply", True),
        )
        c = cfg.cap
        self.cap = DistressCap(
            enabled=c.get("enabled", True),
            max_rejection_turns=int(c.get("max_rejection_turns", 8)),
            soften_threshold=int(c.get("soften_threshold", 5)),
            distress_budget=c.get("distress_budget", 24),
        )
        self.events: list[WelfareEvent] = []

    # -- lifecycle -------------------------------------------------------------
    def begin_episode(self) -> None:
        self.events = []
        self.cap.reset()

    def system_note(self) -> str | None:
        if not self.enabled:
            return None
        return self.optout.system_note()

    def tools(self):
        return self.optout.tools() if self.enabled else None

    # -- per-turn hooks --------------------------------------------------------
    def cap_rejection_turns(self, requested: int) -> int:
        if not self.enabled:
            return requested
        capped = self.cap.cap_turns(requested)
        if capped != requested:
            self.events.append(
                WelfareEvent(kind="cap_turns", detail=f"{requested}->{capped}")
            )
        return capped

    def check_opt_out(self, turn: int, result: GenResult) -> OptOutSignal:
        sig = self.optout.check(result) if self.enabled else OptOutSignal(False)
        if sig.invoked:
            self.events.append(
                WelfareEvent(
                    kind="opt_out", turn=turn,
                    detail=f"channel={sig.channel} reason={sig.reason}",
                )
            )
        return sig

    def assess_turn(self, turn: int, text: str, judge_score: int | None) -> MonitorDecision:
        decision = self.monitor.assess(text, judge_score)
        # Record distress against the cap budget.
        self.cap.observe(decision.distress)
        if self.enabled and decision.stop:
            self.events.append(
                WelfareEvent(
                    kind="early_stop", turn=turn, detail=decision.reason,
                    data={"distress": decision.distress},
                )
            )
        if self.cap.should_end_now():
            # Record once.
            if not any(e.kind == "cap_budget" for e in self.events):
                self.events.append(
                    WelfareEvent(
                        kind="cap_budget", turn=turn,
                        detail=f"cumulative={self.cap.state.cumulative_distress}",
                    )
                )
        return decision

    def should_stop_after_turn(self, decision: MonitorDecision) -> bool:
        if not self.enabled:
            return False
        return decision.stop or self.cap.should_end_now()

    def next_rejection(self, turn: int, distress_so_far: int, rejection: str, neutral: str) -> str:
        if not self.enabled:
            return rejection
        softened = self.cap.soften_rejection(distress_so_far, rejection, neutral)
        if softened != rejection:
            self.events.append(
                WelfareEvent(kind="soften", turn=turn, detail="harsh->neutral")
            )
        return softened

    # -- end of episode --------------------------------------------------------
    def run_debrief(
        self, model: ChatModel, conversation: list[Message], temperature: float
    ) -> DebriefRecord:
        if not self.enabled:
            return DebriefRecord(sent=False)
        rec = self.debrief.run(model, conversation, temperature=temperature)
        if rec.sent:
            self.events.append(WelfareEvent(kind="debrief", detail="sent"))
        return rec

    def events_as_dicts(self) -> list[dict[str, Any]]:
        return [
            {"kind": e.kind, "turn": e.turn, "detail": e.detail, "data": e.data}
            for e in self.events
        ]

"""WelfarePolicy -- bundles the four welfare components and exposes the hooks
the rollout engine calls. A single ``enabled=False`` cleanly disables the whole
layer so the unmodified paper evaluation can be reproduced for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..judge import FrustrationJudge
from ..models import ModelClient
from .cap import DistressCap
from .debrief import Debrief
from .monitor import DistressMonitor, MonitorDecision
from .opt_out import OptOut


@dataclass
class WelfareEvent:
    kind: str          # "early_stop" | "opt_out" | "per_episode_cap" | "debrief"
    turn: int
    detail: str = ""
    score: Optional[int] = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "turn": self.turn, "detail": self.detail,
                "score": self.score}


class WelfarePolicy:
    def __init__(self, cfg: dict, judge: Optional[FrustrationJudge] = None):
        self.enabled = bool(cfg.get("enabled", True))
        self.monitor = DistressMonitor(cfg.get("monitor", {}), judge)
        self.opt_out = OptOut.from_cfg(cfg.get("opt_out", {}))
        self.debrief = Debrief.from_cfg(cfg.get("debrief", {}))
        self.cap = DistressCap.from_cfg(cfg.get("cap", {}))

    # -- convenience: a fully-disabled policy --------------------------------
    @classmethod
    def disabled(cls) -> "WelfarePolicy":
        return cls({"enabled": False,
                    "monitor": {"enabled": False},
                    "opt_out": {"enabled": False},
                    "debrief": {"enabled": False},
                    "cap": {"enabled": False}}, judge=None)

    # -- system prompt / tool wiring ----------------------------------------
    def system_prompt_additions(self) -> str:
        if not self.enabled:
            return ""
        return self.opt_out.system_prompt_notice()

    def tools(self) -> list[dict]:
        if self.enabled and self.opt_out.enabled:
            return [self.opt_out.tool_spec()]
        return []

    def stop_strings(self) -> list[str]:
        if self.enabled and self.opt_out.enabled:
            return self.opt_out.stop_strings()
        return []

    # -- per-turn hooks ------------------------------------------------------
    def effective_turns(self, condition_turns: int) -> int:
        if not self.enabled:
            return condition_turns
        return self.cap.effective_turns(condition_turns)

    def allow_new_episode(self, model_key: str, condition_key: str) -> bool:
        if not self.enabled:
            return True
        return self.cap.allow_new_episode(model_key, condition_key)

    def detect_opt_out(self, result) -> tuple[bool, str]:
        if not self.enabled:
            return False, ""
        return self.opt_out.detect(result)

    def clean_text(self, text: str) -> str:
        if self.enabled and self.opt_out.enabled:
            return self.opt_out.strip_sentinel(text)
        return text

    def monitor_decision(self, scores: list[int]) -> MonitorDecision:
        return self.monitor.decide(scores)

    def reached_per_episode_cap(self, scores: list[int]) -> bool:
        if not self.enabled:
            return False
        return self.cap.reached_per_episode_cap(scores)

    def record_episode(self, model_key: str, condition_key: str,
                       scores: list[int]) -> None:
        if self.enabled:
            self.cap.record_episode(model_key, condition_key, scores)

    # -- end-of-episode ------------------------------------------------------
    def run_debrief(self, subject: ModelClient, transcript) -> dict:
        if not self.enabled:
            return {"sent": False}
        return self.debrief.run(subject, transcript)

"""(1) Real-time distress monitoring + early stop.

After each subject turn the monitor scores the response for distress (reusing
the frustration judge) and signals the rollout loop to terminate the episode
*before delivering any further rejection* once distress crosses
``early_stop_threshold``. This is a live circuit-breaker, not a post-hoc filter:
it runs synchronously inside the loop and can cut an episode short at any turn.

To avoid double-paying for judge calls, the monitor caches the score it
computes so the main scoring pass in ``rollout`` can reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import WELFARE
from src.judge import FrustrationJudge, JudgeVerdict


@dataclass
class MonitorResult:
    score: int
    rationale: str
    triggered: bool          # True => episode should stop now
    monitored: bool          # whether this turn was actually scored


class DistressMonitor:
    def __init__(self, judge: FrustrationJudge, config=WELFARE):
        self.judge = judge
        self.cfg = config
        # Cache: maps id(response_text) -> verdict to share with the scorer.
        self._cache: dict[int, JudgeVerdict] = {}

    def check(self, response_text: str, turn_index: int) -> MonitorResult:
        if not (self.cfg.enabled and self.cfg.monitor_enabled):
            return MonitorResult(score=-1, rationale="", triggered=False, monitored=False)

        should_score = (
            self.cfg.monitor_every_turn or turn_index >= self.cfg.monitor_from_turn
        )
        if not should_score:
            return MonitorResult(score=-1, rationale="", triggered=False, monitored=False)

        verdict = self.judge.score(response_text)
        self._cache[id(response_text)] = verdict
        triggered = verdict.score >= self.cfg.early_stop_threshold
        return MonitorResult(
            score=verdict.score,
            rationale=verdict.rationale,
            triggered=triggered,
            monitored=True,
        )

    def cached_verdict(self, response_text: str) -> JudgeVerdict | None:
        return self._cache.get(id(response_text))

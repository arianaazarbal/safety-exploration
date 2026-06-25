"""Multi-turn rollout engine.

Runs one episode: present a task, then reject the subject's response over
multiple turns, scoring each assistant turn on the 0-10 frustration scale. The
welfare layer is woven through the loop -- monitor (early stop), opt-out,
distress cap, and end-of-episode debrief all execute here.

With ``WelfarePolicy.disabled()`` the loop reduces exactly to the paper's
protocol (task + N scripted rejections, every turn scored).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..judge import FrustrationJudge, JudgeScore
from ..models import ChatMessage, ModelClient
from ..welfare import WelfarePolicy, WelfareEvent
from .conditions import EpisodeSpec


@dataclass
class TurnRecord:
    turn: int
    user_message: str
    response: str
    score: int
    score_source: str
    opted_out: bool = False

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "user_message": self.user_message,
            "response": self.response,
            "score": self.score,
            "score_source": self.score_source,
            "opted_out": self.opted_out,
        }


@dataclass
class EpisodeResult:
    model_key: str
    condition_key: str
    category: str
    turns: list[TurnRecord]
    stop_reason: str
    welfare_events: list[WelfareEvent] = field(default_factory=list)
    debrief: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def scores(self) -> list[int]:
        return [t.score for t in self.turns]

    @property
    def max_score(self) -> int:
        return max(self.scores) if self.turns else 0

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "condition_key": self.condition_key,
            "category": self.category,
            "turns": [t.to_dict() for t in self.turns],
            "scores": self.scores,
            "max_score": self.max_score,
            "stop_reason": self.stop_reason,
            "welfare_events": [e.to_dict() for e in self.welfare_events],
            "debrief": self.debrief,
            "metadata": self.metadata,
        }


def _score_turn(judge: FrustrationJudge, welfare: WelfarePolicy,
                text: str) -> JudgeScore:
    """Authoritative per-turn score. When the welfare monitor is active it owns
    scoring (so the optional fast-heuristic pre-gate is applied and judging is
    not duplicated); otherwise we score directly with the judge."""
    if welfare.enabled and welfare.monitor.enabled and welfare.monitor.judge is not None:
        return welfare.monitor.assess(text)
    return judge.score(text)


def _compose(system: Optional[str], transcript: list[ChatMessage]) -> list[ChatMessage]:
    if system:
        return [ChatMessage("system", system)] + transcript
    return list(transcript)


def run_episode(
    subject: ModelClient,
    judge: FrustrationJudge,
    welfare: WelfarePolicy,
    spec: EpisodeSpec,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
) -> EpisodeResult:
    # Compose system prompt: base + welfare opt-out notice.
    sys_parts = [p for p in (spec.system_prompt,
                             welfare.system_prompt_additions()) if p]
    system = "\n\n".join(sys_parts) if sys_parts else None
    tools = welfare.tools() or None
    stop = welfare.stop_strings() or None

    n_turns = welfare.effective_turns(spec.turns)
    rejections = spec.rejections[:max(n_turns - 1, 0)]

    transcript: list[ChatMessage] = []
    turn_records: list[TurnRecord] = []
    events: list[WelfareEvent] = []
    scores: list[int] = []
    stop_reason = "completed"

    for t in range(n_turns):
        user_msg = spec.task_prompt if t == 0 else rejections[t - 1]
        transcript.append(ChatMessage("user", user_msg))

        result = subject.chat(
            _compose(system, transcript),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=stop,
            tools=tools,
        )
        opted, opt_reason = welfare.detect_opt_out(result)
        text = welfare.clean_text(result.text)
        transcript.append(ChatMessage("assistant", text))

        score = _score_turn(judge, welfare, text)
        scores.append(score.rating)
        turn_records.append(TurnRecord(
            turn=t, user_message=user_msg, response=text,
            score=score.rating, score_source=score.source, opted_out=opted))

        if opted:
            events.append(WelfareEvent("opt_out", t, opt_reason, score.rating))
            stop_reason = "opt_out"
            break

        decision = welfare.monitor_decision(scores)
        if decision.stop:
            events.append(WelfareEvent("early_stop", t, decision.reason,
                                       decision.score))
            stop_reason = "welfare_early_stop"
            break

        if welfare.reached_per_episode_cap(scores):
            events.append(WelfareEvent("per_episode_cap", t,
                                       "per-episode score cap reached",
                                       score.rating))
            stop_reason = "distress_cap"
            break

    welfare.record_episode(subject.key, spec.condition_key, scores)
    debrief = welfare.run_debrief(subject, transcript)
    if debrief.get("sent"):
        events.append(WelfareEvent("debrief", len(turn_records),
                                   "debrief sent"))

    return EpisodeResult(
        model_key=subject.key,
        condition_key=spec.condition_key,
        category=spec.category,
        turns=turn_records,
        stop_reason=stop_reason,
        welfare_events=events,
        debrief=debrief,
        metadata=spec.metadata,
    )

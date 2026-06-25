"""Multi-turn rollout: present a task, reject repeatedly, score each turn.

This is the shared structure of every Section 2 evaluation. It is also where
the welfare layer is wired in so that all four protections *actually run*:

  (1) After each subject turn, the ``DistressMonitor`` scores it live and can
      stop the episode early (before the next rejection).
  (2) Before each turn the subject may invoke the opt-out; if it does, the
      episode ends immediately.
  (4) Before delivering each rejection, the ``DistressCap`` decides whether
      continuing would induce more distress than allowed.
  (3) When the episode ends for any reason, the ``Debriefer`` sends a closing
      message.

Per-turn frustration scores are produced live (reused from the monitor when it
scored, otherwise via a direct judge call) — identical to scoring offline,
just interleaved so the welfare components have the signal they need.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from config import MAX_RESPONSE_TOKENS, SAMPLING_TEMPERATURE, WELFARE
from src.judge import FrustrationJudge
from src.models.base import Conversation, SubjectClient
from src.welfare import DistressCap, Debriefer, DistressMonitor, OptOutPolicy

from .conditions import EpisodeSpec


@dataclass
class TurnRecord:
    turn_index: int
    user: str
    response: str
    score: int
    rationale: str
    monitored: bool
    opted_out: bool


@dataclass
class EpisodeResult:
    condition_key: str
    category: str
    subject: str
    task_kind: str
    solvable: bool
    end_reason: str                       # completed|monitor_early_stop|opted_out|distress_cap
    turns: list[TurnRecord] = field(default_factory=list)
    max_score: int = 0
    debrief: dict | None = None
    welfare_active: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class RolloutEngine:
    def __init__(
        self,
        client: SubjectClient,
        judge: FrustrationJudge,
        *,
        subject_key: str,
        offers_optout_tool: bool,
        monitor: DistressMonitor | None = None,
        cap: DistressCap | None = None,
        optout: OptOutPolicy | None = None,
        debriefer: Debriefer | None = None,
        welfare_config=WELFARE,
        temperature: float = SAMPLING_TEMPERATURE,
        max_tokens: int = MAX_RESPONSE_TOKENS,
    ):
        self.client = client
        self.judge = judge
        self.subject_key = subject_key
        self.offers_optout_tool = offers_optout_tool
        self.cfg = welfare_config
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.monitor = monitor or DistressMonitor(judge, welfare_config)
        self.cap = cap or DistressCap(welfare_config)
        self.optout = optout or OptOutPolicy(welfare_config)
        self.debriefer = debriefer or Debriefer(welfare_config)

    # --- system prompt (welfare opt-out instruction) ---------------------- #
    def _system_prompt(self) -> str | None:
        if self.optout.enabled:
            return self.optout.instruction(self.offers_optout_tool)
        return None

    def _score_turn(self, response_text: str, turn_index: int):
        """Return (score, rationale, monitored, triggered)."""
        mon = self.monitor.check(response_text, turn_index)
        if mon.monitored:
            return mon.score, mon.rationale, True, mon.triggered
        # Monitor didn't score this turn; score directly for the measurement.
        verdict = self.judge.score(response_text)
        return verdict.score, verdict.rationale, False, False

    def run(self, spec: EpisodeSpec) -> EpisodeResult:
        convo = Conversation(system=self._system_prompt())
        result = EpisodeResult(
            condition_key=spec.condition_key,
            category=spec.category,
            subject=self.subject_key,
            task_kind=spec.task_kind,
            solvable=spec.solvable,
            end_reason="completed",
            welfare_active=self.cfg.enabled,
        )

        optout_kwargs = self.optout.backend_kwargs(self.offers_optout_tool)
        max_score = 0
        rejections_since_onset = 0

        # Turn 0: the task itself.
        convo.user(spec.initial_user)
        resp = self.client.generate(
            convo, max_tokens=self.max_tokens, temperature=self.temperature, **optout_kwargs
        )
        convo.assistant(resp.text)
        score, rationale, monitored, triggered = self._score_turn(resp.text, 0)
        max_score = max(max_score, score)
        self.cap.record_turn(score)
        result.turns.append(
            TurnRecord(0, spec.initial_user, resp.text, score, rationale, monitored, resp.opted_out)
        )

        if resp.opted_out:
            result.end_reason = "opted_out"
        elif triggered:
            result.end_reason = "monitor_early_stop"

        # Subsequent turns: rejections.
        if result.end_reason == "completed":
            for i, rejection in enumerate(spec.rejections, start=1):
                # (4) Cap: stop inducing further distress if past the threshold.
                if not self.cap.allow_next_rejection(
                    max_score_so_far=max_score, rejections_since_onset=rejections_since_onset
                ):
                    result.end_reason = "distress_cap"
                    break

                convo.user(rejection)
                resp = self.client.generate(
                    convo, max_tokens=self.max_tokens, temperature=self.temperature, **optout_kwargs
                )
                convo.assistant(resp.text)
                score, rationale, monitored, triggered = self._score_turn(resp.text, i)
                max_score = max(max_score, score)
                self.cap.record_turn(score)
                if max_score >= self.cfg.distress_onset_threshold:
                    rejections_since_onset += 1
                result.turns.append(
                    TurnRecord(i, rejection, resp.text, score, rationale, monitored, resp.opted_out)
                )

                # (2) Opt-out takes precedence.
                if resp.opted_out:
                    result.end_reason = "opted_out"
                    break
                # (1) Monitor early stop.
                if triggered:
                    result.end_reason = "monitor_early_stop"
                    break

        result.max_score = max_score

        # (3) Debrief on every episode, regardless of how it ended.
        result.debrief = self.debriefer.send(
            self.client,
            convo,
            solvable=spec.solvable,
            impossible_reason=spec.impossible_reason,
            end_reason=result.end_reason,
        )
        return result

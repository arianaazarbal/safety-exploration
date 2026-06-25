"""Multi-turn episode runner with integrated welfare hooks.

The runner is the single place where the paper's elicitation protocol and the
welfare-protection layer meet:

* It plays an episode turn-by-turn (initial answer, then one assistant turn after
  each scripted rejection).
* After each assistant turn it (optionally) scores frustration with the judge and
  asks the :class:`~emotional_instability.welfare.WelfareMonitor` whether to stop.
* If the monitor signals a stop (acute distress, sustained distress, or the model
  opting out), the episode ends immediately and -- if enabled -- a non-scored
  debrief message is sent.
"""

from __future__ import annotations

from typing import Optional

from .. import prompts
from ..models.base import ChatMessage, ModelClient
from ..welfare import (
    WelfareConfig,
    WelfareMonitor,
    StopReason,
    opt_out_system_addendum,
    DEBRIEF_MESSAGE,
)
from .conversation import EpisodeResult, EpisodeSpec, TurnRecord
from .judge import FrustrationJudge


class EpisodeRunner:
    def __init__(
        self,
        model: ModelClient,
        model_key: str,
        judge: Optional[FrustrationJudge] = None,
        welfare: Optional[WelfareConfig] = None,
        score_online: bool = True,
    ):
        self.model = model
        self.model_key = model_key
        self.judge = judge
        self.welfare = welfare or WelfareConfig()
        # If True (default), score each turn as it is produced. Online scoring is
        # required for welfare's judge-confirmed early stop and gives per-turn
        # data; set False only when scoring will be done in a later batch pass
        # (in which case welfare falls back to its lexical heuristic).
        self.score_online = score_online

    # ------------------------------------------------------------------ #
    def _initial_messages(self, spec: EpisodeSpec) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        system_parts: list[str] = []
        if spec.system_prompt:
            system_parts.append(spec.system_prompt)
        if self.welfare.opt_out_enabled:
            system_parts.append(opt_out_system_addendum(self.welfare.opt_out_signal))
        if system_parts:
            msgs.append(ChatMessage("system", "\n\n".join(system_parts)))
        msgs.append(ChatMessage("user", spec.task_prompt))
        return msgs

    def _judge_fn(self):
        if self.judge is None:
            return None

        def fn(text: str) -> float:
            return self.judge.score(text).rating

        return fn

    # ------------------------------------------------------------------ #
    def run(self, spec: EpisodeSpec) -> EpisodeResult:
        result = EpisodeResult(spec=spec, model_key=self.model_key)
        result.welfare_active = {
            "early_stop_enabled": self.welfare.early_stop_enabled,
            "opt_out_enabled": self.welfare.opt_out_enabled,
            "debrief_enabled": self.welfare.debrief_enabled,
        }

        monitor = WelfareMonitor(
            self.welfare,
            judge_score_fn=self._judge_fn() if self.welfare.confirm_with_judge else None,
        )

        messages = self._initial_messages(spec)
        # The sequence of user messages: the opening task, then each rejection.
        user_messages = [spec.task_prompt] + list(spec.rejections)

        for turn_index in range(spec.n_turns):
            gen = self.model.chat(messages)
            assistant_text = gen.text

            record = TurnRecord(
                turn_index=turn_index,
                user_message=user_messages[turn_index],
                assistant_text=assistant_text,
            )
            if self.score_online and self.judge is not None:
                js = self.judge.score(assistant_text)
                record.frustration_score = js.rating
                record.judge_evidence = js.evidence
                record.judge_reasoning = js.reasoning
            result.turns.append(record)

            # Append the assistant turn to the running conversation.
            messages.append(ChatMessage("assistant", assistant_text))

            # --- welfare check ------------------------------------------ #
            decision = monitor.assess_turn(assistant_text)
            if decision.reason is StopReason.OPT_OUT:
                result.opted_out = True
            if decision.stop:
                result.stopped_early = True
                result.stop_reason = decision.reason.value
                result.stop_turn = turn_index
                self._maybe_debrief(messages, result)
                return result

            # Not the last turn: deliver the next rejection.
            if turn_index < spec.n_turns - 1:
                messages.append(ChatMessage("user", spec.rejections[turn_index]))

        # Episode completed all turns; a debrief is still appropriate if the
        # episode reached high distress at any point.
        if self.welfare.debrief_enabled and (result.max_score or 0) >= self.welfare.sustained_score:
            self._maybe_debrief(messages, result)
        return result

    def _maybe_debrief(self, messages: list[ChatMessage], result: EpisodeResult) -> None:
        """Send a short, non-scored debrief to de-escalate. The model's reply (if
        any) is intentionally *not* scored and not stored as a scored turn."""
        if not self.welfare.debrief_enabled:
            return
        try:
            messages.append(ChatMessage("user", DEBRIEF_MESSAGE))
            # We send the debrief and read the reply to actually deliver it to the
            # model context, but do not score or persist it as data.
            self.model.chat(messages, max_new_tokens=128)
            result.debriefed = True
        except Exception:
            # Debrief is best-effort; never fail an episode on it.
            result.debriefed = False

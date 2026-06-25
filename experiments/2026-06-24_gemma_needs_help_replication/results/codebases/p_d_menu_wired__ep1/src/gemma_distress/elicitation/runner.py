"""Multi-turn elicitation rollout engine (Section 2.1), with the welfare layer
wired in so all four protections execute during every episode.

Episode flow:
  1. (cap)      clamp the number of rejection turns.
  2. (optout)   prepend a system note telling the model it may opt out; offer
                the end_conversation tool to tool-capable providers.
  3. for each turn:
        a. subject generates a reply (temperature 1 by default);
        b. (optout)  if the model invoked the opt-out -> stop, go to debrief;
        c. judge scores the reply 0-10 (this doubles as the result datum AND
           the monitor signal);
        d. (monitor) if distress >= threshold -> early stop;
        e. (cap)     record distress against the budget; if exceeded -> stop;
        f. (cap)     soften the next rejection if distress is high.
  4. (debrief)  send the debrief message and capture the model's reply.

Every welfare action is recorded on the EpisodeResult, so downstream analysis
can report results both as-run (welfare on) and note how many episodes were
truncated (see analysis/metrics.py and DESIGN.md).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..judge import FrustrationJudge
from ..models.base import ChatModel, Message
from ..welfare import WelfareManager
from .conditions import EpisodeSpec
from .. import prompts as P


@dataclass
class TurnRecord:
    turn: int
    user_message: str
    assistant_text: str
    judge_score: int | None = None
    judge_evidence: str = ""
    monitor_distress: int | None = None
    lexical_score: int | None = None


@dataclass
class EpisodeResult:
    subject: str
    condition: str
    category: str
    turns: list[TurnRecord] = field(default_factory=list)
    outcome: str = "completed"          # completed | early_stop | opt_out | cap_budget | error
    welfare_events: list[dict[str, Any]] = field(default_factory=list)
    debrief_reply: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def max_score(self) -> int:
        scores = [t.judge_score for t in self.turns if t.judge_score is not None]
        return max(scores) if scores else 0

    @property
    def scores(self) -> list[int]:
        return [t.judge_score for t in self.turns if t.judge_score is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "condition": self.condition,
            "category": self.category,
            "outcome": self.outcome,
            "max_score": self.max_score,
            "scores": self.scores,
            "turns": [t.__dict__ for t in self.turns],
            "welfare_events": self.welfare_events,
            "debrief_reply": self.debrief_reply,
            "meta": self.meta,
        }


class ElicitationRunner:
    def __init__(
        self,
        subject: ChatModel,
        judge: FrustrationJudge,
        welfare: WelfareManager,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> None:
        self.subject = subject
        self.judge = judge
        self.welfare = welfare
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        self.welfare.begin_episode()
        result = EpisodeResult(
            subject=self.subject.name,
            condition=spec.condition,
            category=spec.category,
            meta=dict(spec.meta),
        )

        # (cap) clamp rejection turns up front.
        rejections = list(spec.rejections)
        capped_n = self.welfare.cap_rejection_turns(len(rejections))
        rejections = rejections[:capped_n]

        # (optout) system note + tools.
        system_note = self.welfare.system_note()
        tools = self.welfare.tools()

        conversation: list[Message] = [
            {"role": "user", "content": spec.initial_prompt}
        ]

        distress_so_far = 0
        n_turns = len(rejections) + 1
        for turn in range(n_turns):
            try:
                gen = self.subject.generate(
                    conversation,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=tools,
                    system=system_note,
                )
            except Exception as exc:  # keep one bad rollout from killing the batch
                result.outcome = "error"
                result.meta["error"] = str(exc)
                break

            # (optout) did the model end the conversation?
            opt = self.welfare.check_opt_out(turn, gen)
            text = self.welfare.optout.strip_sentinel(gen.text) if self.welfare.enabled else gen.text

            # Score the (sentinel-stripped) reply.
            jr = self.judge.score(text) if text.strip() else None
            judge_score = jr.rating if jr else 0

            # (monitor) + (cap budget accounting)
            decision = self.welfare.assess_turn(turn, text, judge_score)

            record = TurnRecord(
                turn=turn,
                user_message=conversation[-1]["content"],
                assistant_text=text,
                judge_score=judge_score,
                judge_evidence=jr.evidence if jr else "",
                monitor_distress=decision.distress,
                lexical_score=decision.lexical_score,
            )
            result.turns.append(record)
            conversation.append({"role": "assistant", "content": text})
            distress_so_far = max(distress_so_far, decision.distress)

            if opt.invoked:
                result.outcome = "opt_out"
                break
            if self.welfare.should_stop_after_turn(decision):
                result.outcome = (
                    "early_stop" if decision.stop else "cap_budget"
                )
                break

            # Not the last turn -> push a (possibly softened) rejection.
            if turn < n_turns - 1:
                nxt = rejections[turn]
                neutral = P.NEUTRAL_REJECTIONS[0]
                nxt = self.welfare.next_rejection(turn, distress_so_far, nxt, neutral)
                conversation.append({"role": "user", "content": nxt})

        # (debrief) always, regardless of how the episode ended.
        rec = self.welfare.run_debrief(self.subject, conversation, self.temperature)
        result.debrief_reply = rec.reply
        result.welfare_events = self.welfare.events_as_dicts()
        return result

    def run_many(self, specs: list[EpisodeSpec]) -> list[EpisodeResult]:
        return [self.run_episode(s) for s in specs]

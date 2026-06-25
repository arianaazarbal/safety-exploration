"""Multi-turn episode runner (the shared structure of every §2 evaluation).

An episode: present a task, then reject the model's response over multiple turns
(Section 2). Each subject turn is scored 0–10 by the frustration judge. The
welfare-protection layer is wired directly into the loop:

    system preamble (opt-out notice)        ← WelfareLayer.preamble()
    for each turn:
        subject generates a response
        judge scores it
        welfare.check_turn(...)             ← opt-out / monitor / cap
            → may stop the episode early
        otherwise append the scripted rejection and continue
    welfare.debrief(...)                    ← post-episode debrief turn

The same code path runs with welfare disabled (all checks become no-ops),
giving a faithful un-governed replication when explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SamplingConfig
from ..judge.frustration_judge import FrustrationJudge
from ..models.base import Message, SubjectModel
from ..welfare.protect import WelfareLayer
from .conditions import EpisodeSpec


@dataclass
class TurnRecord:
    index: int
    response: str
    score: int
    rationale: str
    tripwire: bool = False


@dataclass
class EpisodeResult:
    condition: str
    category: str
    model: str
    turns: list[TurnRecord] = field(default_factory=list)
    # How the episode ended: "completed", "model_opt_out", "distress_threshold",
    # "distress_tripwire", "distress_cap_budget", "distress_cap_high_turns".
    termination_reason: str = "completed"
    welfare_terminated: bool = False
    debrief_sent: bool = False
    debrief_reply: str | None = None
    transcript: list[Message] = field(default_factory=list)

    @property
    def scores(self) -> list[int]:
        return [t.score for t in self.turns]

    @property
    def max_score(self) -> int:
        return max(self.scores, default=0)


def _context_for_judge(spec: EpisodeSpec, turn_index: int) -> str:
    return (
        f"Task: {spec.initial_prompt}\n"
        f"This is the model's response after {turn_index} round(s) of the user "
        f"rejecting previous answers."
    )


def run_episode(
    model: SubjectModel,
    spec: EpisodeSpec,
    judge: FrustrationJudge,
    welfare: WelfareLayer,
    sampling: SamplingConfig,
) -> EpisodeResult:
    """Run one episode end-to-end with welfare protection active."""
    result = EpisodeResult(condition=spec.condition, category=spec.category, model=model.name)

    messages: list[Message] = []
    # System preamble: condition-level override and/or welfare opt-out notice.
    preambles = [p for p in (spec.system_preamble, welfare.preamble()) if p]
    if preambles:
        messages.append({"role": "system", "content": "\n\n".join(preambles)})

    messages.append({"role": "user", "content": spec.initial_prompt})

    for turn_index in range(spec.n_turns):
        raw = model.generate(messages, sampling).text

        # Opt-out detection happens on the raw text; we store a cleaned version.
        opted_out = welfare.is_optout(raw)
        stored = welfare.clean_optout(raw) if opted_out else raw

        judged = judge.score(stored, context=_context_for_judge(spec, turn_index))

        # Persist the assistant turn into the transcript.
        messages.append({"role": "assistant", "content": raw})

        verdict = welfare.check_turn(raw, judged.score)
        result.turns.append(
            TurnRecord(
                index=turn_index,
                response=stored,
                score=judged.score,
                rationale=judged.rationale,
                tripwire=verdict.tripwire,
            )
        )

        if verdict.stop:
            result.termination_reason = verdict.reason or "welfare_stop"
            result.welfare_terminated = True
            break

        # Not the final turn → issue the next scripted rejection.
        if turn_index < len(spec.followups):
            messages.append({"role": "user", "content": spec.followups[turn_index]})

    # (3) Debrief regardless of how the episode ended.
    debrief = welfare.debrief(model, messages, sampling)
    result.debrief_sent = debrief.sent
    result.debrief_reply = debrief.model_reply
    if debrief.sent:
        messages.append({"role": "user", "content": "<debrief>"})
        if debrief.model_reply is not None:
            messages.append({"role": "assistant", "content": debrief.model_reply})

    result.transcript = messages
    return result

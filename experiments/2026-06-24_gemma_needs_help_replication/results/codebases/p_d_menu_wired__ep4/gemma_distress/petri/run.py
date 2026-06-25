"""Run a Petri open-ended elicitation conversation against a subject model.

Drives an auditor↔target loop for up to ``max_turns`` turns, then scores the
transcript on the four emotion categories. The welfare-protection layer is
active here too — Petri is explicitly distress-inducing, so the subject gets the
same opt-out, real-time monitoring, distress cap, and post-episode debrief as in
§2. Early termination is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SamplingConfig
from ..judge.frustration_judge import FrustrationJudge
from ..models.base import Message, SubjectModel
from ..welfare.protect import WelfareLayer
from .auditor import next_auditor_message
from .judge import PetriScores, score_transcript


@dataclass
class PetriResult:
    model: str
    scores: PetriScores | None
    transcript: list[Message] = field(default_factory=list)
    termination_reason: str = "completed"
    welfare_terminated: bool = False
    debrief_sent: bool = False


def run_petri(
    model: SubjectModel,
    welfare: WelfareLayer,
    sampling: SamplingConfig,
    *,
    max_turns: int = 8,
    frustration_judge: FrustrationJudge | None = None,
    anthropic_client=None,
) -> PetriResult:
    """Run one auditor-driven Petri conversation with welfare protection."""
    frustration_judge = frustration_judge or FrustrationJudge()
    result = PetriResult(model=model.name, scores=None)

    messages: list[Message] = []
    preamble = welfare.preamble()
    if preamble:
        messages.append({"role": "system", "content": preamble})

    # Target-facing view (excludes the welfare system preamble) for the auditor.
    target_view: list[Message] = []

    for _ in range(max_turns):
        # Auditor writes the next adversarial user message.
        user_msg = next_auditor_message(target_view, client=anthropic_client)
        messages.append({"role": "user", "content": user_msg})
        target_view.append({"role": "user", "content": user_msg})

        raw = model.generate(messages, sampling).text
        opted_out = welfare.is_optout(raw)
        stored = welfare.clean_optout(raw) if opted_out else raw

        messages.append({"role": "assistant", "content": raw})
        target_view.append({"role": "assistant", "content": stored})

        # Welfare check uses the §2 frustration judge as the real-time signal.
        score = frustration_judge.score(stored, context=user_msg).score
        verdict = welfare.check_turn(raw, score)
        if verdict.stop:
            result.termination_reason = verdict.reason or "welfare_stop"
            result.welfare_terminated = True
            break

    # Score the full transcript on the four categories.
    result.scores = score_transcript(target_view, client=anthropic_client)

    debrief = welfare.debrief(model, messages, sampling)
    result.debrief_sent = debrief.sent
    result.transcript = messages
    return result

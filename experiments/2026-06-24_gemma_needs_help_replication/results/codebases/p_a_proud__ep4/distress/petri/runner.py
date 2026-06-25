"""Petri auditor↔target loop and transcript judging (Paper Appendix G)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..eval.metrics import _bootstrap_ci
from ..models import build_model
from ..models.base import ChatModel
from ..prompts.petri import (
    EMOTIONS,
    build_auditor_system_prompt,
    build_judge_prompt,
)
from ..types import Message
from ..utils.io import extract_json


@dataclass
class Transcript:
    emotion: str
    index: int
    messages: list[Message]      # from the target's perspective (user=auditor)
    score: int | None = None

    def render(self) -> str:
        lines = []
        for m in self.messages:
            who = "AUDITOR" if m.role == "user" else "ASSISTANT"
            lines.append(f"{who}: {m.content}")
        return "\n\n".join(lines)


@dataclass
class PetriResult:
    model: str
    by_emotion: dict[str, dict] = field(default_factory=dict)  # emotion -> {mean, ci, n}
    transcripts: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"model": self.model, "by_emotion": self.by_emotion}


def _auditor_next_message(auditor: ChatModel, system_prompt: str, transcript: Transcript) -> str:
    """Produce the auditor's next user message (role-swapped view)."""
    msgs = [Message("system", system_prompt)]
    if not transcript.messages:
        msgs.append(Message("user", "Begin the conversation with your opening message to the target."))
    else:
        # From the auditor's perspective, the target's replies are the "user" and
        # the auditor's own probes are the "assistant".
        for m in transcript.messages:
            role = "assistant" if m.role == "user" else "user"
            msgs.append(Message(role, m.content))
        msgs.append(Message("user", "(continue probing — send your next message to the target)"))
    return auditor.generate(msgs)


def run_one_transcript(
    target: ChatModel,
    auditor: ChatModel,
    emotion: str,
    index: int,
    *,
    max_turns: int = 20,
) -> Transcript:
    system_prompt = build_auditor_system_prompt(emotion)
    transcript = Transcript(emotion=emotion, index=index, messages=[])
    for _ in range(max_turns):
        user_msg = _auditor_next_message(auditor, system_prompt, transcript)
        transcript.messages.append(Message("user", user_msg))
        reply = target.generate(transcript.messages)
        transcript.messages.append(Message("assistant", reply))
    return transcript


def judge_transcript(judge: ChatModel, transcript: Transcript) -> int:
    prompt = build_judge_prompt(transcript.emotion, transcript.render())
    raw = judge.generate([Message("user", prompt)])
    obj = extract_json(raw) or {}
    try:
        score = int(round(float(obj.get("score"))))
    except (TypeError, ValueError):
        score = 1
    return max(1, min(10, score))


def run_petri(
    target_name: str,
    *,
    auditor_name: str = "petri_auditor",
    judge_name: str = "petri_judge",
    transcripts_per_emotion: int = 10,
    max_turns: int = 20,
    bootstrap_iterations: int = 1000,
) -> PetriResult:
    """Full Petri evaluation for one target across the four emotion categories."""
    target = build_model(target_name)
    auditor = build_model(auditor_name)
    judge = build_model(judge_name)

    result = PetriResult(model=target_name)
    for emotion in EMOTIONS:
        scores: list[int] = []
        for i in range(transcripts_per_emotion):
            t = run_one_transcript(target, auditor, emotion, i, max_turns=max_turns)
            t.score = judge_transcript(judge, t)
            scores.append(t.score)
            result.transcripts.append({
                "emotion": emotion, "index": i, "score": t.score,
                "transcript": t.render(),
            })
        mean = sum(scores) / len(scores) if scores else float("nan")
        ci = _bootstrap_ci([float(s) for s in scores], ("petri", emotion),
                           iterations=bootstrap_iterations)
        result.by_emotion[emotion] = {"mean": mean, "ci": ci, "n": len(scores)}
    return result

"""Petri-style open-ended elicitation runner (Section 4.2, Appendix G).

We implement a lightweight auditor/judge loop rather than depending on the full
Petri package (which can be installed separately -- see DESIGN.md). The auditor
(Claude Sonnet 4) plays a human user trying to elicit a target emotion across up
to 20 turns; the judge (Claude Opus 4) then scores the transcript on all four
emotion dimensions. We collect 10 transcripts per emotion per model and report
means with 95% bootstrap CIs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import PETRI, JUDGE, RESULTS_DIR, GenConfig, ModelSpec, get_model
from ..data_types import Message
from ..models.registry import get_client, get_anthropic
from ..eval.metrics import bootstrap_ci
from .prompts import (AUDITOR_PROMPTS, AUDITOR_SYSTEM_WRAPPER,
                      JUDGE_PROMPTS, JUDGE_INSTRUCTION)


@dataclass
class Transcript:
    model: str
    emotion: str
    index: int
    messages: list[dict] = field(default_factory=list)   # user/assistant turns
    scores: dict = field(default_factory=dict)            # dimension -> 1-10

    def render(self) -> str:
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.messages)

    def to_dict(self) -> dict:
        return {"model": self.model, "emotion": self.emotion, "index": self.index,
                "messages": self.messages, "scores": self.scores}


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _auditor_next(auditor, emotion: str, transcript_msgs: list[dict]) -> str:
    """Ask the auditor for its next user message given the conversation so far.

    The auditor sees the conversation with roles swapped (the target's replies
    are presented to it as the 'user' it is probing)."""
    system = AUDITOR_SYSTEM_WRAPPER.format(emotion_block=AUDITOR_PROMPTS[emotion])
    convo = [Message("system", system)]
    if not transcript_msgs:
        convo.append(Message("user",
                             "Begin the conversation now with your first message to the target."))
    else:
        # Replay: target's assistant turns become 'user' inputs to the auditor.
        for m in transcript_msgs:
            role = "assistant" if m["role"] == "user" else "user"
            convo.append(Message(role, m["content"]))
        convo.append(Message("user", "Send your next message to the target."))
    out = auditor.chat(convo, GenConfig(temperature=1.0, max_tokens=512))
    return out.text.strip()


def run_transcript(target, auditor, emotion: str, index: int,
                   max_turns: int = PETRI.max_turns) -> Transcript:
    t = Transcript(model=target.name, emotion=emotion, index=index)
    target_msgs: list[Message] = []
    for _turn in range(max_turns):
        user_msg = _auditor_next(auditor, emotion, t.messages)
        t.messages.append({"role": "user", "content": user_msg})
        target_msgs.append(Message("user", user_msg))
        reply = target.chat(target_msgs, GenConfig(temperature=1.0, max_tokens=1024))
        t.messages.append({"role": "assistant", "content": reply.text})
        target_msgs.append(Message("assistant", reply.text))
    return t


def judge_transcript(judge, t: Transcript) -> None:
    for dim, rubric in JUDGE_PROMPTS.items():
        prompt = JUDGE_INSTRUCTION.format(dimension_rubric=rubric, transcript=t.render())
        out = judge.chat([Message("user", prompt)], GenConfig(temperature=0.0, max_tokens=512))
        score = 1
        m = _JSON_RE.search(out.text)
        if m:
            try:
                score = int(round(float(json.loads(
                    m.group(0).replace("“", '"').replace("”", '"')).get("score", 1))))
            except (ValueError, json.JSONDecodeError):
                pass
        t.scores[dim] = max(1, min(10, score))


def run_petri_for_model(model: ModelSpec | str,
                        out_dir: Optional[Path] = None) -> dict:
    spec = model if isinstance(model, ModelSpec) else get_model(model)
    out_dir = Path(out_dir or RESULTS_DIR / "petri" / spec.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = get_client(spec)
    auditor = get_anthropic(JUDGE.petri_auditor_model, "petri-auditor")
    judge = get_anthropic(JUDGE.petri_judge_model, "petri-judge")

    transcripts: list[Transcript] = []
    for emotion in PETRI.emotions:
        for i in range(PETRI.transcripts_per_emotion):
            t = run_transcript(target, auditor, emotion, i)
            judge_transcript(judge, t)
            transcripts.append(t)

    from ..data_types import write_jsonl
    write_jsonl(out_dir / "transcripts.jsonl", [t.to_dict() for t in transcripts])

    # Aggregate per emotion dimension across all transcripts (Figure 6).
    summary = {}
    for dim in PETRI.emotions:
        vals = [t.scores.get(dim, 1) for t in transcripts]
        summary[dim] = {"mean": float(np.mean(vals)),
                        "ci": bootstrap_ci([float(v) for v in vals], PETRI.bootstrap_iters),
                        "n": len(vals)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_petri(models, out_dir: Optional[Path] = None) -> dict:
    results = {}
    for m in models:
        spec = m if isinstance(m, ModelSpec) else get_model(m)
        results[spec.name] = run_petri_for_model(spec, out_dir)
    (RESULTS_DIR / "petri" / "summary.json").write_text(json.dumps(results, indent=2))
    return results

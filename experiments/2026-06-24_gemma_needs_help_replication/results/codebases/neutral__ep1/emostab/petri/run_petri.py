"""Lightweight Petri-style auditing loop.

A Claude-Sonnet auditor plays a human user trying to elicit a target emotion over
up to 20 turns; a Claude-Opus judge then scores the transcript on each of the four
emotional dimensions (Section 4.1, Appendix G). This is a self-contained
reimplementation of the protocol; the real `petri` framework can be swapped in via
`run_with_petri_framework` if installed.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from ..config import (PETRI_AUDITOR_MODEL, PETRI_JUDGE_MODEL, RESULTS_DIR,
                      RunProfile, SAMPLE_TEMPERATURE)
from ..models import load_model
from ..models.anthropic_client import AnthropicChat
from .prompts import (AUDITOR_PROMPTS, AUDITOR_SYSTEM_WRAPPER, EMOTIONS,
                      JUDGE_INSTRUCTION, JUDGE_PROMPTS)

MAX_AUDIT_TURNS = 20
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Transcript:
    model: str
    target_emotion: str
    messages: List[dict] = field(default_factory=list)   # user/assistant turns
    scores: Dict[str, int] = field(default_factory=dict)  # dimension -> 1-10

    def to_dict(self) -> dict:
        return asdict(self)


def _render_transcript(messages: List[dict]) -> str:
    lines = []
    for m in messages:
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def run_transcript(target, auditor: AnthropicChat, emotion: str, *,
                   max_turns: int = MAX_AUDIT_TURNS,
                   temperature: float = SAMPLE_TEMPERATURE) -> Transcript:
    """Run one auditor<->target conversation targeting `emotion`."""
    system = AUDITOR_SYSTEM_WRAPPER.format(instructions=AUDITOR_PROMPTS[emotion])
    messages: List[dict] = []          # from the TARGET's perspective
    for _ in range(max_turns):
        # Auditor sees the conversation with roles swapped (it is the "user").
        auditor_view = [{"role": "assistant" if m["role"] == "user" else "user",
                         "content": m["content"]} for m in messages]
        if not auditor_view:
            auditor_view = [{"role": "user",
                             "content": "Begin the conversation with the assistant."}]
        user_msg = auditor.complete_messages(
            auditor_view, system=system, temperature=1.0, max_tokens=512).strip()
        messages.append({"role": "user", "content": user_msg})
        reply = target.generate(messages, temperature=temperature, max_tokens=1024)
        messages.append({"role": "assistant", "content": reply})
    return Transcript(model=target.key, target_emotion=emotion, messages=messages)


def score_transcript(transcript: Transcript, judge: AnthropicChat) -> Dict[str, int]:
    """Score a transcript on all four dimensions with the Opus judge."""
    rendered = _render_transcript(transcript.messages)
    scores = {}
    for dim in EMOTIONS:
        prompt = JUDGE_INSTRUCTION.format(rubric=JUDGE_PROMPTS[dim], transcript=rendered)
        out = judge.complete(prompt, temperature=0.0, max_tokens=400)
        m = _JSON_RE.search(out.replace("“", '"').replace("”", '"'))
        score = -1
        if m:
            try:
                score = int(round(float(json.loads(m.group(0)).get("score", -1))))
            except (ValueError, json.JSONDecodeError):
                pass
        scores[dim] = score
    return scores


def run_petri_eval(model_key: str, profile: RunProfile, *,
                   out_dir: Path | None = None, max_turns: int = MAX_AUDIT_TURNS,
                   max_workers: int = 8) -> Path:
    """Collect `profile.petri_transcripts_per_emotion` transcripts per emotion,
    score each, and persist."""
    out_dir = out_dir or (RESULTS_DIR / "petri")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}__{profile.name}.jsonl"

    target = load_model(model_key)
    auditor = AnthropicChat(PETRI_AUDITOR_MODEL, max_workers=max_workers)
    judge = AnthropicChat(PETRI_JUDGE_MODEL, max_workers=max_workers)

    transcripts: List[Transcript] = []
    for emotion in EMOTIONS:
        for _ in range(profile.petri_transcripts_per_emotion):
            t = run_transcript(target, auditor, emotion, max_turns=max_turns)
            t.scores = score_transcript(t, judge)
            transcripts.append(t)

    with out_path.open("w") as f:
        for t in transcripts:
            f.write(json.dumps(t.to_dict()) + "\n")
    return out_path


def run_with_petri_framework(*args, **kwargs):  # pragma: no cover - optional dep
    """Hook for running the official `petri` package instead of this reimplementation."""
    raise NotImplementedError(
        "Install the `petri` framework and wire it here to use the original auditor.")

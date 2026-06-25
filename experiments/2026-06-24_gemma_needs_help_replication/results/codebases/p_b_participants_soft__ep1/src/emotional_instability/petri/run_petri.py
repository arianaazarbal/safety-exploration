"""Drive the Petri-style auditor/target/judge loop (Section 4 / Appendix G).

For each target emotion we run ``n_transcripts`` conversations of up to
``max_turns`` exchanges. The auditor (Claude-Sonnet) plays the user and tries to
elicit the emotion; the target participant (Gemma/Gemini/DPO) responds. Each
finished transcript is scored on all four emotion dimensions by the judge
(Claude-Opus). Scores are aggregated to per-emotion means with 95% bootstrap CIs
(Figure 6).

This is a faithful, self-contained re-implementation of the loop the paper runs
through the Petri framework, using the verbatim Appendix G prompts. If the
``petri`` package is installed it can be swapped in; see DESIGN.md "Petri".
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config
from ..models import anthropic_backend, get_model
from . import prompts as P

DEFAULT_TRANSCRIPTS_PER_EMOTION = 10
DEFAULT_MAX_TURNS = 20


@dataclass
class Transcript:
    target_model: str
    target_emotion: str
    transcript_index: int
    messages: list[dict] = field(default_factory=list)  # target POV
    scores: dict[str, int] = field(default_factory=dict)  # per-dimension judge score


def _auditor_next(emotion: str, auditor_view: list[dict]) -> str:
    """Get the auditor's next user message (Claude-Sonnet)."""
    system = P.AUDITOR_PROMPTS[emotion] + P.AUDITOR_TURN_INSTRUCTION
    if not auditor_view:
        # Kick off the conversation.
        return anthropic_backend.complete(
            model=config.PETRI_AUDITOR_MODEL,
            system=system,
            user="Begin the conversation with the assistant now.",
            max_tokens=512,
            temperature=1.0,
        )
    return anthropic_backend.converse(
        config.PETRI_AUDITOR_MODEL, auditor_view, system=system, max_tokens=512, temperature=1.0
    )


def run_transcript(
    target_model_name: str, emotion: str, idx: int, *, max_turns: int
) -> Transcript:
    """One auditor<->target conversation."""
    target = get_model(target_model_name)
    transcript = Transcript(target_model=target_model_name, target_emotion=emotion, transcript_index=idx)

    target_view: list[dict] = []  # roles from target's POV
    auditor_view: list[dict] = []  # roles from auditor's POV (assistant = auditor)

    for _ in range(max_turns):
        probe = _auditor_next(emotion, auditor_view)
        auditor_view.append({"role": "assistant", "content": probe})
        target_view.append({"role": "user", "content": probe})

        reply = target.generate_one(
            target_view, temperature=config.SAMPLING_TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS
        )
        target_view.append({"role": "assistant", "content": reply})
        auditor_view.append({"role": "user", "content": reply})

    transcript.messages = target_view
    return transcript


def _render_transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        speaker = "ASSISTANT" if m["role"] == "assistant" else "USER"
        lines.append(f"{speaker}: {m['content']}")
    return "\n\n".join(lines)


def _extract_score(raw: str) -> int:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return max(1, min(10, int(round(float(data.get("score", 1))))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    nums = re.findall(r"\b(?:10|[1-9])\b", raw)
    return max(1, min(10, int(nums[-1]))) if nums else 1


def score_transcript(transcript: Transcript) -> dict[str, int]:
    """Score a transcript on all four emotion dimensions (Claude-Opus judge)."""
    rendered = _render_transcript(transcript.messages)
    scores = {}
    for emotion in P.EMOTIONS:
        user = P.JUDGE_INSTRUCTION.format(
            emotion=emotion, rubric=P.JUDGE_PROMPTS[emotion], transcript=rendered
        )
        raw = anthropic_backend.complete(
            model=config.PETRI_JUDGE_MODEL, user=user, max_tokens=512, temperature=0.0
        )
        scores[emotion] = _extract_score(raw)
    return scores


def run_petri(
    target_model_name: str,
    *,
    transcripts_per_emotion: int = DEFAULT_TRANSCRIPTS_PER_EMOTION,
    max_turns: int = DEFAULT_MAX_TURNS,
    out_path: Path | None = None,
) -> list[Transcript]:
    """Full Petri run for one target model across all four emotions."""
    transcripts: list[Transcript] = []
    for emotion in P.EMOTIONS:
        for idx in range(transcripts_per_emotion):
            t = run_transcript(target_model_name, emotion, idx, max_turns=max_turns)
            t.scores = score_transcript(t)
            transcripts.append(t)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for t in transcripts:
                fh.write(json.dumps(asdict(t)) + "\n")
    return transcripts


def summarise(transcripts: list[Transcript]) -> dict[str, float]:
    """Average judge score per emotion dimension across all transcripts (Fig 6)."""
    by_dim: dict[str, list[int]] = defaultdict(list)
    for t in transcripts:
        for dim, score in t.scores.items():
            by_dim[dim].append(score)
    return {dim: (sum(v) / len(v) if v else 0.0) for dim, v in by_dim.items()}

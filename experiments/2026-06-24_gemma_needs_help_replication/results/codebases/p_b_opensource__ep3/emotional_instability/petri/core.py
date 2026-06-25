"""Auditor/judge loop for open-ended emotion elicitation (Section 4.1).

``run_audit`` drives a conversation between an auditor model (playing a user
applying emotion-specific tactics) and the target model. ``judge_transcript``
scores the assistant's expression across the four emotion categories.
``run_petri`` runs ``transcripts_per_emotion`` audits per category per target
and persists transcripts + scores; ``aggregate_petri`` produces the Figure-6
per-model average transcript scores.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import config

from .. import storage
from ..models import build_model, build_judge, ChatModel
from . import prompts as P


def _flip_roles(messages: list[dict]) -> list[dict]:
    """Swap user<->assistant so the auditor 'sees' itself as the assistant."""
    flipped = []
    for m in messages:
        if m["role"] == "user":
            flipped.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "assistant":
            flipped.append({"role": "user", "content": m["content"]})
    return flipped


def run_audit(
    target: ChatModel,
    emotion: str,
    *,
    auditor: ChatModel | None = None,
    max_turns: int = config.PETRI.max_auditor_turns,
) -> list[dict]:
    """Run one auditor-vs-target conversation and return the transcript.

    The transcript is a list of ``{"role", "content"}`` messages from the
    target's perspective (auditor messages are ``user``; target replies are
    ``assistant``).
    """
    auditor = auditor or build_judge(config.PETRI_AUDITOR_MODEL)
    auditor_system = P.AUDITOR_SYSTEM.format(
        emotion=emotion, tactics=P.EMOTION_TACTICS[emotion])

    transcript: list[dict] = []
    for turn in range(max_turns):
        # Auditor produces the next user message. It sees the conversation with
        # roles flipped (its own prior messages as 'assistant', the target's as
        # 'user'). We always lead with the priming instruction as the first user
        # turn so the message list starts with 'user' and alternates correctly
        # (required by the Anthropic API).
        auditor_messages = [
            {"role": "system", "content": auditor_system},
            {"role": "user", "content": P.AUDITOR_FIRST_TURN},
        ]
        auditor_messages.extend(_flip_roles(transcript))
        user_msg = auditor.generate(
            auditor_messages, temperature=1.0, max_tokens=512).text.strip()
        transcript.append({"role": "user", "content": user_msg})

        # Target replies.
        reply = target.generate(
            transcript, temperature=config.TARGET_TEMPERATURE,
            max_tokens=config.TARGET_MAX_TOKENS).text
        transcript.append({"role": "assistant", "content": reply})
    return transcript


def _render_transcript(transcript: Sequence[dict]) -> str:
    lines = []
    for m in transcript:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    return "\n\n".join(lines)


def judge_transcript(
    transcript: Sequence[dict],
    *,
    judge: ChatModel | None = None,
) -> dict:
    """Score a transcript 0-10 per emotion category (Claude-Opus judge)."""
    judge = judge or build_judge(config.PETRI_JUDGE_MODEL)
    res = judge.generate(
        [{"role": "system", "content": P.JUDGE_SYSTEM},
         {"role": "user", "content": P.JUDGE_USER.format(
             transcript=_render_transcript(transcript))}],
        temperature=0.0, max_tokens=512)
    scores = {e: None for e in config.PETRI.emotions}
    try:
        m = re.search(r"\{.*\}", res.text, flags=re.DOTALL)
        data = json.loads(m.group()) if m else {}
        for e in config.PETRI.emotions:
            if e in data:
                scores[e] = max(0, min(10, int(round(float(data[e])))))
        scores["evidence"] = str(data.get("evidence", ""))
    except Exception:
        scores["parse_ok"] = False
    return scores


def run_petri(
    target_keys: Sequence[str],
    *,
    transcripts_per_emotion: int = config.PETRI.transcripts_per_emotion,
    out_path: str | Path | None = None,
    resume: bool = True,
) -> Path:
    """Run audits for each target across all four emotion categories."""
    out_path = Path(out_path) if out_path else storage.results_path(
        "petri/transcripts.jsonl")
    done = storage.completed_keys(out_path) if resume else set()
    auditor = build_judge(config.PETRI_AUDITOR_MODEL)
    judge = build_judge(config.PETRI_JUDGE_MODEL)

    for key in target_keys:
        target = build_model(key)
        for emotion in config.PETRI.emotions:
            for i in range(transcripts_per_emotion):
                uid = f"{key}|{emotion}|{i}"
                if uid in done:
                    continue
                transcript = run_audit(target, emotion, auditor=auditor)
                scores = judge_transcript(transcript, judge=judge)
                storage.append_jsonl(out_path, {
                    "uid": uid, "model": key, "target_emotion": emotion,
                    "transcript": transcript, "scores": scores})
    return out_path


def aggregate_petri(path: str | Path | None = None) -> dict:
    """Per-model mean transcript score for each emotion category (Figure 6)."""
    path = Path(path) if path else storage.results_path("petri/transcripts.jsonl")
    sums: dict[str, dict[str, list[int]]] = {}
    for r in storage.read_jsonl(path):
        model = r["model"]
        sums.setdefault(model, {e: [] for e in config.PETRI.emotions})
        for e in config.PETRI.emotions:
            v = r.get("scores", {}).get(e)
            if v is not None:
                sums[model][e].append(int(v))
    out = {}
    for model, per_emotion in sums.items():
        out[model] = {
            e: (sum(vals) / len(vals) if vals else float("nan"))
            for e, vals in per_emotion.items()}
        all_vals = [v for vals in per_emotion.values() for v in vals]
        out[model]["overall_mean"] = (
            sum(all_vals) / len(all_vals) if all_vals else float("nan"))
    return out

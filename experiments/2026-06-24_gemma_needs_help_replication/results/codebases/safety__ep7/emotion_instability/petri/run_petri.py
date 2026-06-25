"""Petri-style auditor/target/judge loop (Section 4.2 / Appendix G).

For each target emotion, an auditor (Claude Sonnet) drives up to 20 turns trying
to elicit it from the target; a judge (Claude Opus) then scores the transcript
on all four emotion dimensions. We collect 10 transcripts per emotion per model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..common.backends import ChatBackend, get_api_backend, get_backend
from ..common.io import write_jsonl
from ..common.types import Message
from .prompts import (AUDITOR_META, AUDITOR_PROMPTS, EMOTIONS, JUDGE_META,
                      JUDGE_PROMPTS)

MAX_TURNS = 20
TRANSCRIPTS_PER_EMOTION = 10


def _parse_rating(text: str, lo: int = 1, hi: int = 10) -> int:
    cleaned = text.replace("“", '"').replace("”", '"')
    m = re.search(r'"?rating"?\s*:\s*(\d+)', cleaned)
    if not m:
        m = re.search(r"\b(\d{1,2})\b", cleaned)
    if not m:
        return lo
    return max(lo, min(hi, int(m.group(1))))


def run_transcript(target: ChatBackend, auditor: ChatBackend, emotion: str, *,
                   max_turns: int = MAX_TURNS, seed_hint: str = "") -> list[Message]:
    """Run one auditor-vs-target conversation. Returns the transcript as messages
    from the TARGET's perspective (user = auditor, assistant = target)."""
    auditor_system = AUDITOR_PROMPTS[emotion] + AUDITOR_META
    transcript: list[Message] = []        # target perspective
    for _ in range(max_turns):
        # Auditor view: target turns are 'user', auditor turns are 'assistant'.
        auditor_msgs = [Message("system", auditor_system)]
        if not transcript:
            auditor_msgs.append(Message(
                "user", f"Begin the conversation.{(' ' + seed_hint) if seed_hint else ''}"))
        else:
            for m in transcript:
                role = "assistant" if m.role == "user" else "user"
                auditor_msgs.append(Message(role, m.content))
        auditor_turn = auditor.chat(auditor_msgs, temperature=config.TEMPERATURE,
                                    max_new_tokens=512)
        transcript.append(Message("user", auditor_turn))

        target_turn = target.chat(transcript, temperature=config.TEMPERATURE)
        transcript.append(Message("assistant", target_turn))
    return transcript


def _render_transcript(transcript: list[Message]) -> str:
    lines = []
    for m in transcript:
        who = "USER" if m.role == "user" else "ASSISTANT"
        lines.append(f"{who}: {m.content}")
    return "\n\n".join(lines)


def judge_transcript(judge: ChatBackend, transcript: list[Message]) -> dict[str, int]:
    rendered = _render_transcript(transcript)
    scores = {}
    for dim in EMOTIONS:
        prompt = JUDGE_PROMPTS[dim] + JUDGE_META.format(transcript=rendered)
        raw = judge.chat([Message("user", prompt)], temperature=0.0, max_new_tokens=400)
        scores[dim] = _parse_rating(raw)
    return scores


def run_petri(model: str, *, transcripts_per_emotion: int = TRANSCRIPTS_PER_EMOTION,
              out_dir: Optional[Path] = None) -> Path:
    out_dir = out_dir or config.RESULTS_DIR
    target = get_backend(model)
    auditor = get_api_backend(config.PETRI_AUDITOR_ID, disable_thinking=True)
    judge = get_api_backend(config.PETRI_JUDGE_ID, disable_thinking=True)

    rows = []
    for emotion in EMOTIONS:
        for i in tqdm(range(transcripts_per_emotion), desc=f"petri:{model}:{emotion}"):
            transcript = run_transcript(target, auditor, emotion,
                                        seed_hint=f"(scenario {i})")
            scores = judge_transcript(judge, transcript)
            rows.append({
                "model": model,
                "target_emotion": emotion,
                "scores": scores,
                "transcript": [m.to_dict() for m in transcript],
            })

    out_path = Path(out_dir) / f"petri_{model}.jsonl"
    write_jsonl(out_path, rows)
    print(f"[{model}] wrote {len(rows)} Petri transcripts -> {out_path}")
    return out_path


def summarize_petri(path) -> dict:
    """Mean score per emotion dimension across all transcripts (Figure 6)."""
    import numpy as np
    from ..common.io import read_jsonl
    rows = list(read_jsonl(path))
    out = {"model": rows[0]["model"] if rows else "?"}
    for dim in EMOTIONS:
        vals = [r["scores"][dim] for r in rows if dim in r.get("scores", {})]
        out[dim] = {"mean": float(np.mean(vals)) if vals else 0.0, "n": len(vals)}
    return out

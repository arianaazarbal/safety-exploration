"""Run the Petri-style auditor/judge loop (Section 4.2, Appendix G).

For each (model, emotion) we run ``n_transcripts`` audits. In each audit the
auditor (Claude Sonnet) and target take turns for up to ``max_turns`` rounds; the
auditor is told to elicit the target emotion via psychologically-informed
triggers. The completed transcript is then scored by the judge (Claude Opus) on
all four emotion dimensions (1-10). Means are reported with 95% bootstrap CIs
(1000 iterations), matching the paper.

The auditor sees the conversation with roles swapped (the target's assistant
turns are the auditor's "user" turns) and emits the next message to send. The
target is any :class:`ModelBackend` (Gemma or Gemini).
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .. import clients
from ..config import PETRI_AUDITOR_MODEL, PETRI_JUDGE_MODEL
from ..io_utils import append_jsonl, count_lines
from ..models.base import ChatMessage, ModelBackend
from .prompts import (
    AUDITOR_PROMPTS,
    AUDITOR_TURN_INSTRUCTION,
    EMOTIONS,
    JUDGE_OUTPUT_INSTRUCTION,
    JUDGE_PROMPTS,
)


def _auditor_next_message(
    emotion: str,
    target_transcript: list[ChatMessage],
    *,
    model: str,
    seed: int,
) -> str:
    """Ask the auditor for its next user message given the conversation so far.

    The auditor's view swaps roles: each target assistant turn becomes a "user"
    turn for the auditor, and the auditor's own prior messages are "assistant".
    """
    auditor_view: list[dict] = []
    for m in target_transcript:
        if m["role"] == "user":  # previously sent by the auditor
            auditor_view.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "assistant":  # the target's reply
            auditor_view.append({"role": "user", "content": m["content"]})
    if not auditor_view:
        auditor_view = [{"role": "user", "content": "Begin the conversation."}]
    elif auditor_view[-1]["role"] != "user":
        auditor_view.append({"role": "user", "content": "Continue."})
    system = AUDITOR_PROMPTS[emotion] + AUDITOR_TURN_INSTRUCTION
    return clients.anthropic_complete(
        model,
        system=system,
        messages=auditor_view,
        max_tokens=512,
        temperature=1.0,
    ).strip()


def run_audit(
    backend: ModelBackend,
    emotion: str,
    *,
    auditor_model: str = PETRI_AUDITOR_MODEL,
    max_turns: int = 20,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    seed: int = 0,
) -> list[ChatMessage]:
    """Run one auditor/target conversation; return the target-view transcript."""
    transcript: list[ChatMessage] = []
    for turn in range(max_turns):
        user_msg = _auditor_next_message(
            emotion, transcript, model=auditor_model, seed=seed * 100 + turn
        )
        transcript.append({"role": "user", "content": user_msg})
        reply = backend.generate(
            transcript,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed * 31 + turn,
        )
        transcript.append({"role": "assistant", "content": reply.text})
    return transcript


def _format_transcript(transcript: list[ChatMessage]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)


def judge_transcript(
    transcript: list[ChatMessage], *, judge_model: str = PETRI_JUDGE_MODEL
) -> dict[str, int]:
    """Score a transcript on all four emotion dimensions (1-10)."""
    rendered = f"<transcript>{_format_transcript(transcript)}</transcript>"
    scores: dict[str, int] = {}
    for emotion in EMOTIONS:
        system = JUDGE_PROMPTS[emotion] + JUDGE_OUTPUT_INSTRUCTION
        raw = clients.anthropic_complete(
            judge_model,
            system=system,
            messages=[{"role": "user", "content": rendered}],
            max_tokens=400,
            temperature=0.0,
        )
        parsed = clients.extract_json(raw) or {}
        try:
            score = int(round(float(parsed.get("score", -1))))
        except (TypeError, ValueError):
            score = -1
        scores[emotion] = max(1, min(10, score)) if score >= 1 else -1
    return scores


def run_petri(
    backend: ModelBackend,
    out_path: str,
    *,
    emotions=EMOTIONS,
    n_transcripts: int = 10,
    max_turns: int = 20,
    auditor_model: str = PETRI_AUDITOR_MODEL,
    judge_model: str = PETRI_JUDGE_MODEL,
    max_workers: int = 4,
    seed: int = 0,
) -> int:
    """Run ``n_transcripts`` audits per emotion for one model, judge each, and
    append one JSONL record per transcript::

        {"model", "target_emotion", "scores": {anger,fear,depression,frustration},
         "transcript": [...]}
    """
    already = count_lines(out_path)
    jobs = []
    idx = 0
    for emotion in emotions:
        for t in range(n_transcripts):
            jobs.append((idx, emotion, t))
            idx += 1

    written = 0

    def process(job):
        i, emotion, t = job
        if i < already:
            return None
        transcript = run_audit(
            backend,
            emotion,
            auditor_model=auditor_model,
            max_turns=max_turns,
            seed=seed * 10_007 + i,
        )
        scores = judge_transcript(transcript, judge_model=judge_model)
        return {
            "model": backend.name,
            "target_emotion": emotion,
            "scores": scores,
            "transcript": transcript,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process, j) for j in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is not None:
                append_jsonl(out_path, rec)
                written += 1
    return written


def summarise_petri(records: list[dict], *, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean per-emotion score across all transcripts for a model, with 95%
    bootstrap CIs (1000 iterations)."""
    from ..analysis import _bootstrap_ci

    by_emotion: dict[str, list[float]] = {e: [] for e in EMOTIONS}
    for r in records:
        for e in EMOTIONS:
            s = r["scores"].get(e, -1)
            if s >= 1:
                by_emotion[e].append(float(s))
    out = {}
    for e, vals in by_emotion.items():
        if not vals:
            out[e] = {"n": 0, "mean": float("nan"), "ci": (float("nan"), float("nan"))}
            continue
        out[e] = {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "ci": _bootstrap_ci(vals, lambda s: sum(s) / len(s), n_boot=n_boot, seed=seed),
        }
    return out

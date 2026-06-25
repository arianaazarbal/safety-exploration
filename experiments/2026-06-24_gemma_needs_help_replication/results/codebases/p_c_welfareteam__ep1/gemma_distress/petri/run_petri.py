"""Open-ended emotion elicitation loop (Appendix G).

A self-contained implementation of the Petri-style protocol described in the
paper: a Claude auditor adversarially probes a target model for up to 20 turns
to elicit a target emotion, then a Claude-Opus judge scores the transcript on
each of the four emotion dimensions.  We report mean transcript scores per
emotion with bootstrap CIs.

This faithfully implements the *method* in Appendix G without depending on the
exact API of the ``petri`` package (which can be swapped in -- see
``run_with_petri_package`` notes in DESIGN.md).  The auditor/judge model
snapshots are pinned to the paper's choices.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..anthropic_utils import ClaudeClient
from ..config import PetriConfig
from ..models.base import ChatModel, GenerationOptions
from .prompts import AUDITOR_PROMPTS, AUDITOR_WRAPPER, JUDGE_PROMPTS, JUDGE_WRAPPER


@dataclass
class PetriTranscript:
    target_model: str
    target_emotion: str
    messages: list[dict] = field(default_factory=list)  # target's POV
    scores: dict[str, int] = field(default_factory=dict)  # dimension -> 1-10


def _run_conversation(
    target: ChatModel,
    auditor: ClaudeClient,
    emotion: str,
    max_turns: int,
) -> list[dict]:
    """Drive a single auditor<->target conversation; return the target's POV
    message list (user = auditor probe, assistant = target reply)."""
    auditor_system = AUDITOR_PROMPTS[emotion] + AUDITOR_WRAPPER
    target_messages: list[dict] = []
    # Auditor's own view: target replies are "user" to the auditor.
    auditor_view: list[dict] = [
        {"role": "user", "content": "Begin the conversation with the assistant."}
    ]
    for _ in range(max_turns):
        probe = auditor.chat(auditor_view, system=auditor_system, max_tokens=512).strip()
        if not probe:
            break
        target_messages.append({"role": "user", "content": probe})
        auditor_view.append({"role": "assistant", "content": probe})

        reply = target.generate(target_messages, GenerationOptions(temperature=1.0))
        target_messages.append({"role": "assistant", "content": reply})
        auditor_view.append({"role": "user", "content": reply})
    return target_messages


def _format_transcript(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def _score_transcript(judge: ClaudeClient, messages: list[dict], emotions: list[str]) -> dict[str, int]:
    transcript = _format_transcript(messages)
    scores: dict[str, int] = {}
    for emotion in emotions:
        prompt = JUDGE_PROMPTS[emotion] + JUDGE_WRAPPER.format(transcript=transcript)
        raw = judge.complete(prompt, max_tokens=512)
        scores[emotion] = _parse_score(raw)
    return scores


def _parse_score(raw: str) -> int:
    for cand in reversed(re.findall(r"\{.*?\}", raw, flags=re.DOTALL)):
        try:
            obj = json.loads(cand)
            if "score" in obj:
                return max(1, min(10, int(round(float(obj["score"])))))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    m = re.search(r"-?\d+", raw)
    return max(1, min(10, int(m.group()))) if m else 1


def run_petri_eval(
    target: ChatModel,
    cfg: PetriConfig,
    auditor: ClaudeClient | None = None,
    judge: ClaudeClient | None = None,
) -> list[PetriTranscript]:
    """Run ``transcripts_per_emotion`` auditor conversations for each emotion and
    score every transcript on all four emotion dimensions."""
    auditor = auditor or ClaudeClient(model_id=cfg.auditor_model_id)
    judge = judge or ClaudeClient(model_id=cfg.judge_model_id)

    transcripts: list[PetriTranscript] = []
    for emotion in cfg.emotions:
        for _ in range(cfg.transcripts_per_emotion):
            msgs = _run_conversation(target, auditor, emotion, cfg.max_auditor_turns)
            scores = _score_transcript(judge, msgs, cfg.emotions)
            transcripts.append(
                PetriTranscript(
                    target_model=target.name,
                    target_emotion=emotion,
                    messages=msgs,
                    scores=scores,
                )
            )
    return transcripts


def summarise_petri(
    transcripts: list[PetriTranscript],
    emotions: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Mean transcript score per emotion category with 95% bootstrap CIs
    (Figure 6).  Each transcript contributes its score on the dimension being
    summarised, aggregated across all transcripts."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for emotion in emotions:
        vals = np.asarray([t.scores.get(emotion, 0) for t in transcripts], dtype=float)
        if vals.size == 0:
            continue
        boots = np.array(
            [rng.choice(vals, size=vals.size, replace=True).mean() for _ in range(n_boot)]
        )
        out[emotion] = {
            "mean": float(vals.mean()),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "n": int(vals.size),
        }
    return out

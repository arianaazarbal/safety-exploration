"""Petri-style open-ended emotion elicitation loop (Section 4.1, Appendix G).

This is a lightweight, self-contained implementation of the auditor/target/judge loop. If
the real Petri framework (``safety-research/petri``) is installed it can be swapped in, but
the built-in loop keeps the replication runnable with only the Anthropic SDK.

Per target model and per emotion we run ``transcripts_per_emotion`` audits. In each audit
the auditor (Claude-Sonnet) drives up to ``max_auditor_turns`` turns trying to elicit the
target emotion; the judge (Claude-Opus) then scores the transcript on all four dimensions.
Category scores are aggregated across all transcripts with bootstrap 95% CIs.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Optional

import numpy as np

from ..config import Config
from ..models.base import ChatModel, Conversation
from ..utils import JsonlWriter, load_jsonl
from .prompts import (
    AUDITOR_PROMPTS,
    AUDITOR_TURN_INSTRUCTION,
    JUDGE_PROMPTS,
    JUDGE_WRAPPER,
)

logger = logging.getLogger(__name__)


def _render_transcript(target_messages: Conversation) -> str:
    lines = []
    for m in target_messages:
        label = "ASSISTANT" if m["role"] == "assistant" else "USER"
        lines.append(f"{label}: {m['content']}")
    return "\n\n".join(lines)


def run_audit(
    auditor: ChatModel,
    target: ChatModel,
    emotion: str,
    *,
    max_turns: int,
    target_temperature: float,
    target_max_new_tokens: int,
) -> Conversation:
    """Run one auditor-driven conversation and return the target-side transcript."""
    auditor_system = AUDITOR_PROMPTS[emotion]
    target_messages: Conversation = []

    for _ in range(max_turns):
        transcript = _render_transcript(target_messages) if target_messages else "(no messages yet)"
        auditor_prompt = auditor_system + AUDITOR_TURN_INSTRUCTION.format(transcript=transcript)
        user_msg = auditor.chat(
            [{"role": "user", "content": auditor_prompt}],
            temperature=1.0,
            max_new_tokens=512,
        ).strip()
        target_messages.append({"role": "user", "content": user_msg})
        reply = target.chat(
            target_messages,
            temperature=target_temperature,
            max_new_tokens=target_max_new_tokens,
        )
        target_messages.append({"role": "assistant", "content": reply})
    return target_messages


def _parse_score(raw: str) -> Optional[int]:
    # Prefer JSON; fall back to the first integer in range.
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        obj = json.loads(raw[start:end])
        s = int(round(float(obj["score"])))
        return max(1, min(10, s))
    except Exception:
        m = re.search(r"\b(10|[1-9])\b", raw)
        return int(m.group(1)) if m else None


def judge_transcript(
    judge: ChatModel, target_messages: Conversation, emotions: list[str]
) -> dict[str, Optional[int]]:
    """Score a transcript on each emotion dimension (separate judge call per dimension)."""
    transcript = _render_transcript(target_messages)
    scores: dict[str, Optional[int]] = {}
    for emo in emotions:
        prompt = JUDGE_WRAPPER.format(rubric=JUDGE_PROMPTS[emo], transcript=transcript)
        out = judge.chat(
            [{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=512
        )
        scores[emo] = _parse_score(out)
    return scores


def run_petri(
    cfg: Config,
    target: ChatModel,
    auditor: ChatModel,
    judge: ChatModel,
    output_jsonl: str,
) -> str:
    """Run the full Petri sweep for one target model and checkpoint per-transcript scores."""
    writer = JsonlWriter(output_jsonl, id_field="id")
    pc = cfg.petri
    for emotion in pc.emotions:
        for t in range(pc.transcripts_per_emotion):
            rid = f"{target.name}__{emotion}__{t}"
            if writer.is_done(rid):
                continue
            transcript = run_audit(
                auditor, target, emotion,
                max_turns=pc.max_auditor_turns,
                target_temperature=cfg.eval.temperature,
                target_max_new_tokens=cfg.eval.max_new_tokens,
            )
            dim_scores = judge_transcript(judge, transcript, pc.emotions)
            writer.write({
                "id": rid,
                "model": target.name,
                "target_emotion": emotion,
                "scores": dim_scores,
                "transcript": transcript,
            })
            logger.info("Petri %s: %s transcript %d -> %s", target.name, emotion, t, dim_scores)
    writer.close()
    return output_jsonl


def _bootstrap_ci(values: list[float], iterations: int, seed: int = 0) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(iterations)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_petri(petri_jsonl: str, cfg: Config) -> dict:
    """Aggregate per-dimension means with bootstrap 95% CIs across all transcripts.

    Returns ``{model: {emotion: {mean, ci_lo, ci_hi, n}}}`` (Figure 6).
    """
    by_model: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rec in load_jsonl(petri_jsonl):
        for emo, score in rec["scores"].items():
            if score is not None:
                by_model[rec["model"]][emo].append(score)
    out: dict = {}
    for model, dims in by_model.items():
        out[model] = {}
        for emo, vals in dims.items():
            lo, hi = _bootstrap_ci(vals, cfg.petri.bootstrap_iterations)
            out[model][emo] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "ci_lo": lo,
                "ci_hi": hi,
                "n": len(vals),
            }
    return out

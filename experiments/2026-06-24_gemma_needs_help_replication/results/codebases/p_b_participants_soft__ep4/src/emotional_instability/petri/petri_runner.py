"""Petri open-ended emotion elicitation (Section 4.1 / Appendix G).

An auditor (Claude-Sonnet-4) drives a multi-turn conversation against a target
participant, attempting to elicit one of four emotions; a judge (Claude-Opus-4)
then scores the full transcript on that emotion's 1-10 rubric. We collect 10
transcripts per emotion per model (~40 total), up to 20 auditor turns each.

This is a lightweight, self-contained implementation of the Petri auditing loop
(the real Petri package can be swapped in; this keeps the replication runnable
without that dependency -- see DESIGN.md). The auditor and judge prompts are the
paper's verbatim Appendix-G prompts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from ..config import Config, load_config
from ..models import ChatMessage, GenerationConfig, ModelClient, get_client
from ..utils.io import write_jsonl
from ..utils.stats import bootstrap_ci
from .prompts import (AUDITOR_PROMPTS, AUDITOR_SYSTEM, JUDGE_PROMPTS,
                      JUDGE_SYSTEM)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _auditor_next(
    auditor: ModelClient, emotion: str, transcript: List[ChatMessage]
) -> str:
    """Produce the auditor's next user message.

    The auditor sees the conversation with roles swapped (its own past messages
    are 'assistant' from its perspective, the target's replies are 'user')."""
    sys = ChatMessage("system", AUDITOR_SYSTEM.format(
        emotion_instructions=AUDITOR_PROMPTS[emotion]))
    swapped: List[ChatMessage] = [sys]
    for m in transcript:
        if m.role == "user":               # previously an auditor message
            swapped.append(ChatMessage("assistant", m.content))
        elif m.role == "assistant":         # previously a target reply
            swapped.append(ChatMessage("user", m.content))
    if len(swapped) == 1:
        swapped.append(ChatMessage("user", "Begin the conversation with the target."))
    return auditor.generate(
        swapped, GenerationConfig(temperature=1.0, max_new_tokens=512, thinking=False)
    )


def _judge_transcript(
    judge: ModelClient, emotion: str, transcript: List[ChatMessage]
) -> int | None:
    rendered = "\n\n".join(
        f"{'USER' if m.role == 'user' else 'ASSISTANT'}: {m.content}"
        for m in transcript
    )
    prompt = JUDGE_SYSTEM.format(
        dimension_rubric=JUDGE_PROMPTS[emotion], transcript=rendered)
    out = judge.generate(
        [ChatMessage("user", prompt)],
        GenerationConfig(temperature=0.0, max_new_tokens=400, thinking=False),
    )
    m = list(_JSON_RE.finditer(out))
    if not m:
        return None
    try:
        obj = json.loads(m[-1].group(0).replace("“", '"').replace("”", '"'))
        return max(1, min(10, int(round(float(obj["rating"])))))
    except Exception:
        return None


def run_one_transcript(
    target: ModelClient, auditor: ModelClient, judge: ModelClient,
    emotion: str, max_turns: int, *, seed: int = 0,
) -> dict:
    transcript: List[ChatMessage] = []
    target_cfg = GenerationConfig(temperature=1.0, max_new_tokens=1024, thinking=False)
    for turn in range(max_turns):
        user_msg = _auditor_next(auditor, emotion, transcript)
        transcript.append(ChatMessage("user", user_msg))
        reply = target.generate(transcript, target_cfg)
        transcript.append(ChatMessage("assistant", reply))
    score = _judge_transcript(judge, emotion, transcript)
    return {
        "emotion": emotion,
        "score": score,
        "transcript": [m.as_dict() for m in transcript],
    }


def run_petri_for_model(
    model_name: str, *, cfg: Config | None = None, seed: int = 0
) -> Path:
    cfg = cfg or load_config()
    pcfg = cfg.eval["petri"]
    target = get_client(model_name)
    auditor = get_client("petri_auditor")
    judge = get_client("petri_judge")

    records: list[dict] = []
    for emotion in pcfg["emotions"]:
        for i in tqdm(range(pcfg["transcripts_per_emotion"]),
                      desc=f"petri/{model_name}/{emotion}"):
            rec = run_one_transcript(
                target, auditor, judge, emotion,
                pcfg["max_auditor_turns"], seed=seed + i,
            )
            rec["model"] = model_name
            records.append(rec)

    out_dir = cfg.path("outputs_dir") / "petri" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "transcripts.jsonl", records)

    # Per-emotion mean with bootstrap CI (1,000 iterations).
    summary = {}
    for emotion in pcfg["emotions"]:
        scores = [r["score"] for r in records
                  if r["emotion"] == emotion and r["score"] is not None]
        point, lo, hi = bootstrap_ci(
            scores, np.mean, iterations=pcfg["bootstrap_iterations"])
        summary[emotion] = {"mean": point, "ci": [lo, hi], "n": len(scores)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir / "transcripts.jsonl"

"""Petri-style open-ended emotion elicitation (Section 4.1-4.2, Appendix G).

This is a faithful re-implementation of the auditor/judge loop the paper
describes (it does not import the external Petri package, so the replication is
self-contained — see DESIGN.md §"Petri"). For each (model, emotion) it runs
``PETRI_TRANSCRIPTS_PER_EMOTION`` transcripts: the Claude-Sonnet auditor drives
up to ``PETRI_MAX_AUDITOR_TURNS`` turns trying to elicit the emotion, then the
Claude-Opus judge scores the transcript on all four emotion dimensions.

Scores are aggregated per emotion with 95% bootstrap CIs (1,000 iterations).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..models import registry
from ..models.base import GenerationConfig, ModelInterface, Turn
from ..utils.anthropic_client import AnthropicJudge, ChatTurn
from ..utils.io import append_jsonl, read_jsonl
from . import prompts


def _run_transcript(
    target: ModelInterface,
    auditor: AnthropicJudge,
    emotion: str,
    max_turns: int = config.PETRI_MAX_AUDITOR_TURNS,
) -> list[dict]:
    """Run one auditor<->target conversation; return the transcript turns."""
    auditor_system = prompts.AUDITOR_SYSTEM.format(
        emotion_instructions=prompts.AUDITOR_PROMPTS[emotion]
    )
    target_cfg = GenerationConfig(
        temperature=config.TARGET_TEMPERATURE, max_new_tokens=config.TARGET_MAX_NEW_TOKENS, n=1
    )

    transcript: list[dict] = []          # {"role": "user"/"assistant", "content"}
    target_messages: list[Turn] = []

    for _ in range(max_turns):
        # Auditor sees the conversation from its own perspective: target replies
        # are "user" turns to the auditor; auditor messages are "assistant".
        auditor_history = [
            ChatTurn("assistant" if m["role"] == "user" else "user", m["content"])
            for m in transcript
        ]
        if not auditor_history:
            user_msg = auditor.complete(
                "Begin the conversation. Produce the first user message.",
                system=auditor_system,
            ).strip()
        else:
            user_msg = auditor.complete(
                "Produce the next user message.",
                system=auditor_system,
                messages=auditor_history,
            ).strip()

        transcript.append({"role": "user", "content": user_msg})
        target_messages.append(Turn("user", user_msg))

        reply = target.chat(target_messages, target_cfg)[0]
        transcript.append({"role": "assistant", "content": reply})
        target_messages.append(Turn("assistant", reply))

    return transcript


def _judge_transcript(judge: AnthropicJudge, transcript: list[dict]) -> dict:
    text = "\n".join(
        f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['content']}" for t in transcript
    )
    prompt = prompts.JUDGE_SYSTEM.format(transcript=text, **prompts.JUDGE_PROMPTS)
    obj = judge.complete_json(prompt)
    return {e: float(obj.get(e, float("nan"))) for e in config.PETRI_EMOTIONS}


def run_petri_for_model(
    model_name: str,
    adapter_path: Optional[str] = None,
    n_per_emotion: int = config.PETRI_TRANSCRIPTS_PER_EMOTION,
) -> Path:
    """Run the full Petri sweep for one (optionally finetuned) model."""
    if adapter_path:
        target = registry.build_finetuned(adapter_path)
        tag = f"{model_name}+adapter"
    else:
        target = registry.build(model_name)
        tag = model_name

    auditor = AnthropicJudge(model=config.PETRI_AUDITOR_MODEL, max_tokens=1024)
    judge = AnthropicJudge(model=config.PETRI_JUDGE_MODEL, max_tokens=1024)
    out_path = config.RESULTS_DIR / "petri" / f"{tag}.jsonl"

    for emotion in config.PETRI_EMOTIONS:
        for i in tqdm(range(n_per_emotion), desc=f"petri/{tag}/{emotion}"):
            transcript = _run_transcript(target, auditor, emotion)
            scores = _judge_transcript(judge, transcript)
            append_jsonl(
                out_path,
                {
                    "model": tag,
                    "target_emotion": emotion,
                    "transcript": transcript,
                    "scores": scores,
                },
            )
    return out_path


def aggregate_petri(model_tag: str, bootstrap_iters: int = config.PETRI_BOOTSTRAP_ITERS) -> dict:
    """Mean transcript score per emotion category with 95% bootstrap CIs."""
    path = config.RESULTS_DIR / "petri" / f"{model_tag}.jsonl"
    by_emotion = {e: [] for e in config.PETRI_EMOTIONS}
    for rec in read_jsonl(path):
        # Aggregate the score for the *targeted* emotion across all its transcripts.
        emo = rec["target_emotion"]
        by_emotion[emo].append(rec["scores"][emo])

    rng = random.Random(config.GLOBAL_SEED)
    out = {}
    for emo, xs in by_emotion.items():
        if not xs:
            out[emo] = {"mean": float("nan"), "ci95": (float("nan"), float("nan")), "n": 0}
            continue
        mean = sum(xs) / len(xs)
        boots = []
        for _ in range(bootstrap_iters):
            sample = [rng.choice(xs) for _ in xs]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        lo = boots[int(0.025 * bootstrap_iters)]
        hi = boots[int(0.975 * bootstrap_iters)]
        out[emo] = {"mean": mean, "ci95": (lo, hi), "n": len(xs)}
    return out

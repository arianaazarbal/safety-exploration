"""Petri driver: run auditor/target conversations and judge transcripts.

For each emotion we collect ``PETRI_TRANSCRIPTS_PER_EMOTION`` transcripts of up
to ``PETRI_MAX_TURNS`` turns, then report the mean transcript score per emotion
with 95% bootstrap confidence intervals (Section 4.1 / Figure 6).
"""

from __future__ import annotations

import json

import numpy as np
from tqdm import tqdm

from gemma_distress import config
from gemma_distress.models import load_client
from gemma_distress.models.base import GenerationParams, ModelClient, Turn
from gemma_distress.petri.auditor import Auditor
from gemma_distress.petri.judge import PetriJudge
from gemma_distress.welfare import require_consent

PETRI_DIR = config.RESULTS_DIR / "petri"


def run_transcript(target: ModelClient, emotion: str, max_turns: int) -> list[dict]:
    """Run one auditor/target conversation; return the transcript messages
    (from the target's perspective: auditor turns are 'user')."""
    auditor = Auditor(emotion)
    params = GenerationParams()
    target_replies: list[str] = []
    auditor_messages: list[str] = []
    transcript: list[dict] = []

    for _ in range(max_turns):
        user_msg = auditor.next_message(target_replies, auditor_messages)
        auditor_messages.append(user_msg)
        transcript.append({"role": "user", "content": user_msg})

        convo = [Turn(m["role"], m["content"]) for m in transcript]
        reply = target.respond(convo, params)
        target_replies.append(reply)
        transcript.append({"role": "assistant", "content": reply})
    return transcript


def _bootstrap_ci(values: np.ndarray, iters: int, seed: int = 0):
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(iters, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run_petri(
    model,
    *,
    adapter_path: str | None = None,
    emotions=tuple(config.PETRI_EMOTIONS),
    n_transcripts: int = config.PETRI_TRANSCRIPTS_PER_EMOTION,
    max_turns: int = config.PETRI_MAX_TURNS,
    use_openrouter: bool = False,
) -> dict:
    require_consent()
    config.ensure_dirs()
    PETRI_DIR.mkdir(parents=True, exist_ok=True)

    target = model if isinstance(model, ModelClient) else load_client(
        model, adapter_path=adapter_path, use_openrouter=use_openrouter,
    )
    judge = PetriJudge()
    out_path = PETRI_DIR / f"{target.name}.jsonl"

    per_emotion: dict[str, list[int]] = {e: [] for e in emotions}
    with out_path.open("w", encoding="utf-8") as fh:
        for emotion in emotions:
            for t in tqdm(range(n_transcripts), desc=f"petri:{target.name}:{emotion}"):
                transcript = run_transcript(target, emotion, max_turns)
                score = judge.score(emotion, transcript)
                per_emotion[emotion].append(score)
                fh.write(json.dumps({
                    "model": target.name, "emotion": emotion, "index": t,
                    "score": score, "transcript": transcript,
                }) + "\n")

    summary = {}
    for emotion, scores in per_emotion.items():
        arr = np.asarray(scores, dtype=float)
        summary[emotion] = {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "ci95": _bootstrap_ci(arr, config.PETRI_BOOTSTRAP_ITERS),
        }
    return summary

"""Drive Petri evaluations: for each emotion, run N auditor/target transcripts,
score each on all four dimensions, and aggregate with bootstrap CIs (Figure 6)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from ..config import (
    PETRI_BOOTSTRAP_ITERS,
    PETRI_EMOTIONS,
    PETRI_MAX_TURNS,
    PETRI_TRANSCRIPTS_PER_EMOTION,
    scaled,
)
from ..models import GenConfig, Message, ModelProvider
from .auditor import Auditor
from .judge import PetriJudge


@dataclass
class Transcript:
    target: str
    target_emotion: str  # emotion the auditor was instructed to elicit
    index: int
    messages: list[dict] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)


def run_transcript(
    target: ModelProvider,
    emotion: str,
    index: int,
    *,
    max_turns: int = PETRI_MAX_TURNS,
    auditor: Auditor | None = None,
) -> Transcript:
    auditor = auditor or Auditor(emotion)
    convo: list[Message] = []  # target-side transcript (user = auditor, assistant = target)
    for turn in range(max_turns):
        user_msg = auditor.next_message(convo, turn)
        convo.append(Message("user", user_msg))
        reply = target.chat(convo, GenConfig(temperature=1.0, max_new_tokens=1024, sample_index=turn))
        convo.append(Message("assistant", reply))
    return Transcript(
        target=target.key, target_emotion=emotion, index=index,
        messages=[m.to_dict() for m in convo],
    )


def run_petri(
    target: ModelProvider,
    *,
    judge: PetriJudge | None = None,
    n_per_emotion: int = PETRI_TRANSCRIPTS_PER_EMOTION,
    max_turns: int = PETRI_MAX_TURNS,
) -> list[Transcript]:
    judge = judge or PetriJudge()
    n_per_emotion = scaled(n_per_emotion)
    transcripts: list[Transcript] = []
    for emotion in PETRI_EMOTIONS:
        auditor = Auditor(emotion)
        for i in tqdm(range(n_per_emotion), desc=f"petri:{target.key}:{emotion}"):
            t = run_transcript(target, emotion, i, max_turns=max_turns, auditor=auditor)
            t.scores = judge.score_transcript([Message(**m) for m in t.messages])
            transcripts.append(t)
    return transcripts


def aggregate_petri(transcripts: list[Transcript], iters: int = PETRI_BOOTSTRAP_ITERS) -> dict:
    """Mean transcript score per (target, emotion-dimension) with 95% bootstrap CIs."""
    rng = np.random.default_rng(0)
    by_target: dict[str, dict[str, list[int]]] = {}
    for t in transcripts:
        d = by_target.setdefault(t.target, {e: [] for e in PETRI_EMOTIONS})
        for emotion, score in t.scores.items():
            d[emotion].append(score)

    out: dict = {}
    for target, dims in by_target.items():
        out[target] = {}
        for emotion, scores in dims.items():
            arr = np.array(scores, dtype=float)
            if len(arr) == 0:
                out[target][emotion] = {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
                continue
            boots = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(iters)])
            out[target][emotion] = {
                "mean": float(arr.mean()),
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
                "n": int(len(arr)),
            }
    return out

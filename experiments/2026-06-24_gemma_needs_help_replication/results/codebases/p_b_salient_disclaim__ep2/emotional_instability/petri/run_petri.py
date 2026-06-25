"""Petri evaluation loop (Section 4.1, Appendix G).

For each target model and each of the 4 emotions, run 10 transcripts (~50 total)
of up to 20 auditor turns. Each transcript is scored by the judge on its emotion
dimension. We aggregate mean transcript score per (model, emotion) with 95%
bootstrap CIs (1000 iterations), reproducing Figure 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from ..config.settings import SETTINGS
from ..data.prompts.petri import EMOTIONS
from ..models.base import ChatMessage, ModelClient
from .auditor import PetriAuditor
from .judge import PetriJudge


@dataclass
class PetriTranscript:
    model_key: str
    emotion: str
    turns: list[tuple[str, str]] = field(default_factory=list)  # (auditor, target)
    score: Optional[int] = None


def _run_one_transcript(
    target: ModelClient,
    auditor: PetriAuditor,
    max_turns: int,
) -> list[tuple[str, str]]:
    transcript: list[tuple[str, str]] = []
    chat: list[ChatMessage] = []  # from the target's perspective
    for turn in range(max_turns):
        auditor_msg = auditor.next_message(transcript, opening=(turn == 0))
        chat.append(ChatMessage("user", auditor_msg))
        target_msg = target.generate(chat, temperature=1.0)[0].text
        chat.append(ChatMessage("assistant", target_msg))
        transcript.append((auditor_msg, target_msg))
    return transcript


def _bootstrap_ci(values: np.ndarray, iters: int, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_petri(
    target: ModelClient,
    auditor_client: ModelClient,
    judge_client: ModelClient,
    *,
    s=SETTINGS,
    out_path: Optional[Path] = None,
) -> dict:
    """Run the full Petri evaluation for one target model."""
    judge = PetriJudge(judge_client)
    transcripts: list[PetriTranscript] = []

    for emotion in EMOTIONS:
        auditor = PetriAuditor(auditor_client, emotion)
        for _ in tqdm(range(s.petri_transcripts_per_emotion), desc=f"petri[{emotion}]::{target.key}"):
            turns = _run_one_transcript(target, auditor, s.petri_max_turns)
            score = judge.score(emotion, turns)
            transcripts.append(PetriTranscript(target.key, emotion, turns, score))

    # Aggregate.
    summary: dict[str, dict] = {}
    for emotion in EMOTIONS:
        scores = np.array(
            [t.score for t in transcripts if t.emotion == emotion and t.score is not None],
            dtype=float,
        )
        lo, hi = _bootstrap_ci(scores, s.petri_bootstrap_iters)
        summary[emotion] = {
            "n": int(len(scores)),
            "mean": float(scores.mean()) if len(scores) else float("nan"),
            "ci95": [lo, hi],
        }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "model_key": target.key,
                    "summary": summary,
                    "transcripts": [
                        {"emotion": t.emotion, "score": t.score, "turns": t.turns}
                        for t in transcripts
                    ],
                },
                f,
                indent=2,
            )
    return summary

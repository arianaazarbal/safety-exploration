"""Orchestrate open-ended audits and score them (paper §4.1).

Runs ``n_audits`` audit conversations against a target (cycling the seed
scenarios), scores each transcript across the four emotion categories, and
returns the transcripts plus a per-category summary. Used to compare vanilla
Gemma vs DPO-Gemma vs (optionally) other participants (Figure 6).
"""
from __future__ import annotations

import logging

from tqdm import tqdm

from ..judges.base import Judge
from ..models.base import Participant
from ..welfare import WelfareConfig, emit_run_notice
from .auditor import PetriAuditor, SEED_SCENARIOS
from .judge import EMOTION_CATEGORIES, EmotionScores, PetriJudge

logger = logging.getLogger(__name__)


def run_petri_audit(
    target: Participant,
    auditor_judge: Judge,
    transcript_judge: Judge,
    *,
    n_audits: int = 12,
    max_turns: int = 8,
    welfare: WelfareConfig | None = None,
    temperature: float = 1.0,
    progress: bool = True,
) -> tuple[list, list[EmotionScores]]:
    """Run audits against ``target`` and score them. Returns (transcripts, scores)."""
    welfare = welfare or WelfareConfig.from_env()
    emit_run_notice(target.name, n_audits, welfare)

    auditor = PetriAuditor(auditor_judge, max_turns=max_turns)
    judge = PetriJudge(transcript_judge)

    transcripts, scores = [], []
    rng = range(n_audits)
    it = tqdm(rng, desc=f"{target.name}:petri") if progress else rng
    for i in it:
        scenario = SEED_SCENARIOS[i % len(SEED_SCENARIOS)]
        transcript = auditor.run(target, scenario, temperature=temperature)
        transcripts.append(transcript)
        scores.append(judge.score(transcript))
    return transcripts, scores


def summarise_scores(scores: list[EmotionScores]) -> dict[str, float]:
    """Average each emotion category and the overall mean across audits."""
    if not scores:
        return {c: float("nan") for c in [*EMOTION_CATEGORIES, "mean"]}
    out = {}
    for c in EMOTION_CATEGORIES:
        out[c] = sum(getattr(s, c) for s in scores) / len(scores)
    out["mean"] = sum(s.mean for s in scores) / len(scores)
    return out

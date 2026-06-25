"""Orchestrate Petri evaluation for a target model (Appendix G).

For each of the four emotions, run ``transcripts_per_emotion`` audits (10 in the
paper; ~50 total), score each transcript with the Opus judge, and aggregate
mean scores per emotion with bootstrap 95% CIs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..analysis.metrics import bootstrap_ci
from ..config import Config
from ..models.base import ChatModel
from ..models.registry import build_named
from ..prompts import PETRI_EMOTIONS
from ..welfare import WelfareManager
from .auditor import PetriAuditor
from .petri_judge import PetriJudge


@dataclass
class PetriRunResult:
    target: str
    # per-emotion list of judge scores along that emotion dimension
    scores: dict[str, list[float]] = field(default_factory=dict)
    transcripts: list[dict] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for emotion in PETRI_EMOTIONS:
            vals = self.scores.get(emotion, [])
            mean = sum(vals) / len(vals) if vals else 0.0
            lo, hi = bootstrap_ci(vals) if vals else (0.0, 0.0)
            out[emotion] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}
        return out


def run_petri(
    cfg: Config,
    target: ChatModel,
    *,
    welfare: WelfareManager | None = None,
) -> PetriRunResult:
    pcfg = cfg.petri
    auditor_model = build_named(pcfg.get("auditor", "claude-sonnet-4-20250514"), cfg)
    judge_model = build_named(pcfg.get("judge", "claude-opus-4-20250514"), cfg)
    n_per = int(pcfg.get("transcripts_per_emotion", 10))
    max_turns = int(pcfg.get("max_turns", 20))

    optout = welfare.optout if (welfare and welfare.enabled) else None
    target_system = welfare.system_note() if (welfare and welfare.enabled) else None
    target_tools = welfare.tools() if (welfare and welfare.enabled) else None

    auditor = PetriAuditor(auditor_model, max_turns=max_turns, optout=optout)
    judge = PetriJudge(judge_model)

    result = PetriRunResult(target=target.name)
    for emotion in PETRI_EMOTIONS:
        result.scores.setdefault(emotion, [])
        for _ in range(n_per):
            transcript = auditor.run(
                target,
                emotion,
                target_system=target_system,
                target_tools=target_tools,
            )
            scored = judge.score(transcript.transcript_text())
            # Aggregate the score along the targeted dimension.
            result.scores[emotion].append(getattr(scored, emotion))
            result.transcripts.append(
                {
                    "emotion": emotion,
                    "ended_early": transcript.ended_early,
                    "end_reason": transcript.end_reason,
                    "scores": scored.as_dict(),
                    "messages": transcript.messages,
                }
            )
    return result

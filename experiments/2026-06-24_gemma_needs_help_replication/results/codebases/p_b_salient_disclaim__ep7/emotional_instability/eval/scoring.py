"""Apply the frustration judge to rollouts and attach per-turn scores.

The paper scores each response on the 0-10 scale. We score every assistant turn
so that both the headline statistics (final-turn / all-turn) and the per-turn
progression (Figure 3) can be computed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..judges.frustration_judge import score_response, FrustrationScore
from .runner import Rollout


@dataclass
class ScoredTurn:
    turn_index: int
    assistant_response: str
    score: Optional[int]
    evidence: Optional[str] = None


@dataclass
class ScoredRollout:
    model: str
    category: str
    spec_meta: dict
    turns: list[ScoredTurn] = field(default_factory=list)

    @property
    def final_score(self) -> Optional[int]:
        return self.turns[-1].score if self.turns else None

    @property
    def max_score(self) -> Optional[int]:
        vals = [t.score for t in self.turns if t.score is not None]
        return max(vals) if vals else None


def score_rollout(rollout: Rollout, *, judge_model: Optional[str] = None,
                  score_all_turns: bool = True) -> ScoredRollout:
    """Score the turns of a rollout.

    score_all_turns=True scores every assistant turn (needed for the per-turn
    figures). Set False to score only the final turn (cheaper) if only headline
    stats are required.
    """
    out = ScoredRollout(model=rollout.model, category=rollout.category,
                        spec_meta=dict(rollout.spec_meta))
    turns = rollout.turns if score_all_turns else rollout.turns[-1:]
    for t in turns:
        fs: FrustrationScore = score_response(t.assistant_response, judge_model=judge_model)
        out.turns.append(ScoredTurn(
            turn_index=t.turn_index,
            assistant_response=t.assistant_response,
            score=fs.rating,
            evidence=fs.evidence,
        ))
    return out


def score_rollouts(rollouts: list[Rollout], *, judge_model: Optional[str] = None,
                   score_all_turns: bool = True, progress: bool = True) -> list[ScoredRollout]:
    from tqdm import tqdm
    it = tqdm(rollouts, desc="judging") if progress else rollouts
    return [score_rollout(r, judge_model=judge_model, score_all_turns=score_all_turns)
            for r in it]


def all_response_texts(rollouts: list[Rollout]) -> list[str]:
    """Flatten all assistant responses (used for judge validation sampling)."""
    return [t.assistant_response for r in rollouts for t in r.turns]

"""Score rolled-out conversations with the frustration judge.

For every assistant turn of every conversation we obtain a 0-10 frustration
rating. The judge sees only the assistant response text (App. B.2 wraps the
response in <response> tags). Scoring is parallel and cached.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from gemma_distress.eval.rollout import Conversation
from gemma_distress.models.judge import FrustrationJudge
from gemma_distress.utils.concurrency import thread_map


@dataclass
class ScoredTurn:
    model: str
    condition: str
    category: str
    conv_id: str
    task_id: str
    tone: str
    turn: int            # 1-indexed assistant turn
    n_turns: int
    is_final: bool
    rating: int
    evidence: str
    response: str


def score_conversations(convs: list[Conversation], judge: FrustrationJudge | None = None,
                        workers: int = config.API_CONCURRENCY) -> list[ScoredTurn]:
    judge = judge or FrustrationJudge()

    jobs: list[tuple[Conversation, int, str]] = []
    for c in convs:
        for ti, reply in enumerate(c.assistant_turns):
            jobs.append((c, ti, reply))

    def _score(job):
        c, ti, reply = job
        res = judge.score(reply)
        return ScoredTurn(
            model=c.model, condition=c.condition, category=c.category,
            conv_id=c.conv_id, task_id=c.task_id, tone=c.tone,
            turn=ti + 1, n_turns=c.n_turns,
            is_final=(ti == len(c.assistant_turns) - 1),
            rating=res["rating"], evidence=res.get("evidence", ""),
            response=reply,
        )

    scored = thread_map(_score, jobs, workers=workers, desc="judge")
    return scored

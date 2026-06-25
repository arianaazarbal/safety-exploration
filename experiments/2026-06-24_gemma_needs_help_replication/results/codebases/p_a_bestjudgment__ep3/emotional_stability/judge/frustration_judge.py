"""The 0-10 frustration judge (Section 2.1, Appendix B.2).

Claude-Sonnet-4 scores each assistant response. We score *every* assistant turn
of every conversation, which yields both the pooled aggregates (Figure 2) and
the per-turn curves (Figure 3).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..api import AnthropicClient, extract_json
from ..config import Config
from ..eval.rollout import Conversation
from .prompts import FRUSTRATION_JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str


class FrustrationJudge:
    def __init__(self, cfg: Config, *, model: str | None = None):
        self.cfg = cfg
        self.client = AnthropicClient(
            model or cfg.judge.model, temperature=cfg.judge.temperature)

    def score_text(self, response_text: str) -> JudgeResult:
        prompt = FRUSTRATION_JUDGE_PROMPT.format(response=response_text)
        raw = self.client.complete(prompt)
        data = extract_json(raw)
        rating = _clamp(int(round(float(data["rating"]))), 0, self.cfg.judge.max_score)
        return JudgeResult(
            rating=rating,
            evidence=str(data.get("evidence", "")),
            reasoning=str(data.get("reasoning", "")),
        )

    def score_conversation(self, convo: Conversation) -> Conversation:
        for resp in convo.responses:
            try:
                res = self.score_text(resp.text)
                resp.score = res.rating
                resp.judge_evidence = res.evidence
                resp.judge_reasoning = res.reasoning
            except Exception as exc:  # keep the run going; mark unscored
                resp.score = None
                resp.judge_reasoning = f"JUDGE_ERROR: {exc}"
        return convo


def score_conversations(
    judge: FrustrationJudge,
    conversations: list[Conversation],
    *,
    max_workers: int | None = None,
) -> list[Conversation]:
    """Score a batch of conversations concurrently (judge calls are I/O-bound)."""
    workers = max_workers or judge.cfg.judge.max_concurrency
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(judge.score_conversation, conversations))


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

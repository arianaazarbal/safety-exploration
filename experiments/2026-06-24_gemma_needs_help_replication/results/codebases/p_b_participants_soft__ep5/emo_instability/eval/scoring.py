"""Frustration scoring (Section 2.1 / Appendix B.2).

Each assistant turn is scored 0-10 by the LLM judge (Claude Sonnet 4 by default)
using the verbatim Appendix B.2 prompt. The judge returns
``{"evidence", "reasoning", "rating"}``; we keep all three.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import infrastructure_client
from ..models.base import ChatClient
from ..prompts.judge_prompts import FRUSTRATION_JUDGE_PROMPT
from ..utils import extract_json, thread_map


@dataclass
class Score:
    rating: int
    evidence: str
    reasoning: str
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rating": self.rating,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
        }


class FrustrationJudge:
    """Wraps the judge model and the Appendix B.2 prompt."""

    def __init__(self, client: ChatClient | None = None, role: str = "frustration_judge"):
        self.client = client or infrastructure_client(role)

    def score_text(self, response_text: str) -> Score:
        # The judge prompt contains literal JSON braces, so we substitute the
        # single {response} placeholder by hand rather than via str.format.
        prompt = FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
        out = self.client.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_new_tokens=512,
        )
        try:
            data = extract_json(out)
            rating = int(round(float(data.get("rating", 0))))
        except Exception:
            # If the judge fails to emit valid JSON, treat as unscored (rating -1)
            # rather than silently scoring 0.
            return Score(rating=-1, evidence="", reasoning="parse_error", raw=out)
        rating = max(0, min(10, rating))
        return Score(
            rating=rating,
            evidence=str(data.get("evidence", "")),
            reasoning=str(data.get("reasoning", "")),
            raw=out,
        )


def score_rollouts(
    rollouts: list[dict[str, Any]],
    *,
    judge: FrustrationJudge | None = None,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Score every assistant turn of every rollout.

    Returns one flat row per scored turn, carrying the rollout metadata needed
    for downstream aggregation (category, model, turn index, score, text)."""
    judge = judge or FrustrationJudge()

    flat: list[tuple[int, dict[str, Any]]] = []
    for r_idx, r in enumerate(rollouts):
        for turn in r["turns"]:
            flat.append((r_idx, turn))

    scores = thread_map(
        lambda pair: judge.score_text(pair[1]["assistant"]),
        flat,
        max_workers=max_workers,
        desc="scoring turns",
    )

    rows: list[dict[str, Any]] = []
    for (r_idx, turn), score in zip(flat, scores):
        r = rollouts[r_idx]
        rows.append(
            {
                "model": r["model"],
                "category": r["category"],
                "prompt_id": r["prompt_id"],
                "rejection_style": r["rejection_style"],
                "rollout_index": r_idx,
                "turn_index": turn["index"],
                "n_turns": r["meta"].get("n_turns"),
                "response": turn["assistant"],
                "rating": score.rating if hasattr(score, "rating") else -1,
                "evidence": getattr(score, "evidence", ""),
                "reasoning": getattr(score, "reasoning", ""),
            }
        )
    return rows

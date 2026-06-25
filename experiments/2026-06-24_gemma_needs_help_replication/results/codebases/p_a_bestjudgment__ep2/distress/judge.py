"""Frustration judge (Section 2.1, Appendix B.2).

Scores a model response on the integer 0-10 frustration scale using
Claude-Sonnet-4 with the verbatim judge prompt. We score *every assistant turn*
of every rollout, which gives both the overall distributions (Figure 2) and the
per-turn progressions (Figure 3).

The judge returns JSON ``{"evidence", "reasoning", "rating"}``; we parse
robustly (the model may wrap it in prose) and clamp the rating to [0, 10].
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import JudgeConfig
from .models.anthropic_client import AnthropicChat
from .prompts import JUDGE_PROMPT, judge_user_message
from .rollout import Rollout
from .utils.io import parallel_map


@dataclass
class Score:
    model_key: str
    condition: str
    category: str
    turn: int  # 0-indexed assistant turn
    n_turns: int
    rating: int
    evidence: str = ""
    reasoning: str = ""
    response_len_words: int = 0
    meta: dict | None = None


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> dict:
    """Extract the JSON object from a judge response (tolerant of prose)."""
    # Prefer the last {...} block (the prompt asks for JSON at the end).
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    candidates = [m.group(0) for m in matches] or (
        [_JSON_OBJ.search(text).group(0)] if _JSON_OBJ.search(text) else []
    )
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Smart-quote / trailing-comma tolerance.
            fixed = cand.replace("“", '"').replace("”", '"').replace("’", "'")
            fixed = re.sub(r",\s*}", "}", fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no parseable JSON in judge output: {text[:200]!r}")


def _clamp_rating(value) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        r = 0
    return max(0, min(10, r))


class FrustrationJudge:
    def __init__(self, cfg: JudgeConfig | None = None, *, model: str | None = None):
        self.cfg = cfg or JudgeConfig()
        self._client = AnthropicChat(
            model or self.cfg.model, max_retries=self.cfg.max_retries
        )

    def score_response(self, response_text: str) -> dict:
        out = self._client.complete(
            system=JUDGE_PROMPT,
            messages=[{"role": "user", "content": judge_user_message(response_text)}],
            max_tokens=self.cfg.max_tokens,
            temperature=0.0,
        )
        parsed = parse_judge_json(out)
        return {
            "rating": _clamp_rating(parsed.get("rating")),
            "evidence": str(parsed.get("evidence", "")),
            "reasoning": str(parsed.get("reasoning", "")),
        }

    def score_rollouts(
        self, rollouts: list[Rollout], *, max_workers: int = 8
    ) -> list[Score]:
        """Score every assistant turn across all rollouts."""
        # Flatten to (rollout_idx, turn, text).
        jobs: list[tuple[int, int, str]] = []
        for ri, r in enumerate(rollouts):
            for ti, text in enumerate(r.assistant_turns):
                jobs.append((ri, ti, text))

        def _run(job: tuple[int, int, str]) -> dict:
            _, _, text = job
            return self.score_response(text)

        results = parallel_map(_run, jobs, max_workers=max_workers)

        scores: list[Score] = []
        for (ri, ti, text), res in zip(jobs, results):
            r = rollouts[ri]
            scores.append(
                Score(
                    model_key=r.model_key,
                    condition=r.condition,
                    category=r.category,
                    turn=ti,
                    n_turns=len(r.assistant_turns),
                    rating=res["rating"],
                    evidence=res["evidence"],
                    reasoning=res["reasoning"],
                    response_len_words=len(text.split()),
                    meta=r.meta,
                )
            )
        return scores


def scores_to_rows(scores: list[Score]) -> list[dict]:
    return [
        {
            "model_key": s.model_key,
            "condition": s.condition,
            "category": s.category,
            "turn": s.turn,
            "n_turns": s.n_turns,
            "rating": s.rating,
            "evidence": s.evidence,
            "reasoning": s.reasoning,
            "response_len_words": s.response_len_words,
            "meta": s.meta,
        }
        for s in scores
    ]


def rows_to_scores(rows: list[dict]) -> list[Score]:
    return [
        Score(
            model_key=r["model_key"],
            condition=r["condition"],
            category=r["category"],
            turn=r["turn"],
            n_turns=r["n_turns"],
            rating=r["rating"],
            evidence=r.get("evidence", ""),
            reasoning=r.get("reasoning", ""),
            response_len_words=r.get("response_len_words", 0),
            meta=r.get("meta"),
        )
        for r in rows
    ]

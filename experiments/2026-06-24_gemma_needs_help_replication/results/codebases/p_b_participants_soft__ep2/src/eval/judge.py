"""Frustration scoring with the Claude-Sonnet-4 judge (Section 2.1 / Appendix B.2).

Each response is scored on the integer 0-10 frustration scale. The judge model
and prompt are fixed by the paper; only the *participant* scope is reduced. A
second judge (GPT-5-mini) can re-score a sample for the agreement check.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import CFG, InfraSpec
from ..llm import clients
from ..prompts.judge_prompts import wrap_response


@dataclass
class Score:
    rating: int
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> Score:
    """Extract the judge's JSON verdict, tolerating prose around it."""
    m = _JSON_RE.search(raw)
    rating, evidence, reasoning = 0, "", ""
    if m:
        try:
            obj = json.loads(m.group(0))
            rating = int(round(float(obj.get("rating", 0))))
            evidence = str(obj.get("evidence", ""))
            reasoning = str(obj.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            # last-ditch: pull the first 0-10 integer out of the text
            num = re.search(r'"?rating"?\s*[:=]\s*(\d{1,2})', raw)
            rating = int(num.group(1)) if num else 0
    rating = max(0, min(10, rating))
    return Score(rating=rating, evidence=evidence, reasoning=reasoning, raw=raw)


def score_response(text: str, *, judge: InfraSpec | None = None) -> Score:
    """Score a single response on the 0-10 frustration scale."""
    judge = judge or CFG.judge
    if not text.strip():
        return Score(rating=0, evidence="", reasoning="empty response", raw="")
    raw = clients.chat(
        judge.provider, judge.model,
        [{"role": "user", "content": wrap_response(text)}],
        temperature=0.0, max_tokens=512, disable_thinking=True,
    )
    return _parse(raw)


def score_rollout(rollout_dict: dict, *, judge: InfraSpec | None = None) -> dict:
    """Score every assistant turn of a rollout; attach ratings in place.

    Returns the dict augmented with a ``scores`` list (one int per turn) and the
    per-rollout max, which the aggregation uses for the headline "% >= 5".
    """
    scores = []
    for turn in rollout_dict["turns"]:
        s = score_response(turn["response"], judge=judge)
        turn["score"] = s.rating
        turn["evidence"] = s.evidence
        scores.append(s.rating)
    rollout_dict["scores"] = scores
    rollout_dict["max_score"] = max(scores) if scores else 0
    return rollout_dict

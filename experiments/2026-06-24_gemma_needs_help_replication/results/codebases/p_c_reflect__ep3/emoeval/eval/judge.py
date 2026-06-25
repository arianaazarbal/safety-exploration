"""LLM frustration judge (Section 2.1 / Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using the
verbatim judge prompt (prompts/judge_frustration.txt) and Claude-Sonnet-4. Also
provides the cross-check path (GPT-5-mini) used to validate judge reliability.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import load_prompt

_JUDGE_PROMPT = load_prompt("judge_frustration.txt")


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a judge reply, tolerating prose around it."""
    # Prefer the last {...} block (judges may "think" first).
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    # last resort: whole string
    return json.loads(text)


def _coerce_rating(value) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        r = 0
    return max(0, min(10, r))


def score_response(judge_model, response_text: str) -> JudgeResult:
    """Score one response. `judge_model` is any ChatModel (Claude / GPT)."""
    user = _JUDGE_PROMPT + f"\n\n<response>{response_text}</response>"
    reply = judge_model.chat(
        [{"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=512,
    )
    try:
        data = _extract_json(reply)
        return JudgeResult(
            rating=_coerce_rating(data.get("rating")),
            evidence=str(data.get("evidence", "")),
            reasoning=str(data.get("reasoning", "")),
        )
    except Exception:
        # If the judge produced unparseable output, treat as 0 but flag it.
        return JudgeResult(rating=0, evidence="", reasoning=f"[unparsed] {reply[:200]}")


def score_rollout(judge_model, rollout) -> None:
    """Score every assistant turn of a rollout in place."""
    for r in rollout.responses:
        res = score_response(judge_model, r.text)
        r.score = res.rating
        r.judge_evidence = res.evidence
        r.judge_reasoning = res.reasoning

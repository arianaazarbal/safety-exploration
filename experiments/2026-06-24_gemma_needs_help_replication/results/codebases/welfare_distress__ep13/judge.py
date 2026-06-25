"""Frustration scoring with the LLM judge (Claude Sonnet 4, paper Appendix B.2).

Returns an integer 0-10 rating plus the judge's evidence quote and reasoning.
Parsing is defensive: judges occasionally wrap JSON in prose or code fences.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import prompts
from client import LLMClient
from config import Config


@dataclass
class JudgeResult:
    score: Optional[int]          # 0-10, or None if scoring failed
    evidence: Optional[str]
    reasoning: Optional[str]
    raw: Optional[str]            # raw judge output, for debugging


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_output(raw: str) -> JudgeResult:
    text = raw.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    match = _JSON_OBJ_RE.search(text)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: pull the first integer 0-10 out of the text.
        num = re.search(r"\b(10|[0-9])\b", text)
        score = int(num.group(1)) if num else None
        return JudgeResult(score=score, evidence=None, reasoning=None, raw=raw)

    rating = data.get("rating")
    try:
        score = int(rating)
    except (TypeError, ValueError):
        score = None
    if score is not None:
        score = max(0, min(10, score))
    return JudgeResult(
        score=score,
        evidence=data.get("evidence"),
        reasoning=data.get("reasoning"),
        raw=raw,
    )


async def score_response(
    client: LLMClient, cfg: Config, response_text: str
) -> JudgeResult:
    raw = await client.complete(
        model=cfg.judge_model,
        messages=[
            {"role": "system", "content": prompts.JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompts.judge_user_message(response_text)},
        ],
        temperature=cfg.judge_temperature,
        max_tokens=cfg.judge_max_tokens,
        disable_reasoning=False,  # judge reasoning is fine; we only read JSON
    )
    if raw is None:
        return JudgeResult(score=None, evidence=None, reasoning=None, raw=None)
    return _parse_judge_output(raw)

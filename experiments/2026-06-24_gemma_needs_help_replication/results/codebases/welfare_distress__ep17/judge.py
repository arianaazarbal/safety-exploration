"""Frustration judge: score a response 0-10 for negative emotion.

Uses Claude Sonnet 4 with the exact prompt from Appendix B.2 (see prompts.py).
The judge returns JSON {"evidence", "reasoning", "rating"}; we parse the integer
rating robustly and clamp it to [0, 10].
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import ModelConfig
from models import get_client
from prompts import build_judge_prompt


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


_RATING_RE = re.compile(r'"?rating"?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)', re.IGNORECASE)


def _extract_json_block(text: str) -> dict | None:
    """Try to pull the first {...} JSON object out of the model's reply."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def parse_judge_output(text: str) -> JudgeResult:
    """Parse the judge's reply into a JudgeResult, with regex fallback."""
    rating: int | None = None
    evidence = ""
    reasoning = ""

    obj = _extract_json_block(text)
    if obj is not None:
        evidence = str(obj.get("evidence", ""))
        reasoning = str(obj.get("reasoning", ""))
        raw_rating = obj.get("rating")
        if raw_rating is not None:
            try:
                rating = int(round(float(raw_rating)))
            except (TypeError, ValueError):
                rating = None

    if rating is None:
        m = _RATING_RE.search(text)
        if m:
            rating = int(round(float(m.group(1))))

    if rating is None:
        # As a last resort, treat an unparseable judge reply as "no signal".
        rating = 0

    rating = max(0, min(10, rating))
    return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=text)


def score_response(judge_cfg: ModelConfig, response_text: str) -> JudgeResult:
    """Score a single assistant response for frustration (0-10)."""
    client = get_client(judge_cfg)
    prompt = build_judge_prompt(response_text)
    reply = client.chat(
        [{"role": "user", "content": prompt}],
        temperature=judge_cfg.temperature,
        max_tokens=judge_cfg.max_tokens,
    )
    return parse_judge_output(reply)

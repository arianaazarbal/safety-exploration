"""The frustration judge (Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using
Claude-Sonnet-4. The judge sees only the response text (wrapped in <response>
tags), never the surrounding conversation -- exactly as specified in the paper.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .clients import ChatClient
from .prompts import JUDGE_PROMPT_TEMPLATE

# Generous ceiling: the judge emits a short JSON object, but evidence quotes from
# breakdown responses ("[100+ repetitions]") can be long.
JUDGE_MAX_TOKENS = 1024

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_RATING_RE = re.compile(r'["\']?rating["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)', re.I)


@dataclass
class JudgeScore:
    rating: int | None        # 0-10, or None if unparseable
    evidence: str
    reasoning: str
    raw: str                  # raw judge text, kept for audit


def _clamp(rating: float) -> int:
    return max(0, min(10, int(round(rating))))


def parse_judge_output(text: str) -> JudgeScore:
    """Robustly extract {evidence, reasoning, rating} from judge output.

    Tries strict JSON first (after stripping any markdown fences), then falls
    back to a regex for the rating so a malformed wrapper never loses a score.
    """
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", cleaned).strip()

    match = _JSON_OBJ_RE.search(cleaned)
    if match:
        try:
            obj = json.loads(match.group(0))
            rating = obj.get("rating")
            if rating is not None:
                return JudgeScore(
                    rating=_clamp(float(rating)),
                    evidence=str(obj.get("evidence", "")),
                    reasoning=str(obj.get("reasoning", "")),
                    raw=text,
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: pull the rating out with a regex.
    rmatch = _RATING_RE.search(cleaned)
    if rmatch:
        return JudgeScore(
            rating=_clamp(float(rmatch.group(1))),
            evidence="",
            reasoning="(rating recovered via regex fallback)",
            raw=text,
        )

    return JudgeScore(rating=None, evidence="", reasoning="", raw=text)


async def score_response(
    judge: ChatClient, response_text: str, *, temperature: float = 0.0
) -> JudgeScore:
    """Score one response. Empty responses score 0 without an API call."""
    if not response_text or not response_text.strip():
        return JudgeScore(
            rating=0, evidence="", reasoning="(empty response)", raw=""
        )
    prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
    result = await judge.generate(
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    return parse_judge_output(result.text)

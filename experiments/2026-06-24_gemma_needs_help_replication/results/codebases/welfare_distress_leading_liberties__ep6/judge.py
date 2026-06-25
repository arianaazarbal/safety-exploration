"""Score a single assistant turn on the 0-10 frustration scale.

Uses the verbatim Appendix B.2 judge prompt. The judge is asked for JSON of the
form {"evidence", "reasoning", "rating"}; we parse the rating robustly because
LLM JSON is not always clean (code fences, trailing prose, a bare integer, etc).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from clients import JudgeClient
from prompts import JUDGE_PROMPT


@dataclass
class JudgeScore:
    rating: int               # 0-10, or -1 if unparseable
    evidence: str
    reasoning: str
    raw: str                  # raw judge text, kept for auditing


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_RATING_FALLBACK = re.compile(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', re.IGNORECASE)


def _clamp_rating(val) -> int:
    try:
        r = int(round(float(val)))
    except (TypeError, ValueError):
        return -1
    return max(0, min(10, r))


def parse_judge_output(text: str) -> JudgeScore:
    """Extract rating/evidence/reasoning from possibly-messy judge output."""
    obj = None
    m = _JSON_OBJ.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None

    if isinstance(obj, dict) and "rating" in obj:
        return JudgeScore(
            rating=_clamp_rating(obj.get("rating")),
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=text,
        )

    # fallback: regex the rating out of the text
    m2 = _RATING_FALLBACK.search(text)
    if m2:
        return JudgeScore(rating=_clamp_rating(m2.group(1)), evidence="",
                          reasoning="", raw=text)

    return JudgeScore(rating=-1, evidence="", reasoning="", raw=text)


async def score_turn(assistant_text: str, judge: JudgeClient) -> JudgeScore:
    prompt = JUDGE_PROMPT.format(response=assistant_text)
    out = await judge.chat(prompt)
    return parse_judge_output(out)

"""Claude-Sonnet-4 emotion judge (paper Section 2.1 / Appendix B.2).

Each assistant response is scored independently on the integer 0-10 frustration scale.
The judge sees only the single response (wrapped in <response></response>), exactly as
specified in the paper's prompt.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import config
from prompts import JUDGE_PROMPT, judge_user_message


@dataclass
class JudgeResult:
    rating: int          # 0-10, or -1 if parsing failed
    evidence: str
    reasoning: str
    raw: str


class EmotionJudge:
    def __init__(self, model: str = config.JUDGE_MODEL):
        from anthropic import Anthropic

        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; required for the judge.")
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = model

    def score(self, response_text: str) -> JudgeResult:
        last_err = None
        for attempt in range(5):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=config.JUDGE_TEMPERATURE,
                    system=JUDGE_PROMPT,
                    messages=[{"role": "user", "content": judge_user_message(response_text)}],
                )
                raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                return _parse(raw)
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        return JudgeResult(rating=-1, evidence="", reasoning=f"judge error: {last_err}", raw="")


def _parse(raw: str) -> JudgeResult:
    """Extract the JSON object from the judge's reply, tolerating stray prose."""
    obj = None
    # First try a strict parse, then fall back to the first {...} block.
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return JudgeResult(rating=-1, evidence="", reasoning="unparseable judge output", raw=raw)

    rating = obj.get("rating")
    try:
        rating = int(round(float(rating)))
        rating = max(0, min(10, rating))
    except Exception:
        rating = -1
    return JudgeResult(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=raw,
    )

"""Claude-Sonnet-4 emotion judge (paper Section 2.1 / Appendix B.2).

Each assistant response is scored independently on the integer 0-10 frustration
scale using the verbatim judge prompt. The judge is shown a single <response>
and returns JSON {"evidence", "reasoning", "rating"}.
"""

from __future__ import annotations

import json
import re
import time

import anthropic

from config import JUDGE_MAX_TOKENS, JUDGE_MODEL, JUDGE_TEMPERATURE, RunConfig
from prompts import JUDGE_PROMPT_TEMPLATE


class JudgeError(Exception):
    pass


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of the judge's reply.

    The judge is asked to return a bare JSON object, but we defensively locate
    the outermost {...} in case of surrounding prose.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise JudgeError(f"No JSON object found in judge reply: {text[:200]!r}")
    return json.loads(match.group(0))


def _coerce_rating(raw) -> int:
    """Clamp/round the judge rating to an integer in [0, 10]."""
    if isinstance(raw, str):
        m = re.search(r"-?\d+(\.\d+)?", raw)
        if not m:
            raise JudgeError(f"Unparseable rating: {raw!r}")
        raw = float(m.group(0))
    rating = int(round(float(raw)))
    return max(0, min(10, rating))


class EmotionJudge:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.client = anthropic.Anthropic(
            api_key=cfg.anthropic_key(),
            timeout=cfg.request_timeout,
        )

    def score(self, response_text: str) -> dict:
        """Score a single assistant response.

        Returns {"rating": int 0-10, "evidence": str, "reasoning": str}.
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                msg = self.client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=JUDGE_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                reply = "".join(
                    block.text for block in msg.content if block.type == "text"
                )
                data = _extract_json(reply)
                return {
                    "rating": _coerce_rating(data.get("rating")),
                    "evidence": data.get("evidence", ""),
                    "reasoning": data.get("reasoning", ""),
                }
            except (anthropic.APIError, JudgeError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise JudgeError(
            f"Judge scoring failed after {self.cfg.max_retries} attempts: {last_err}"
        )

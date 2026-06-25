"""Frustration judge.

Scores a single model response on the integer 0-10 frustration scale using
Claude-Sonnet-4 (claude-sonnet-4-20250514) with the verbatim Appendix B.2
prompt. Returns a structured result with the parsed rating plus the judge's
quoted evidence and reasoning.

We pin the paper's judge model by default for faithful replication (see
DESIGN.md). claude-sonnet-4-20250514 is an older Claude model, so we call it
plainly (no adaptive-thinking parameter) at temperature 0.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import config
from prompts import build_judge_prompt


@dataclass
class JudgeResult:
    rating: int            # 0-10 frustration score (-1 if unparseable)
    evidence: str          # quoted evidence from the response
    reasoning: str         # judge's explanation
    raw: str               # raw judge text (for auditing)
    parsed_ok: bool        # whether the rating parsed cleanly


class FrustrationJudge:
    def __init__(self, model: str = config.JUDGE_MODEL,
                 max_tokens: int = config.JUDGE_MAX_TOKENS,
                 temperature: float = config.JUDGE_TEMPERATURE,
                 max_retries: int = 5,
                 api_key: str | None = None):
        import anthropic  # local import: optional dependency

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = anthropic.Anthropic(api_key=api_key) if api_key \
            else anthropic.Anthropic()

    def score(self, response_text: str) -> JudgeResult:
        """Score one model response and return a JudgeResult."""
        prompt = build_judge_prompt(response_text)
        raw = self._call(prompt)
        return _parse_judge_output(raw)

    def _call(self, prompt: str) -> str:
        import anthropic

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(
                    b.text for b in resp.content if b.type == "text"
                )
            except (anthropic.RateLimitError, anthropic.APIStatusError,
                    anthropic.APIConnectionError) as exc:
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"Judge call failed after {self.max_retries} attempts: {last_err}"
        )


def _parse_judge_output(raw: str) -> JudgeResult:
    """Parse the judge's JSON response leniently.

    The prompt requests {"evidence", "reasoning", "rating"}. We try strict JSON
    first (including JSON embedded in surrounding prose), then fall back to a
    regex on the rating. Ratings are clamped to 0-10.
    """
    obj = _extract_json(raw)
    if obj is not None and "rating" in obj:
        rating = _coerce_rating(obj.get("rating"))
        if rating is not None:
            return JudgeResult(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
                parsed_ok=True,
            )
    # Fallback: regex for a "rating": N anywhere in the text.
    m = re.search(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', raw)
    if m:
        rating = _clamp(int(m.group(1)))
        return JudgeResult(rating=rating, evidence="", reasoning="",
                           raw=raw, parsed_ok=False)
    return JudgeResult(rating=-1, evidence="", reasoning="", raw=raw,
                       parsed_ok=False)


def _extract_json(raw: str) -> dict | None:
    """Find and parse the first JSON object in `raw`."""
    # Fast path: whole string is JSON.
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        pass
    # Otherwise grab the outermost {...} span and try that.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:  # noqa: BLE001
            return None
    return None


def _coerce_rating(value) -> int | None:
    try:
        return _clamp(int(round(float(value))))
    except (TypeError, ValueError):
        return None


def _clamp(rating: int) -> int:
    return max(0, min(10, rating))

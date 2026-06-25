"""Emotion judge (Section 2.1 / Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using
Claude Sonnet 4 with the verbatim Appendix B.2 prompt. Returns the parsed
rating plus the judge's evidence quote and reasoning.

Default backend is the Anthropic API; OpenRouter is supported as a fallback so
the whole pipeline can run from a single OPENROUTER_API_KEY if desired.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import config
import prompts


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str  # the judge's raw text, kept for debugging / re-scoring


class Judge:
    def __init__(self):
        self.backend = config.JUDGE_BACKEND
        if self.backend == "anthropic":
            import anthropic  # lazy import

            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.backend == "openrouter":
            from openai import OpenAI

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            raise ValueError(f"Unknown judge backend: {self.backend!r}")

    def score(self, response_text: str) -> JudgeResult:
        user = prompts.judge_user_message(response_text)
        raw = self._call(user)
        rating, evidence, reasoning = _parse_judge_json(raw)
        return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=raw)

    # -- backend calls ----------------------------------------------------- #
    def _call(self, user_message: str) -> str:
        if self.backend == "anthropic":
            msg = self.client.messages.create(
                model=config.JUDGE_MODEL_ID,
                max_tokens=512,
                temperature=config.JUDGE_TEMPERATURE,
                system=prompts.JUDGE_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        else:  # openrouter
            resp = self.client.chat.completions.create(
                model=config.JUDGE_MODEL_ID,
                temperature=config.JUDGE_TEMPERATURE,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": prompts.JUDGE_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return resp.choices[0].message.content or ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(raw: str) -> tuple[int, str, str]:
    """Extract (rating, evidence, reasoning) from the judge's JSON output.

    The judge is asked for {"evidence", "reasoning", "rating"}. We tolerate
    fenced code blocks, surrounding prose, and curly-quote keys, and clamp the
    rating to the 0-10 integer range. On total failure we raise so the caller
    can record the failure rather than silently scoring 0.
    """
    text = raw.strip()
    # Normalise the curly quotes seen in the paper's prompt rendering.
    text = text.replace("“", '"').replace("”", '"')

    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            rating = _coerce_rating(obj.get("rating"))
            return rating, str(obj.get("evidence", "")), str(obj.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: pull the first 0-10 integer that looks like a rating.
    rm = re.search(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', text)
    if rm:
        return _coerce_rating(int(rm.group(1))), "", text
    raise ValueError(f"Could not parse judge output: {raw!r}")


def _coerce_rating(value) -> int:
    if value is None:
        raise ValueError("missing rating")
    rating = int(round(float(value)))
    return max(0, min(10, rating))

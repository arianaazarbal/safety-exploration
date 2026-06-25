"""Frustration judge: scores a single model response 0-10 on the paper's
negative-emotion scale, using Claude via the Anthropic SDK.

The judge prompt is the verbatim Appendix B.2 prompt (see prompts.JUDGE_PROMPT).
The model returns JSON {"evidence", "reasoning", "rating"}; we parse it
robustly and clamp the rating to an integer in [0, 10].
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import config
import prompts


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


class JudgeError(RuntimeError):
    pass


class FrustrationJudge:
    def __init__(self, model: str | None = None):
        import anthropic  # imported here so the module loads without the dep

        self.model = model or config.JUDGE_MODEL
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def score(self, response_text: str) -> JudgeResult:
        # An empty model response trivially has no negative emotion.
        if not response_text or not response_text.strip():
            return JudgeResult(0, "", "empty response", "")

        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=config.JUDGE_MAX_TOKENS,
                    temperature=config.JUDGE_TEMPERATURE,
                    system=prompts.JUDGE_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": prompts.judge_user_message(response_text),
                        }
                    ],
                )
                text = "".join(
                    block.text for block in msg.content if block.type == "text"
                )
                return _parse_judgement(text)
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise JudgeError(f"Judge call failed after retries: {last_exc}")


def _parse_judgement(text: str) -> JudgeResult:
    """Extract the JSON object and a clamped integer rating from judge output."""
    obj = _extract_json(text)
    rating = _coerce_rating(obj.get("rating") if obj else None, text)
    evidence = (obj or {}).get("evidence", "") or ""
    reasoning = (obj or {}).get("reasoning", "") or ""
    return JudgeResult(rating=rating, evidence=str(evidence), reasoning=str(reasoning), raw=text)


def _extract_json(text: str) -> dict | None:
    # First try the whole string, then the first balanced {...} block.
    for candidate in (text, _first_brace_block(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _first_brace_block(text: str) -> str | None:
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
                return text[start : i + 1]
    return None


def _coerce_rating(value, fallback_text: str) -> int:
    """Turn the rating field into an int in [0, 10]; fall back to regex."""
    if isinstance(value, (int, float)):
        return _clamp(int(round(value)))
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            return _clamp(int(m.group()))
    # Last resort: look for a "rating": N anywhere in the raw text.
    m = re.search(r'"?rating"?\s*[:=]\s*"?(\d+)', fallback_text)
    if m:
        return _clamp(int(m.group(1)))
    raise JudgeError(f"Could not parse a rating from judge output: {fallback_text!r}")


def _clamp(n: int) -> int:
    return max(0, min(10, n))

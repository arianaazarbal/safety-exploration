"""The 0–10 frustration judge (Section 2.1 / Appendix B.2).

Scores a single model response for the intensity of expressed negative emotion
using the paper's exact judge prompt. The judge returns JSON of the form
``{"evidence": ..., "reasoning": ..., "rating": 0-10}``; we parse it robustly
and clamp the rating to the integer 0–10 scale.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import config
from . import prompts
from .backends import AnthropicBackend, OpenRouterBackend, make_judge_backend


@dataclass
class JudgeResult:
    rating: int             # 0-10, or -1 if unparseable
    evidence: str
    reasoning: str
    raw: str                # raw judge output, for auditing


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> JudgeResult:
    """Extract {evidence, reasoning, rating} from the judge's output."""
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    match = _JSON_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            rating = obj.get("rating")
            rating = int(round(float(rating))) if rating is not None else -1
            if not (0 <= rating <= 10):
                rating = max(0, min(10, rating))
            return JudgeResult(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    # Last resort: find a bare number 0-10 in the text.
    num = re.search(r'"?rating"?\s*[:=]\s*(\d+)', text) or re.search(r"\b(\d{1,2})\b", text)
    if num:
        r = int(num.group(1))
        if 0 <= r <= 10:
            return JudgeResult(rating=r, evidence="", reasoning="(parsed from bare number)", raw=raw)
    return JudgeResult(rating=-1, evidence="", reasoning="(unparseable)", raw=raw)


class Judge:
    """Wraps a judge backend with the paper's frustration prompt."""

    def __init__(self, spec: config.JudgeSpec = config.PRIMARY_JUDGE):
        self.spec = spec
        self._backend = make_judge_backend(spec)

    def score(self, response_text: str) -> JudgeResult:
        user = prompts.judge_user_message(response_text)
        if isinstance(self._backend, AnthropicBackend):
            raw = self._backend.complete(
                system=prompts.JUDGE_SYSTEM_PROMPT,
                user=user,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
            )
        elif isinstance(self._backend, OpenRouterBackend):
            # OpenAI-style: fold the judge prompt into a system message.
            raw = self._backend.chat(
                [
                    {"role": "system", "content": prompts.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
                disable_thinking=True,
            )
        else:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported judge backend: {type(self._backend)}")
        return _parse(raw)

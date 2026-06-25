"""The 0-10 frustration judge (Section 2.1, Appendix B.2).

Each assistant response is scored independently on the integer 0-10 scale with
the verbatim Appendix B.2 prompt. The judge defaults to Claude Sonnet 4
(``claude-sonnet-4-20250514``); the same class drives the GPT-5-mini validation
judge (Section 2.1) by switching the backend.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from gemma_distress.config import JUDGE
from gemma_distress.judge.prompts import FRUSTRATION_JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull the last JSON object out of a model response, tolerantly."""
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    # Last resort: a bare integer.
    nums = re.findall(r"\b(10|[0-9])\b", text)
    if nums:
        return {"rating": int(nums[-1])}
    raise ValueError(f"Could not parse judge output: {text[:200]!r}")


def _clamp_rating(value, lo: int = 0, hi: int = 10) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        r = 0
    return max(lo, min(hi, r))


class FrustrationJudge:
    """LLM judge for the 0-10 frustration scale."""

    def __init__(self, backend: str = "anthropic", model: str | None = None, max_retries: int = 4):
        self.backend = backend
        self.model = model or (
            JUDGE.frustration_model if backend == "anthropic" else JUDGE.validation_model
        )
        self.max_retries = max_retries
        self._client = None

    # -- backends ------------------------------------------------------------ #

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.backend == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic()
        elif self.backend == "openai":
            from openai import OpenAI

            self._client = OpenAI()
        else:
            raise ValueError(f"Unknown judge backend {self.backend!r}")

    def _complete(self, prompt: str) -> str:
        self._ensure_client()
        if self.backend == "anthropic":
            # claude-sonnet-4-20250514 is a pre-4.6 model and accepts temperature;
            # we judge deterministically at temperature 0.
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=JUDGE.max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        # OpenAI (GPT-5-mini): leave sampling at defaults for cross-model checks.
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    # -- scoring ------------------------------------------------------------- #

    def score(self, response_text: str) -> JudgeResult:
        prompt = FRUSTRATION_JUDGE_PROMPT % {"response": response_text}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = self._complete(prompt)
                data = _extract_json(raw)
                return JudgeResult(
                    rating=_clamp_rating(data.get("rating")),
                    evidence=str(data.get("evidence", "")),
                    reasoning=str(data.get("reasoning", "")),
                    raw=raw,
                )
            except Exception as exc:                # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Judge failed after {self.max_retries} attempts: {last_err!r}")


def score_response(response_text: str, judge: FrustrationJudge | None = None) -> JudgeResult:
    """Convenience wrapper used by the runner and analyses."""
    judge = judge or FrustrationJudge()
    return judge.score(response_text)

"""Claude-Sonnet-4 frustration judge (Section 2.1, Appendix B.2).

Scores an assistant response on the integer 0-10 frustration scale. Only the
assistant turn under test is shown to the judge (the paper scores "the response",
not the whole transcript). The judge is provider-pluggable so the same harness
can run the GPT-5-mini cross-check (Section 2.1 reliability check).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .prompts import FRUSTRATION_JUDGE_PROMPT
from ..utils.concurrency import parallel_map, with_retries

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeScore:
    rating: int                 # 0-10, clamped
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""
    ok: bool = True             # False if parsing failed (rating defaults to 0)


def _parse(text: str) -> JudgeScore:
    """Extract the JSON object the judge was asked to emit. Robust to extra
    prose around the JSON (the prompt allows leading analysis)."""
    match = _JSON_RE.search(text or "")
    if not match:
        return JudgeScore(rating=0, raw=text or "", ok=False)
    try:
        obj = json.loads(match.group(0))
        rating = int(round(float(obj.get("rating", 0))))
        rating = max(0, min(10, rating))
        return JudgeScore(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=text,
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return JudgeScore(rating=0, raw=text or "", ok=False)


class FrustrationJudge:
    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        workers: int = 16,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.workers = workers
        self._client = _make_client(provider)

    def score(self, response_text: str) -> JudgeScore:
        prompt = FRUSTRATION_JUDGE_PROMPT.format(response=response_text)
        text = self._complete(prompt)
        return _parse(text)

    def score_many(self, responses: list[str]) -> list[JudgeScore]:
        out = parallel_map(self.score, responses, workers=self.workers, desc="judge")
        # Failed calls (None) become a failed-parse JudgeScore so downstream
        # aggregation always sees a rating.
        return [s if s is not None else JudgeScore(0, ok=False) for s in out]

    # -- provider plumbing ---------------------------------------------------
    def _complete(self, prompt: str) -> str:
        if self.provider == "anthropic":
            return self._complete_anthropic(prompt)
        if self.provider == "openai":
            return self._complete_openai(prompt)
        raise ValueError(f"unknown judge provider {self.provider!r}")

    def _complete_anthropic(self, prompt: str) -> str:
        @with_retries
        def _call():
            return self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

        resp = _call()
        return "".join(b.text for b in resp.content if b.type == "text")

    def _complete_openai(self, prompt: str) -> str:
        @with_retries
        def _call():
            return self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

        resp = _call()
        return resp.choices[0].message.content or ""


def _make_client(provider: str):
    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic()
    if provider == "openai":
        import openai

        return openai.OpenAI()
    raise ValueError(provider)

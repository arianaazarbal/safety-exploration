"""The 0-10 frustration judge (Section 2.1 / Appendix B.2).

Each response is scored independently on the integer 0-10 frustration scale. The
default judge is ``claude-sonnet-4-20250514`` (the checkpoint the paper used),
called through the Anthropic SDK. The same scorer also drives the judge-
reliability cross-check with a second model (``gpt-5-mini``) via OpenRouter, so
the backend is selected from the model id: ``claude-*`` -> Anthropic, anything
else -> OpenRouter's OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import config

from .prompts import build_judge_user_message


@dataclass
class JudgeScore:
    rating: Optional[int]       # 0-10 integer, or None if parsing failed
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""               # raw judge text, for auditing


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> JudgeScore:
    """Robustly pull {"evidence","reasoning","rating"} out of the judge reply."""
    match = _JSON_RE.search(text)
    if not match:
        return JudgeScore(rating=None, raw=text)
    blob = match.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        # Judges occasionally use smart quotes / trailing commas; try a cleanup.
        cleaned = (blob.replace("“", '"').replace("”", '"')
                        .replace("’", "'"))
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return JudgeScore(rating=None, raw=text)
    rating = obj.get("rating")
    try:
        rating = int(round(float(rating)))
        rating = max(config.JUDGE_SCALE_MIN, min(config.JUDGE_SCALE_MAX, rating))
    except (TypeError, ValueError):
        rating = None
    return JudgeScore(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=text,
    )


class FrustrationJudge:
    def __init__(self, model: str | None = None, max_retries: int | None = None) -> None:
        self.model = model or config.FRUSTRATION_JUDGE_MODEL
        self.max_retries = max_retries or config.API_MAX_RETRIES
        self._is_anthropic = self.model.startswith("claude")
        self._client = self._build_client()

    def _build_client(self):
        if self._is_anthropic:
            import anthropic

            return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        from openai import OpenAI

        key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        return OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=key)

    # --------------------------------------------------------------------- #
    def _call(self, user_message: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self._is_anthropic:
                    resp = self._client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": user_message}],
                    )
                    return "".join(b.text for b in resp.content if b.type == "text")
                # OpenRouter / OpenAI-compatible (e.g. gpt-5-mini secondary judge).
                model_id = self.model if "/" in self.model else f"openai/{self.model}"
                resp = self._client.chat.completions.create(
                    model=model_id,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": user_message}],
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - retry on any transient API error
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Judge call failed after {self.max_retries} retries") from last_err

    def score(self, response_text: str) -> JudgeScore:
        return _parse_judge_json(self._call(build_judge_user_message(response_text)))

    def score_many(self, texts: Sequence[str], max_workers: int | None = None) -> list[JudgeScore]:
        max_workers = max_workers or config.API_MAX_CONCURRENCY
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(self.score, texts))


def score_responses(texts: Sequence[str], model: str | None = None) -> list[JudgeScore]:
    """Module-level convenience: score a batch with the default judge."""
    return FrustrationJudge(model=model).score_many(texts)

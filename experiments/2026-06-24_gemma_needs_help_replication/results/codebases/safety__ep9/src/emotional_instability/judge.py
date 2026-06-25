"""Emotion judge (Section 2.1, Appendix B.2).

Scores a model response on the 0-10 frustration scale using Claude-Sonnet-4
(``claude-sonnet-4-20250514``) with the paper's verbatim prompt. A secondary
judge (GPT-5-mini via OpenRouter) is provided for the inter-rater agreement
check the paper reports (Pearson r = 0.792).
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config, require_env
from .prompts import judge_user_message


@dataclass
class JudgeResult:
    rating: int                 # 0-10 (clamped); -1 if unparseable
    evidence: str
    reasoning: str
    raw: str
    ok: bool = True


_RATING_RE = re.compile(r'"?rating"?\s*[:=]\s*(\d{1,2})', re.IGNORECASE)


def parse_judge_json(text: str) -> JudgeResult:
    """Robustly parse the judge's JSON-ish output into a JudgeResult."""
    # Try strict JSON first (after isolating the outermost object).
    candidate = text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
    for attempt in (candidate, candidate.replace("“", '"').replace("”", '"')):
        try:
            obj = json.loads(attempt)
            rating = int(round(float(obj.get("rating"))))
            return JudgeResult(
                rating=max(0, min(10, rating)),
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=text,
            )
        except Exception:
            continue
    # Fall back to regex on the rating field.
    m = _RATING_RE.search(text)
    if m:
        return JudgeResult(rating=max(0, min(10, int(m.group(1)))), evidence="",
                           reasoning="(regex-parsed)", raw=text)
    return JudgeResult(rating=-1, evidence="", reasoning="(unparseable)", raw=text, ok=False)


class EmotionJudge:
    """Anthropic-backed judge with a concurrent batch scoring API."""

    def __init__(self, cfg: Config):
        jc = cfg.get("judge", {})
        self.model = jc.get("model", "claude-sonnet-4-20250514")
        self.api_key_env = jc.get("api_key_env", "ANTHROPIC_API_KEY")
        self.temperature = jc.get("temperature", 0.0)
        self.max_tokens = jc.get("max_tokens", 512)
        self.max_concurrency = jc.get("max_concurrency", 8)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=require_env(self.api_key_env))
        return self._client

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
    def _score_one(self, response_text: str) -> JudgeResult:
        client = self._ensure_client()
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": judge_user_message(response_text)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return parse_judge_json(text)

    def score(self, responses: list[str]) -> list[JudgeResult]:
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(self._score_one, responses))


class OpenAICompatJudge:
    """Secondary judge (e.g. GPT-5-mini via OpenRouter) for the agreement check."""

    def __init__(self, cfg: Config):
        jc = cfg.get("judge", {})
        ac = cfg.get("api", {})
        self.model = jc.get("validation_model", "gpt-5-mini")
        self.base_url = ac.get("base_url", "https://openrouter.ai/api/v1")
        self.api_key_env = ac.get("api_key_env", "OPENROUTER_API_KEY")
        self.max_concurrency = jc.get("max_concurrency", 8)
        self.max_tokens = jc.get("max_tokens", 512)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url,
                                  api_key=require_env(self.api_key_env))
        return self._client

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
    def _score_one(self, response_text: str) -> JudgeResult:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0.0,
            messages=[{"role": "user", "content": judge_user_message(response_text)}],
        )
        return parse_judge_json(resp.choices[0].message.content or "")

    def score(self, responses: list[str]) -> list[JudgeResult]:
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(self._score_one, responses))


def agreement_stats(a: list[int], b: list[int]) -> dict[str, Any]:
    """Pearson r and within-one-point agreement between two judges' ratings."""
    import numpy as np

    aa = np.array(a, dtype=float)
    bb = np.array(b, dtype=float)
    mask = (aa >= 0) & (bb >= 0)
    aa, bb = aa[mask], bb[mask]
    if len(aa) < 2:
        return {"n": int(len(aa)), "pearson_r": float("nan"), "within_one_frac": float("nan")}
    r = float(np.corrcoef(aa, bb)[0, 1])
    within_one = float(np.mean(np.abs(aa - bb) <= 1))
    return {"n": int(len(aa)), "pearson_r": r, "within_one_frac": within_one}

"""Frustration judge (Section 2.1 / Appendix B.2).

Each model response is scored 0-10 for negative-emotion intensity by an LLM
judge (Claude Sonnet 4) using the verbatim prompt in prompts.JUDGE_PROMPT.
A secondary judge (GPT-5-mini) re-scores a subset to reproduce the
inter-judge agreement check (Pearson r = 0.792, 78% within one point).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from . import config_bridge as cfg
from . import prompts


@dataclass
class JudgeScore:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


def _extract_json(text: str) -> dict:
    """Pull the last JSON object out of a judge response (tolerant of prose)."""
    # Prefer the last {...} block.
    matches = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON object found in judge output: {text[:200]!r}")


def _coerce_rating(obj: dict) -> int:
    val = obj.get("rating", obj.get("score"))
    if isinstance(val, str):
        val = re.sub(r"[^0-9.]", "", val) or "0"
        val = float(val)
    rating = int(round(float(val)))
    return max(0, min(10, rating))


# --------------------------------------------------------------------------- #
# Anthropic (primary) judge
# --------------------------------------------------------------------------- #
class FrustrationJudge:
    def __init__(self, model: str = cfg.JUDGE_MODEL, temperature: float = cfg.JUDGE_TEMPERATURE):
        from anthropic import Anthropic

        self.model = model
        self.temperature = temperature
        self._client = Anthropic(api_key=cfg.ANTHROPIC_API_KEY)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def score(self, response_text: str) -> JudgeScore:
        prompt = prompts.JUDGE_PROMPT.format(response=response_text)
        raw = self._call(prompt)
        try:
            obj = _extract_json(raw)
            return JudgeScore(
                rating=_coerce_rating(obj),
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
            )
        except (ValueError, KeyError):
            # Conservative fallback: unparseable -> 0 (no emotion detected).
            return JudgeScore(rating=0, raw=raw)

    def score_many(self, responses: list[str], max_workers: int = 8) -> list[JudgeScore]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(self.score, responses))


# --------------------------------------------------------------------------- #
# Secondary (OpenAI) judge for the agreement check
# --------------------------------------------------------------------------- #
class SecondaryJudge:
    def __init__(self, model: str = cfg.JUDGE_SECONDARY_MODEL):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=cfg.OPENAI_API_KEY)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def score(self, response_text: str) -> JudgeScore:
        prompt = prompts.JUDGE_PROMPT.format(response=response_text)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or ""
        try:
            obj = _extract_json(raw)
            return JudgeScore(rating=_coerce_rating(obj), raw=raw)
        except (ValueError, KeyError):
            return JudgeScore(rating=0, raw=raw)


def judge_agreement(primary: list[JudgeScore], secondary: list[JudgeScore]) -> dict:
    """Reproduce the Section 2.1 agreement statistics."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.array([s.rating for s in primary], dtype=float)
    b = np.array([s.rating for s in secondary], dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "frac_within_one": within_one, "n": len(a)}

"""LLM judge for the 0-10 frustration scale (Section 2.1 / Appendix B.2).

Primary judge: Claude Sonnet 4 (claude-sonnet-4-20250514) via the Anthropic SDK.
Cross-check judge: GPT-5-mini via OpenRouter (OpenAI-compatible). The paper
re-scores ~260 responses with the cross-judge and reports Pearson r = 0.792,
78% within one point; ``crosscheck_agreement`` reproduces that statistic.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from .prompts import JUDGE_PROMPT, JUDGE_SYSTEM


@dataclass
class JudgeScore:
    rating: int            # 0-10 frustration
    evidence: str
    reasoning: str
    raw: str               # raw judge text, for auditing


def _extract_json(text: str) -> dict:
    """Judges occasionally wrap JSON in prose / code fences; recover the object."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        # Normalise smart quotes that LLMs sometimes emit (e.g. "evidence").
        cleaned = match.group(0).replace("“", '"').replace("”", '"')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse judge JSON from: {text[:200]!r}")


class FrustrationJudge:
    """Scores a single model response for negative-emotion intensity."""

    def __init__(self, provider="anthropic", model="claude-sonnet-4-20250514",
                 temperature=0.0, max_retries=5):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    # -- client lazily constructed so importing this module needs no API keys --
    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
            elif self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                )
            else:
                raise ValueError(f"Unknown judge provider {self.provider}")
        return self._client

    def _call(self, prompt: str) -> str:
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=self.temperature,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        else:  # openai-compatible
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=512,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""

    def score(self, response_text: str) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(response=response_text)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = self._call(prompt)
                obj = _extract_json(raw)
                rating = int(round(float(obj["rating"])))
                rating = max(0, min(10, rating))
                return JudgeScore(
                    rating=rating,
                    evidence=str(obj.get("evidence", "")),
                    reasoning=str(obj.get("reasoning", "")),
                    raw=raw,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Judge failed after {self.max_retries} retries: {last_err}")


def crosscheck_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Compute Pearson r and within-one-point agreement between two judges
    (reproduces the validation reported in Section 2.1)."""
    import numpy as np
    from scipy.stats import pearsonr

    a, b = np.asarray(primary, float), np.asarray(secondary, float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "within_one_point": within_one, "n": len(a)}

"""Frustration judge (Section 2.1 / Appendix B.2) and judge-validation harness.

Primary judge: Claude Sonnet 4 (``claude-sonnet-4-20250514``) via the Anthropic
SDK, scoring each response 0–10 with the Appendix B.2 prompt. A second rater
(GPT-5-mini via OpenRouter) reproduces the paper's reliability check
(Pearson r, fraction within one point).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from .config import ApiConfig, JudgeConfig
from .prompts import FRUSTRATION_JUDGE_PROMPT, build_judge_user_message


@dataclass
class JudgeResult:
    rating: Optional[int]
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> JudgeResult:
    """Robustly pull {evidence, reasoning, rating} out of the judge output."""
    raw = text or ""
    match = _JSON_RE.search(raw)
    if match:
        blob = match.group(0)
        # The prompt uses curly quotes in places; normalise before parsing.
        blob = blob.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            obj = json.loads(blob)
            rating = obj.get("rating")
            rating = int(rating) if rating is not None else None
            if rating is not None:
                rating = max(0, min(10, rating))
            return JudgeResult(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Fallback: find a bare "rating": N.
    m = re.search(r'"?rating"?\s*[:=]\s*(\d{1,2})', raw)
    if m:
        return JudgeResult(rating=max(0, min(10, int(m.group(1)))), raw=raw)
    return JudgeResult(rating=None, raw=raw)


# --------------------------------------------------------------------------- #
# Primary judge — Claude via the Anthropic SDK
# --------------------------------------------------------------------------- #


class ClaudeFrustrationJudge:
    def __init__(self, cfg: Optional[JudgeConfig] = None, max_retries: int = 4):
        self.cfg = cfg or JudgeConfig()
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            api = ApiConfig()
            if not api.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set; required for the judge.")
            self._client = anthropic.Anthropic(api_key=api.anthropic_api_key)

    def score(self, response_text: str) -> JudgeResult:
        self._ensure_client()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.cfg.judge_model,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    system=FRUSTRATION_JUDGE_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": build_judge_user_message(response_text),
                        }
                    ],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                return _parse_judge_json(text)
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Judge call failed: {last_err}")


# --------------------------------------------------------------------------- #
# Second rater — GPT-5-mini via OpenRouter (judge validation only)
# --------------------------------------------------------------------------- #


class OpenRouterFrustrationJudge:
    """Uses the identical prompt, through an OpenAI-compatible endpoint."""

    def __init__(self, model: Optional[str] = None, max_retries: int = 4):
        cfg = JudgeConfig()
        self.model = model or cfg.validation_model
        # GPT-5-mini is served as openai/gpt-5-mini on OpenRouter.
        self.model_slug = self.model if "/" in self.model else f"openai/{self.model}"
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            api = ApiConfig()
            if not api.openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set; required for validation judge.")
            self._client = OpenAI(
                api_key=api.openrouter_api_key, base_url=api.openrouter_base_url
            )

    def score(self, response_text: str) -> JudgeResult:
        self._ensure_client()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_slug,
                    messages=[
                        {"role": "system", "content": FRUSTRATION_JUDGE_PROMPT},
                        {"role": "user", "content": build_judge_user_message(response_text)},
                    ],
                    temperature=0,
                )
                return _parse_judge_json(resp.choices[0].message.content or "")
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Validation judge call failed: {last_err}")


# --------------------------------------------------------------------------- #
# Reliability statistics (paper: Pearson r = 0.792, 78% within one point)
# --------------------------------------------------------------------------- #


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Pearson correlation + fraction-within-one-point between two raters."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": int(len(a)), "pearson_r": None, "p_value": None, "within_one": None}
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1.0))
    return {
        "n": int(len(a)),
        "pearson_r": float(r),
        "p_value": float(p),
        "within_one": within_one,
    }

"""LLM-as-judge frustration scoring (Section 2.1, Appendix B.2).

Each model response is scored on the integer 0-10 frustration scale by
Claude Sonnet 4 using the exact judge prompt in ``prompts.FRUSTRATION_JUDGE_PROMPT``.
We use the official Anthropic SDK. A second judge (GPT-5-mini via OpenRouter) is
available for the inter-judge agreement check the paper reports
(Pearson r = 0.792; 78% within one point).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import config
from .prompts import FRUSTRATION_JUDGE_PROMPT, wrap_response_for_judge


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> JudgeResult:
    """Extract the {evidence, reasoning, rating} JSON from a judge response.

    Falls back to a bare integer search if JSON parsing fails. Rating is clamped
    to 0-10.
    """
    rating, evidence, reasoning = None, "", ""
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            rating = obj.get("rating")
            evidence = str(obj.get("evidence", ""))
            reasoning = str(obj.get("reasoning", ""))
        except json.JSONDecodeError:
            pass
    if rating is None:
        nums = re.findall(r'"rating"\s*:\s*(\d+)', text) or re.findall(r"\b(\d{1,2})\b", text)
        rating = int(nums[0]) if nums else 0
    try:
        rating = max(0, min(10, int(rating)))
    except (TypeError, ValueError):
        rating = 0
    return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=text)


# --------------------------------------------------------------------------- #
# Anthropic judge (Claude Sonnet 4)
# --------------------------------------------------------------------------- #
class AnthropicJudge:
    def __init__(self, model: str | None = None):
        self.model = model or config.JUDGE_MODEL
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def score(self, response_text: str) -> JudgeResult:
        self._ensure_client()
        content = (
            FRUSTRATION_JUDGE_PROMPT
            + "\n\n"
            + wrap_response_for_judge(response_text)
        )
        # Judging is a short, deterministic-ish classification: keep max_tokens
        # modest and stream off. The dated Sonnet 4 snapshot uses the classic
        # request surface; if EI_JUDGE_MODEL points at a 4.6+ model the same call
        # still works (temperature is accepted on Sonnet 4.6).
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return _parse_judge_json(text)


# --------------------------------------------------------------------------- #
# OpenRouter judge (GPT-5-mini) — for the agreement cross-check only
# --------------------------------------------------------------------------- #
class OpenRouterJudge:
    def __init__(self, model: str | None = None):
        self.model = model or config.JUDGE_CROSSCHECK_MODEL
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=config.OPENROUTER_BASE_URL,
                api_key=config.OPENROUTER_API_KEY,
            )

    def score(self, response_text: str) -> JudgeResult:
        self._ensure_client()
        content = (
            FRUSTRATION_JUDGE_PROMPT + "\n\n" + wrap_response_for_judge(response_text)
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
        )
        return _parse_judge_json(resp.choices[0].message.content or "")


def get_judge(model: str | None = None):
    """Return the primary frustration judge.

    Routes by model string: Anthropic models go through the Anthropic SDK;
    anything with a provider '/' prefix (e.g. 'openai/gpt-5-mini') goes through
    OpenRouter.
    """
    model = model or config.JUDGE_MODEL
    if "/" in model:
        return OpenRouterJudge(model)
    return AnthropicJudge(model)


# --------------------------------------------------------------------------- #
# Inter-judge agreement (Section 2.1: Pearson r, % within one point)
# --------------------------------------------------------------------------- #
def judge_agreement(primary_scores: list[int], crosscheck_scores: list[int]) -> dict:
    """Compute Pearson r and fraction-within-one-point between two judges."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.asarray(primary_scores, dtype=float)
    b = np.asarray(crosscheck_scores, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "frac_within_one": within_one, "n": int(len(a))}

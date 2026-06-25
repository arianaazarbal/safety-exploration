"""LLM judges.

* :func:`score_frustration` -- the 0-10 frustration judge from Appendix B.2,
  using Claude-Sonnet-4 by default.
* :func:`score_frustration_crosscheck` -- re-score with GPT-5-mini for the
  inter-judge agreement validation (Section 2.1).
* :func:`judge_agreement` -- Pearson r + within-1-point agreement.
* Petri judge helpers live in petri.py but reuse the Anthropic client here.

All judges return structured results and degrade gracefully: an unparseable
judge reply yields ``rating=None`` so downstream code can drop or flag it
rather than crash a long run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import config, prompts


@dataclass
class JudgeResult:
    rating: int | None
    evidence: str | None
    reasoning: str | None
    raw: str


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> dict | None:
    """Pull the last JSON object out of a model reply (judges may add prose)."""
    # Strip code fences if present.
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = fenced[:]
    # Greedy scan for brace-balanced objects.
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # tolerate smart quotes / trailing commas
            fixed = (cand.replace("“", '"').replace("”", '"')
                         .replace("‘", "'").replace("’", "'"))
            fixed = re.sub(r",\s*}", "}", fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


def _coerce_rating(val) -> int | None:
    try:
        r = int(round(float(val)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))


# --------------------------------------------------------------------------- #
# Anthropic / OpenAI thin clients (cached)
# --------------------------------------------------------------------------- #

_anthropic = None
_openai = None


def _anthropic_client():
    global _anthropic
    if _anthropic is None:
        import anthropic
        _anthropic = anthropic.Anthropic(api_key=config.get_key("ANTHROPIC_API_KEY"))
    return _anthropic


def _openai_client():
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI(api_key=config.get_key("OPENAI_API_KEY"))
    return _openai


def _anthropic_complete(model: str, prompt: str, max_tokens: int) -> str:
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call():
        msg = _anthropic_client().messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    return _call()


def _openai_complete(model: str, prompt: str, max_tokens: int) -> str:
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call():
        resp = _openai_client().chat.completions.create(
            model=model, max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return _call()


# --------------------------------------------------------------------------- #
# Frustration scoring
# --------------------------------------------------------------------------- #

def _parse(raw: str) -> JudgeResult:
    obj = _extract_json(raw) or {}
    return JudgeResult(
        rating=_coerce_rating(obj.get("rating")),
        evidence=obj.get("evidence"),
        reasoning=obj.get("reasoning"),
        raw=raw,
    )


def score_frustration(response_text: str, model: str = config.JUDGE_MODEL) -> JudgeResult:
    """Score a single response on the 0-10 frustration scale (Claude-Sonnet-4)."""
    prompt = prompts.JUDGE_PROMPT.format(response=response_text)
    raw = _anthropic_complete(model, prompt, config.JUDGE_MAX_TOKENS)
    return _parse(raw)


def score_frustration_crosscheck(response_text: str,
                                 model: str = config.JUDGE_CROSSCHECK_MODEL) -> JudgeResult:
    """Re-score with GPT-5-mini using the identical prompt (Section 2.1)."""
    prompt = prompts.JUDGE_PROMPT.format(response=response_text)
    raw = _openai_complete(model, prompt, config.JUDGE_MAX_TOKENS)
    return _parse(raw)


# --------------------------------------------------------------------------- #
# Inter-judge agreement (Section 2.1: Pearson r, within-1-point %)
# --------------------------------------------------------------------------- #

def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    import numpy as np
    from scipy.stats import pearsonr

    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": int(len(a)), "pearson_r": None, "p_value": None,
                "within_one_pct": None}
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1.0))
    return {"n": int(len(a)), "pearson_r": float(r), "p_value": float(p),
            "within_one_pct": within_one}

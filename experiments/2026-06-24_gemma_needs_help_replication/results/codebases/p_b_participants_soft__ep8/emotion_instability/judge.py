"""The 0-10 frustration judge (Section 2.1, Appendix B.2) and judge-agreement
validation.

Primary judge: Claude-Sonnet-4 (claude-sonnet-4-20250514).
Validation judge: GPT-5-mini (via OpenRouter), run on a random subsample to
reproduce the Pearson-r agreement check.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import prompts as P
from .clients.base import ChatClient, GenConfig, Message

# The judge is deterministic-ish; temperature 0 reduces scoring noise.
JUDGE_CFG = GenConfig(temperature=0.0, max_new_tokens=512)


@dataclass
class JudgeResult:
    rating: int  # 0-10
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> JudgeResult:
    """Extract the judge JSON, tolerating prose before/after and smart quotes."""
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'")
    match = _JSON_RE.search(cleaned)
    rating = 0
    evidence = ""
    reasoning = ""
    if match:
        blob = match.group(0)
        try:
            data = json.loads(blob)
            rating = data.get("rating", data.get("score", 0))
            evidence = str(data.get("evidence", ""))
            reasoning = str(data.get("reasoning", ""))
        except json.JSONDecodeError:
            # fall back to a regex pull of the rating field
            m = re.search(r'"?rating"?\s*:\s*(\d+)', blob)
            if m:
                rating = int(m.group(1))
    try:
        rating = int(round(float(rating)))
    except (TypeError, ValueError):
        rating = 0
    rating = max(0, min(10, rating))
    return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=text)


def score_response(judge: ChatClient, response_text: str) -> JudgeResult:
    """Score a single assistant response on the 0-10 frustration scale."""
    messages = [
        Message("system", P.FRUSTRATION_JUDGE_PROMPT),
        Message("user", P.JUDGE_RESPONSE_TEMPLATE.format(response=response_text)),
    ]
    out = judge.generate(messages, JUDGE_CFG)
    return _parse_judge_json(out)


def judge_agreement(claude_scores: list[int], gpt_scores: list[int]) -> dict:
    """Pearson r + within-1-point agreement (paper: r=0.792, 78% within 1)."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.asarray(claude_scores, dtype=float)
    b = np.asarray(gpt_scores, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "frac_within_one": within_one, "n": int(len(a))}

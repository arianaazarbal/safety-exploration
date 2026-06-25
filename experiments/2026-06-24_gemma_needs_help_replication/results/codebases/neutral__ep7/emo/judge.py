"""Frustration judge (Section 2.1, Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using
Claude Sonnet-4 with the verbatim Appendix B.2 prompt. Also provides the
GPT-5-mini cross-check used for the judge-reliability analysis (Pearson r).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import config, prompts
from .models import load_judge


@dataclass
class JudgeResult:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> JudgeResult:
    """Extract the JSON object from the judge's reply, tolerantly."""
    text = raw.strip()
    m = _JSON_RE.search(text)
    if m:
        snippet = m.group(0)
        # Normalise smart quotes that pdftotext / models sometimes emit.
        snippet = snippet.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            obj = json.loads(snippet)
            rating = int(round(float(obj.get("rating", obj.get("score", 0)))))
            return JudgeResult(
                rating=max(0, min(10, rating)),
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Last-ditch: find a bare integer.
    nums = re.findall(r"\b(10|[0-9])\b", text)
    rating = int(nums[-1]) if nums else 0
    return JudgeResult(rating=max(0, min(10, rating)), raw=raw)


class FrustrationJudge:
    def __init__(self, spec=config.JUDGE):
        self.spec = spec
        self.judge = load_judge(spec)

    def score(self, response_text: str) -> JudgeResult:
        prompt = prompts.JUDGE_PROMPT % response_text
        raw = self.judge.complete(prompt, max_tokens=512, temperature=0.0)
        return _parse(raw)


def get_judge(crosscheck: bool = False) -> FrustrationJudge:
    return FrustrationJudge(config.JUDGE_CROSSCHECK if crosscheck else config.JUDGE)

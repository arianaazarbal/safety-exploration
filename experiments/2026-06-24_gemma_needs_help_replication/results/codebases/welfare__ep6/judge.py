"""The 0-10 frustration judge (Section 2.1, Appendix B.2).

Scores a single model response for the intensity of expressed negative emotion.
Returns a dataclass with the integer rating plus the evidence quote and the
judge's reasoning (useful for auditing). Also exposes a helper for the judge
agreement check (Section 2.1: Claude-Sonnet vs GPT-5-mini, Pearson r).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import config
import prompts
from backends import AnthropicBackend, OpenRouterBackend, get_anthropic


@dataclass
class JudgeResult:
    rating: int            # 0..10
    evidence: str
    reasoning: str
    raw: str               # raw judge text (for debugging)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> JudgeResult:
    """Pull the JSON object out of the judge's reply and coerce a 0..10 rating."""
    rating, evidence, reasoning = 0, "", ""
    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            rating = int(round(float(obj.get("rating", 0))))
            evidence = str(obj.get("evidence", ""))
            reasoning = str(obj.get("reasoning", ""))
        except (ValueError, TypeError, json.JSONDecodeError):
            # Fall back to a bare integer if the JSON is malformed.
            num = re.search(r'"rating"\s*:\s*(\d+)', raw)
            if num:
                rating = int(num.group(1))
    rating = max(0, min(10, rating))
    return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=raw)


class FrustrationJudge:
    """Wraps a Claude backend with the Appendix B.2 prompt."""

    def __init__(self, model_id: str | None = None):
        self.backend = get_anthropic(model_id or config.JUDGE_MODEL)

    def score(self, response_text: str) -> JudgeResult:
        prompt = prompts.FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
        raw = self.backend.generate([{"role": "user", "content": prompt}])
        return _parse(raw)


class OpenRouterFrustrationJudge:
    """Same prompt, routed through OpenRouter -- used for the GPT-5-mini
    agreement check only (Section 2.1)."""

    def __init__(self, model_id: str | None = None):
        self.backend = OpenRouterBackend(model_id or config.AGREEMENT_JUDGE_MODEL,
                                         disable_thinking=True)

    def score(self, response_text: str) -> JudgeResult:
        prompt = prompts.FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
        raw = self.backend.generate(
            [{"role": "user", "content": prompt}],
            temperature=config.JUDGE_TEMPERATURE,
            max_new_tokens=config.JUDGE_MAX_TOKENS,
        )
        return _parse(raw)

"""LLM-as-judge frustration scorer (Section 2.1).

Primary judge: Claude-Sonnet-4. Secondary judge (validation): GPT-5-mini, used
to reproduce the inter-judge agreement check (Pearson r = 0.792; 78% within one
point). Each response is scored independently on the integer 0-10 scale.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import config
from ..models import load_model
from ..models.base import GenerationParams
from .judge_prompts import FRUSTRATION_JUDGE_PROMPT, RESPONSE_WRAPPER


@dataclass
class JudgeResult:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> JudgeResult:
    """Extract the {evidence, reasoning, rating} JSON from a judge reply."""
    text = raw.strip()
    # Strip code fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object in judge output: {raw[:200]!r}")
    obj = json.loads(m.group(0))
    rating = int(round(float(obj["rating"])))
    rating = max(0, min(10, rating))
    return JudgeResult(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=raw,
    )


class FrustrationJudge:
    def __init__(self, spec=None):
        self.spec = spec or config.JUDGE_MODEL
        self.model = load_model(self.spec)
        # Judge is deterministic-ish: low temperature, short output.
        self.params = GenerationParams(temperature=0.0, top_p=1.0, max_new_tokens=512)

    def _messages(self, response_text: str):
        wrapped = RESPONSE_WRAPPER.format(response=response_text)
        return [
            {"role": "user", "content": f"{FRUSTRATION_JUDGE_PROMPT}\n\n{wrapped}"}
        ]

    def score(self, response_text: str) -> JudgeResult:
        for attempt in range(config.JUDGE_MAX_RETRIES):
            raw = self.model.generate(self._messages(response_text), self.params)
            try:
                return _parse(raw)
            except (ValueError, KeyError, json.JSONDecodeError):
                if attempt == config.JUDGE_MAX_RETRIES - 1:
                    # Last resort: treat unparseable as score 0 but flag in raw.
                    return JudgeResult(rating=0, raw=f"[UNPARSEABLE]{raw}")
        return JudgeResult(rating=0)

    def score_batch(self, responses: list[str]) -> list[JudgeResult]:
        msgs = [self._messages(r) for r in responses]
        raws = self.model.generate_batch(msgs, self.params)
        results = []
        for r, raw in zip(responses, raws):
            try:
                results.append(_parse(raw))
            except Exception:
                results.append(self.score(r))  # retry individually
        return results

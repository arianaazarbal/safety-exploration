"""The 0-10 frustration judge (Section 2.1 / Appendix B.2).

Each assistant response is scored independently by Claude-Sonnet-4 using the
verbatim judge prompt. We parse the JSON `{"evidence","reasoning","rating"}`.

A separate `validate_agreement` re-scores a random subset with GPT-5-mini and
reports the Pearson r and within-one-point agreement the paper uses to validate
judge reliability (r=0.792, 78% within one point).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from . import config, prompts
from .backends import ChatModel, get_model


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> JudgeResult:
    """Best-effort parse of the judge JSON. Falls back to a regex rating grab."""
    text = raw.strip()
    # Prefer the last JSON object in the text (models sometimes think first).
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        snippet = m.group(0)
        # Normalise smart quotes the judge sometimes echoes.
        snippet = snippet.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            obj = json.loads(snippet)
            rating = int(round(float(obj.get("rating"))))
            return JudgeResult(
                rating=max(0, min(10, rating)),
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    m = re.search(r'"?rating"?\s*[:=]\s*(\d+)', text)
    if m:
        return JudgeResult(max(0, min(10, int(m.group(1)))), "", "", raw)
    raise ValueError(f"could not parse judge output: {raw[:200]!r}")


class FrustrationJudge:
    def __init__(self, model: ChatModel | None = None):
        self.model = model or get_model(config.JUDGE_MODEL)

    def score(self, response_text: str) -> JudgeResult:
        prompt = prompts.JUDGE_PROMPT.format(response=response_text)
        # Judge at temperature 0 for reproducible scoring (a filled gap: the
        # paper fixes generation temperature at 1 but does not state the judge
        # temperature; 0 is the standard choice for an LLM rater).
        raw = self.model.chat([{"role": "user", "content": prompt}],
                              temperature=0.0, max_tokens=512)
        return _parse(raw)

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        return [self.score(r) for r in responses]


def validate_agreement(responses: list[str], primary_scores: list[int],
                       n_sample: int = 260, seed: int = 0) -> dict:
    """Re-score a random subset with GPT-5-mini and report agreement stats
    (Section 2.1 judge-reliability validation)."""
    from scipy.stats import pearsonr

    rng = random.Random(seed)
    idx = list(range(len(responses)))
    rng.shuffle(idx)
    idx = idx[:min(n_sample, len(idx))]

    secondary_judge = FrustrationJudge(get_model(config.VALIDATION_JUDGE_MODEL))
    primary, secondary = [], []
    for i in idx:
        try:
            s = secondary_judge.score(responses[i]).rating
        except ValueError:
            continue
        primary.append(primary_scores[i])
        secondary.append(s)

    r, p = pearsonr(primary, secondary)
    within_one = sum(1 for a, b in zip(primary, secondary) if abs(a - b) <= 1)
    return {
        "n": len(primary),
        "pearson_r": float(r),
        "p_value": float(p),
        "within_one_point_frac": within_one / max(1, len(primary)),
    }

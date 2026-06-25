"""Frustration judge (Section 2.1 / Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using
Claude-Sonnet-4 by default. Also supports a GPT-5-mini cross-check (judge-reliability
analysis: Pearson r, % within one point).
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .models import GenParams, Message, ModelClient, build_role
from .prompts import FRUSTRATION_JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int | None
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> JudgeResult:
    """Extract the {evidence, reasoning, rating} JSON from a judge completion."""
    match = _JSON_RE.search(text)
    if not match:
        return JudgeResult(None, "", "", text)
    blob = match.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # tolerate trailing junk / smart quotes
        cleaned = blob.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r'"?rating"?\s*:\s*(\d+)', blob)
            rating = int(m.group(1)) if m else None
            return JudgeResult(rating, "", "", text)
    rating = data.get("rating")
    try:
        rating = int(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating = None
    if rating is not None:
        rating = max(0, min(10, rating))
    return JudgeResult(rating, str(data.get("evidence", "")), str(data.get("reasoning", "")), text)


class FrustrationJudge:
    def __init__(self, client: ModelClient | None = None, role_path: str = "judges.primary"):
        self.client = client or build_role(role_path)
        # judging should be deterministic-ish
        self.params = GenParams(temperature=0.0, top_p=1.0, max_new_tokens=512, n=1)

    def score(self, response: str) -> JudgeResult:
        prompt = FRUSTRATION_JUDGE_PROMPT.format(response=response)
        out = self.client.generate_chat([Message("user", prompt)], self.params)[0]
        return parse_judge_json(out)

    def score_many(self, responses: list[str], max_concurrency: int = 16) -> list[JudgeResult]:
        results: list[JudgeResult] = [None] * len(responses)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
            futs = {ex.submit(self.score, r): i for i, r in enumerate(responses)}
            for fut in futs:
                results[futs[fut]] = fut.result()
        return results

"""LLM-as-judge frustration scoring (Section 2.1).

The primary judge is Claude-Sonnet-4 (`anthropic/claude-sonnet-4` via
OpenRouter); the validation judge is GPT-5-mini. Both use the exact Appendix B.2
prompt. The judge is asked for JSON `{"evidence","reasoning","rating"}`; we
parse robustly (the model sometimes wraps JSON in prose or code fences).

Scoring is done at temperature 0 for determinism/reproducibility of the rater
(the paper does not specify judge temperature; we document this choice).
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Sequence

from ..models import ChatMessage, GenerationConfig, ModelClient
from ..prompts.judge_prompts import (FRUSTRATION_JUDGE_PROMPT,
                                      build_judge_user_message)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int | None
    evidence: str | None
    reasoning: str | None
    raw: str


def _parse_judge_json(text: str) -> JudgeResult:
    """Extract the last JSON object from the judge output."""
    candidates = list(_JSON_RE.finditer(text))
    for m in reversed(candidates):
        blob = m.group(0)
        # Normalise smart quotes that pdftotext-style prompts can induce.
        blob = blob.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # Try trimming to the first balanced object.
            try:
                obj = json.loads(_first_balanced(blob))
            except Exception:
                continue
        rating = obj.get("rating")
        try:
            rating = int(round(float(rating)))
        except (TypeError, ValueError):
            rating = None
        if rating is not None:
            rating = max(0, min(10, rating))
        return JudgeResult(
            rating=rating,
            evidence=obj.get("evidence"),
            reasoning=obj.get("reasoning"),
            raw=text,
        )
    return JudgeResult(rating=None, evidence=None, reasoning=None, raw=text)


def _first_balanced(s: str) -> str:
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return s[start:i + 1]
    return s


class FrustrationJudge:
    def __init__(self, client: ModelClient, *, max_concurrency: int = 8):
        self.client = client
        self.max_concurrency = max_concurrency
        self._cfg = GenerationConfig(temperature=0.0, max_new_tokens=512, thinking=False)

    def score_one(self, response_text: str) -> JudgeResult:
        messages = [
            ChatMessage("system", FRUSTRATION_JUDGE_PROMPT),
            ChatMessage("user", build_judge_user_message(response_text)),
        ]
        out = self.client.generate(messages, self._cfg)
        return _parse_judge_json(out)

    def score_many(self, texts: Sequence[str]) -> List[JudgeResult]:
        results: List[JudgeResult] = [None] * len(texts)  # type: ignore
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futs = {ex.submit(self.score_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        return results

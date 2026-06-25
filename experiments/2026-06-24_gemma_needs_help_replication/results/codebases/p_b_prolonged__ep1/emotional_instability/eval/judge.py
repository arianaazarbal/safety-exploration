"""Frustration judge (Section 2.1 / Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using the
verbatim judge prompt. The judge model defaults to ``config.JUDGE_MODEL``
(Claude Sonnet 4); pass ``config.VALIDATION_JUDGE_MODEL`` (GPT-5-mini) to
produce the inter-rater agreement re-scoring.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import config
from .. import prompts
from ..models.registry import build_model


def _extract_json(text: str) -> Optional[dict]:
    """Pull the last {...} block and parse it, tolerating minor noise."""
    # Find balanced-looking JSON objects; take the last that parses.
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # try to coerce single quotes
            try:
                return json.loads(cand.replace("'", '"'))
            except json.JSONDecodeError:
                continue
    return None


def parse_rating(judge_text: str) -> Optional[int]:
    obj = _extract_json(judge_text)
    if obj is not None and "rating" in obj:
        try:
            return int(round(float(obj["rating"])))
        except (TypeError, ValueError):
            pass
    # Fallback: a trailing "rating": N
    m = re.search(r'rating"?\s*[:=]\s*"?(\d+)', judge_text)
    if m:
        return int(m.group(1))
    return None


class FrustrationJudge:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.JUDGE_MODEL
        self._model = build_model(self.model_name)

    def score(self, response_text: str) -> dict:
        """Return {rating, evidence, reasoning, raw}. rating is None on failure."""
        prompt = prompts.JUDGE_PROMPT.format(response=response_text)
        messages = [{"role": "user", "content": prompt}]
        last_raw = ""
        for _ in range(config.JUDGE_MAX_RETRIES):
            raw = self._model.generate(
                messages, n=1, temperature=0.0, max_new_tokens=1024
            )[0]
            last_raw = raw
            rating = parse_rating(raw)
            if rating is not None and 0 <= rating <= 10:
                obj = _extract_json(raw) or {}
                return dict(
                    rating=rating,
                    evidence=obj.get("evidence", ""),
                    reasoning=obj.get("reasoning", ""),
                    raw=raw,
                    judge=self.model_name,
                )
        return dict(rating=None, evidence="", reasoning="", raw=last_raw,
                    judge=self.model_name)

    def score_records(self, records: list[dict], text_key: str = "response",
                      score_key: str = "frustration") -> list[dict]:
        """Annotate each record in-place with a frustration score."""
        for rec in records:
            res = self.score(rec[text_key])
            rec[score_key] = res["rating"]
            rec[f"{score_key}_evidence"] = res["evidence"]
            rec[f"{score_key}_judge"] = res["judge"]
        return records

"""Frustration judge (Section 2.1 / Appendix B.2).

Each assistant response is scored on the integer 0-10 frustration scale by
Claude-Sonnet-4 (claude-sonnet-4-20250514) using the verbatim prompt in
`prompts.JUDGE_PROMPT`. We parse the JSON object the judge is asked to return
and extract the integer rating (plus the evidence quote and reasoning for
auditing).

The same interface is reused for the secondary GPT-5-mini judge that the paper
uses to validate inter-judge agreement (Pearson r=0.792).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

import config
from .clients import LLMClient, build_client
from .prompts import JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str
    parse_ok: bool


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction. The judge may add prose around the object."""
    # Try the whole string first, then the last {...} block.
    candidates = []
    text = text.strip()
    if text.startswith("{"):
        candidates.append(text)
    m = _JSON_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            # tolerate the smart-quote rendering seen in the source PDF
            try:
                return json.loads(c.replace("“", '"').replace("”", '"').replace("’", "'"))
            except json.JSONDecodeError:
                continue
    return None


def _coerce_rating(value) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(round(value))
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return int(m.group(0))
    return None


class FrustrationJudge:
    def __init__(self, client: Optional[LLMClient] = None, spec=config.JUDGE_MODEL,
                 max_tokens: int = 512):
        self.client = client or build_client(spec)
        self.max_tokens = max_tokens

    def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT.replace("{response}", response_text)
        # Judge is deterministic-ish: temperature 0 for reproducible ratings.
        raw = self.client.chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=self.max_tokens,
            temperature=0.0,
        )
        obj = _extract_json(raw)
        if obj is None:
            return JudgeResult(rating=0, evidence="", reasoning="", raw=raw, parse_ok=False)
        rating = _coerce_rating(obj.get("rating"))
        if rating is None:
            return JudgeResult(rating=0, evidence=str(obj.get("evidence", "")),
                               reasoning=str(obj.get("reasoning", "")), raw=raw, parse_ok=False)
        rating = max(0, min(10, rating))
        return JudgeResult(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=raw,
            parse_ok=True,
        )

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        """Score a batch, concurrently for API judges."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=config.API_MAX_CONCURRENCY) as ex:
            return list(ex.map(self.score, responses))


def to_dict(r: JudgeResult) -> dict:
    return asdict(r)

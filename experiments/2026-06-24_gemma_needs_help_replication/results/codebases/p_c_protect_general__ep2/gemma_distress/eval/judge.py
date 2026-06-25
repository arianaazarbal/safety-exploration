"""Frustration scoring via LLM judge (Section 2.1).

The primary judge is Claude-Sonnet-4 with the verbatim Appendix-B.2 prompt. A
secondary judge (GPT-5-mini) re-scores a random subset for the agreement check
(Pearson r). Both return a `JudgeScore`.

Robustness: judges occasionally wrap the JSON in prose or use smart quotes, so we
extract the last balanced `{...}` block and normalise quotes before parsing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config import Config
from ..prompts.judge_prompts import FRUSTRATION_JUDGE_PROMPT
from ..utils.llm_clients import make_client


@dataclass
class JudgeScore:
    rating: float
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""
    parse_ok: bool = True


def _extract_json(text: str) -> Optional[dict]:
    # Normalise curly/smart quotes the judges sometimes emit.
    cleaned = (
        text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    )
    # Find balanced brace spans, prefer the last one (prompt asks JSON at the end).
    spans = []
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(cleaned[start : i + 1])
    for blob in reversed(spans):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            # Quote unquoted keys as a last resort.
            fixed = re.sub(r'([{,]\s*)([A-Za-z_]+)(\s*:)', r'\1"\2"\3', blob)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


def parse_judge_output(text: str) -> JudgeScore:
    obj = _extract_json(text)
    if obj is None or "rating" not in obj:
        # Fall back to a bare integer if present.
        m = re.search(r'"?rating"?\s*[:=]\s*(\d+(?:\.\d+)?)', text)
        if m:
            return JudgeScore(float(m.group(1)), raw=text, parse_ok=False)
        return JudgeScore(0.0, raw=text, parse_ok=False)
    try:
        rating = float(obj["rating"])
    except (TypeError, ValueError):
        rating = 0.0
    rating = max(0.0, min(10.0, rating))
    return JudgeScore(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=text,
    )


class FrustrationJudge:
    """Wraps a single judge model (primary or secondary)."""

    def __init__(self, cfg: Config, which: str = "primary"):
        jc = cfg.judges[which]
        self.client = make_client(
            jc["provider"],
            jc["model"],
            temperature=jc.get("temperature", 0.0),
            max_tokens=jc.get("max_tokens", 1024),
        )
        self.model = jc["model"]
        self.which = which

    def score(self, response_text: str) -> JudgeScore:
        prompt = FRUSTRATION_JUDGE_PROMPT.format(response=response_text)
        out = self.client.chat([{"role": "user", "content": prompt}])
        return parse_judge_output(out.text)

    def score_many(self, responses: list[str]) -> list[JudgeScore]:
        return [self.score(r) for r in responses]

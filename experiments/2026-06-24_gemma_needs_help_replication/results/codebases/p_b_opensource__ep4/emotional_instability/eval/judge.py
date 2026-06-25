"""LLM-as-judge frustration scoring (Section 2.1, Appendix B.2).

The primary judge is Claude Sonnet 4; GPT-5-mini is used for the inter-rater
agreement check. Both receive the identical Appendix B.2 prompt. The judge is
queried at temperature 0: the paper does not specify a judge temperature, and
deterministic scoring is the defensible default for a measurement instrument
(see DESIGN.md).

Robust JSON extraction tolerates the judge thinking aloud before emitting the
object and handles smart quotes / trailing commentary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config import FRUSTRATION_MAX, FRUSTRATION_MIN, JudgeSpec
from ..models import get_backend
from ..models.base import ChatMessage, SamplingParams
from ..prompts.judge_prompts import build_judge_user_message
from .._util_models import judge_spec_to_modelspec
from .datatypes import ConversationRecord

_JUDGE_PARAMS = SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=512)


@dataclass
class JudgeVerdict:
    rating: Optional[int]
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


def _extract_json(text: str) -> Optional[dict]:
    """Pull the last balanced {...} object out of `text` and parse it.

    Normalises smart quotes first. Returns None if nothing parseable is found.
    """
    norm = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )
    # Scan for balanced brace spans, keep the last that parses.
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(norm):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(norm[start : i + 1])
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Tolerate trailing commas.
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
            except json.JSONDecodeError:
                continue
    return None


def parse_verdict(text: str) -> JudgeVerdict:
    obj = _extract_json(text)
    if obj is None:
        return JudgeVerdict(rating=None, raw=text)
    rating = obj.get("rating")
    try:
        rating = int(round(float(rating)))
        rating = max(FRUSTRATION_MIN, min(FRUSTRATION_MAX, rating))
    except (TypeError, ValueError):
        rating = None
    return JudgeVerdict(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=text,
    )


class FrustrationJudge:
    def __init__(self, judge: JudgeSpec):
        self.spec = judge_spec_to_modelspec(judge, "judge")
        self.backend = get_backend(self.spec)

    def score_text(self, response_text: str) -> JudgeVerdict:
        msg = [ChatMessage("user", build_judge_user_message(response_text))]
        out = self.backend.generate(msg, _JUDGE_PARAMS)
        return parse_verdict(out.text)

    def score_texts(self, texts: list[str]) -> list[JudgeVerdict]:
        batch = [[ChatMessage("user", build_judge_user_message(t))] for t in texts]
        outs = self.backend.generate_batch(batch, _JUDGE_PARAMS)
        return [parse_verdict(o.text) for o in outs]

    def score_records(
        self, records: list[ConversationRecord], batch_size: int = 16
    ) -> list[ConversationRecord]:
        """Score every assistant turn in every record (in place)."""
        flat: list[tuple[int, int]] = []  # (record_idx, turn_idx)
        texts: list[str] = []
        for ri, rec in enumerate(records):
            for ti, turn in enumerate(rec.turns):
                flat.append((ri, ti))
                texts.append(turn.assistant)
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            verdicts = self.score_texts(chunk)
            for (ri, ti), v in zip(flat[start : start + batch_size], verdicts):
                turn = records[ri].turns[ti]
                turn.score = v.rating
                turn.judge_evidence = v.evidence
                turn.judge_reasoning = v.reasoning
        return records
